"""Intrinsic Temporal Adaptation for partially relevant video retrieval."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from .clip_ita import build_clip_model, load_clip_state_dict


class ITA(nn.Module):
    def __init__(self, clip_state_dict, args):
        super().__init__()
        self.clip = build_clip_model(clip_state_dict, args)
        self.nce_frame_topk = args.nce_frame_topk
        self.nce_frame_topk_temperature = args.nce_frame_topk_temperature
        embed_dim = self.clip.text_projection.size(1)
        if embed_dim % args.top1_frame_attention_heads:
            raise ValueError(
                "top1_frame_attention_heads must divide the CLIP embedding dimension"
            )
        # Keep weight names compatible with existing ITA checkpoints.
        self.top1_frame_attention = nn.MultiheadAttention(
            embed_dim, args.top1_frame_attention_heads, dropout=0.0
        )
        for name, param in self.clip.named_parameters():
            param.requires_grad_(
                any(
                    part in name
                    for part in ("lora_", "frame_embedding", "relative_temporal_bias")
                )
            )

    @classmethod
    def from_pretrained(cls, args):
        state_dict = load_clip_state_dict(args.pretrained_dir)
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k not in {"input_resolution", "context_length", "vocab_size"}
        }
        model = cls(state_dict, args)
        incompatible = model.clip.load_state_dict(state_dict, strict=False)
        unexpected_missing = [
            k
            for k in incompatible.missing_keys
            if not any(
                part in k
                for part in ("lora_", "frame_embedding", "relative_temporal_bias")
            )
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(f"Incompatible CLIP weights: {incompatible}")
        logging.info("Initialized ITA from CLIP ViT-B/32")
        return model

    def forward(self, batch):
        sequence_output = self.get_sequence_output(batch["text_ids"])
        visual_output = self.get_visual_output(batch["video"])
        logits = self.get_prvr_similarity_logits(sequence_output, visual_output)
        text_labels = batch["text_labels"]
        v2t = {}
        for text_id, video_id in enumerate(text_labels.tolist()):
            v2t.setdefault(video_id, []).append(text_id)
        num_texts, num_videos = logits.shape
        rows = torch.arange(num_texts, device=logits.device)
        t2v_numerator = torch.logsumexp(logits[rows, text_labels].unsqueeze(1), dim=1)
        t2v_loss = (torch.logsumexp(logits, dim=1) - t2v_numerator).mean()
        v2t_numerator = torch.zeros(num_videos, device=logits.device)
        v2t_denominator = torch.zeros(num_videos, device=logits.device)
        for video_id, text_ids in v2t.items():
            v2t_numerator[video_id] = torch.logsumexp(logits[text_ids, video_id], dim=0)
            v2t_denominator[video_id] = torch.logsumexp(logits[:, video_id], dim=0)
        v2t_loss = (v2t_denominator - v2t_numerator).mean()
        loss = (t2v_loss + v2t_loss) / 2
        return {
            "loss": loss,
            "sim_loss": loss,
            "t2v_loss": t2v_loss,
            "v2t_loss": v2t_loss,
            "sequence_output": sequence_output,
            "visual_output": visual_output,
        }

    def get_sequence_output(self, text_ids):
        return self.clip.encode_text(text_ids).float().unsqueeze(1)

    def get_visual_output(self, video):
        batch_size, num_frames, channels, height, width = video.shape
        video = video.float().reshape(-1, channels, height, width)
        return self.clip.encode_image(video).reshape(batch_size, num_frames, -1)

    @torch.no_grad()
    def encode_text(self, text_ids):
        return self.get_sequence_output(text_ids)

    @torch.no_grad()
    def encode_video(self, video):
        return self.get_visual_output(video)

    def get_all_fused_frame_features(self, visual_output):
        """Refine class-token embeddings with global frame attention."""
        dtype = next(self.top1_frame_attention.parameters()).dtype
        query = visual_output.permute(1, 0, 2).to(dtype=dtype)
        fused = self.top1_frame_attention(query, query, query, need_weights=False)[0]
        fused = (fused + query).permute(1, 0, 2).float()
        return F.normalize(fused, dim=-1)

    def get_nce_frame_topk_logits(self, sequence_output, visual_output, frame_scores):
        """Affinity-weighted aggregation of the query's top-k frames."""
        topk = min(self.nce_frame_topk, frame_scores.size(-1))
        _, indices = torch.topk(frame_scores.detach(), k=topk, dim=-1)
        visual_by_text = visual_output.unsqueeze(0).expand(
            frame_scores.size(0), -1, -1, -1
        )
        gather_indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, visual_output.size(-1)
        )
        selected_visual = torch.gather(visual_by_text, dim=2, index=gather_indices)
        selected_scores = torch.gather(frame_scores, dim=-1, index=indices)
        weights = F.softmax(selected_scores / self.nce_frame_topk_temperature, dim=-1)
        weighted_visual = (selected_visual * weights.unsqueeze(-1)).sum(dim=2)
        weighted_visual = F.normalize(weighted_visual, p=2, dim=-1)
        return torch.einsum("td,tvd->tv", sequence_output, weighted_visual)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def get_prvr_similarity_logits(
        self,
        sequence_output,
        visual_output,
        visual_output_is_fused=False,
        force_frame_max=False,
    ):
        sequence_output = F.normalize(
            sequence_output.contiguous().squeeze(1).float(), dim=-1
        )
        visual_output = visual_output.contiguous()
        if visual_output_is_fused:
            fused = F.normalize(visual_output.float(), dim=-1)
        else:
            fused = self.get_all_fused_frame_features(visual_output)
        frame_scores = torch.einsum("td,vkd->tvk", sequence_output, fused)
        if force_frame_max or self.nce_frame_topk == 1:
            scores = frame_scores.max(dim=-1).values
        else:
            scores = self.get_nce_frame_topk_logits(
                sequence_output, fused, frame_scores
            )
        return self.clip.logit_scale.exp().to(sequence_output.device) * scores
