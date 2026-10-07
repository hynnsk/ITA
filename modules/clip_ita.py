"""CLIP encoders with Intrinsic Temporal Adaptation (ITA).

CLIP components adapted from https://github.com/openai/CLIP.
"""

import logging
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, lora_rank=8, lora_alpha=16.0, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        if self.in_proj_weight is None:
            raise ValueError("LoRA requires a packed in_proj_weight.")

        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank
        self.lora_A = nn.Parameter(torch.empty(lora_rank, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3 * embed_dim, lora_rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merged_in_proj_weight(self):
        lora_delta = torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        lora_delta = lora_delta.to(
            dtype=self.in_proj_weight.dtype, device=self.in_proj_weight.device
        )
        return self.in_proj_weight + lora_delta

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        dropout_p = self.dropout if self.training else 0.0
        common_kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.merged_in_proj_weight(),
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=dropout_p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
        )
        try:
            return F.multi_head_attention_forward(
                **common_kwargs,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        except TypeError:
            try:
                return F.multi_head_attention_forward(
                    **common_kwargs,
                    average_attn_weights=average_attn_weights,
                )
            except TypeError:
                return F.multi_head_attention_forward(**common_kwargs)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, args, group_size=1, attn_mask=None):
        super().__init__()
        self.attn = LoRAMultiheadAttention(
            d_model, n_head, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.num_frames = args.Nf
        self.group_size = group_size
        self.relative_temporal_bias = (
            nn.Parameter(torch.zeros(2 * group_size - 1)) if group_size > 1 else None
        )

    def attention(self, x, mask=None):
        if mask is None and self.attn_mask is not None:
            mask = self.attn_mask(x.size(0))
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def grouped_frame_attention(self, x):
        tokens, batch_frames, dim = x.shape
        group_size = self.group_size
        if group_size == 1:
            return self.attention(x)
        if batch_frames % self.num_frames or self.num_frames % group_size:
            raise ValueError("Frame groups must divide the number of frames")
        batch_size = batch_frames // self.num_frames
        num_groups = self.num_frames // group_size
        grouped = x.permute(1, 0, 2).reshape(batch_size, self.num_frames, tokens, dim)
        grouped = grouped.reshape(batch_size, num_groups, group_size, tokens, dim)
        grouped = grouped.reshape(
            batch_size * num_groups, group_size * tokens, dim
        ).permute(1, 0, 2)
        frame_ids = torch.arange(group_size, device=x.device).repeat_interleave(tokens)
        offsets = frame_ids[:, None] - frame_ids[None, :] + group_size - 1
        mask = self.relative_temporal_bias[offsets]
        output = self.attention(grouped, mask).permute(1, 0, 2)
        output = output.reshape(batch_size, num_groups, group_size, tokens, dim)
        return output.reshape(batch_frames, tokens, dim).permute(1, 0, 2)

    def forward(self, x):
        x = x + self.grouped_frame_attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, args, frame_groups=None, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        groups = frame_groups if frame_groups is not None else [1] * layers
        if len(groups) != layers:
            raise ValueError("frame_attn_groups must match the number of visual layers")
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, args, groups[i], attn_mask)
                for i in range(layers)
            ]
        )

    def forward(self, x):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim, args
    ):
        super().__init__()
        self.num_frames = args.Nf
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.frame_embedding = nn.Parameter(torch.zeros(args.Nf, 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width, layers, heads, args, frame_groups=args.frame_attn_groups
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        if x.shape[0] % self.num_frames:
            raise ValueError("Visual batch dimension must be divisible by Nf")
        batch_size = x.shape[0] // self.num_frames
        frame_embedding = self.frame_embedding.to(dtype=x.dtype, device=x.device)
        frame_embedding = frame_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = x + frame_embedding.reshape(x.shape[0], 1, x.size(-1))
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2)[:, :1, :]
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        args,
    ):
        super().__init__()
        self.context_length = context_length
        self.visual = VisualTransformer(
            image_resolution,
            vision_patch_size,
            vision_width,
            vision_layers,
            vision_width // 64,
            embed_dim,
            args,
        )
        self.transformer = Transformer(
            transformer_width,
            transformer_layers,
            transformer_heads,
            args,
            attn_mask=self.build_attention_mask,
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self, context_length):
        # Text attention is causal; PyTorch uses an additive mask.
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        hidden = self.visual(image.type(self.dtype))
        return self.visual.ln_post(hidden) @ self.visual.proj

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding[: x.size(1)].type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection
        # CLIP's end-of-text token has the largest token ID.
        return hidden[
            torch.arange(hidden.shape[0], device=text.device), text.argmax(dim=-1)
        ]


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_clip_model(state_dict, args):
    """Infer CLIP dimensions from the pretrained ViT weights."""
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = sum(
        k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        for k in state_dict
    )
    patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_layers = len(
        {k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks.")}
    )
    model = CLIP(
        state_dict["text_projection"].shape[1],
        patch_size * grid_size,
        vision_layers,
        vision_width,
        patch_size,
        state_dict["positional_embedding"].shape[0],
        state_dict["token_embedding.weight"].shape[0],
        transformer_width,
        transformer_width // 64,
        transformer_layers,
        args,
    )
    # Preserve the original pretrained-weight conversion before AMP training.
    convert_weights(model)
    logging.info(
        "CLIP: %d visual layers, %d text layers, %d frames",
        vision_layers,
        transformer_layers,
        args.Nf,
    )
    return model


def load_clip_state_dict(pretrained_dir):
    model_path = Path(pretrained_dir) / "ViT-B-32.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"CLIP weights not found: {model_path}. See README.md.")
    try:
        return torch.jit.load(str(model_path), map_location="cpu").eval().state_dict()
    except RuntimeError:
        return torch.load(model_path, map_location="cpu")
