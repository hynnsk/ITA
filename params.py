"""Arguments used by ITA training and evaluation."""

import argparse
import json
from pathlib import Path


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ITA for Partially Relevant Video Retrieval"
    )
    parser.add_argument("--do_train", type=int, choices=[0, 1], default=1)
    parser.add_argument("--do_eval", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["activity", "tvr", "charades", "qvhighlights"],
    )
    parser.add_argument("--video_dir", required=True)
    parser.add_argument(
        "--prvr_train_annos", help="Training annotations in JSONL format"
    )
    parser.add_argument("--prvr_val_annos", required=True)
    parser.add_argument("--pretrained_dir", default="CLIP_weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resume", help="ITA checkpoint for resuming training or evaluation"
    )
    parser.add_argument("--num_thread_reader", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--batch_size_val", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_display", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["amp", "fp32"], default="amp")
    parser.add_argument("--max_words", type=int, default=64)
    parser.add_argument("--Nf", type=int, default=32, help="Sampled frames per video")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument(
        "--frame_attn_groups",
        default="[1,1,1,1,1,1,1,1,1,1,2,4]",
        help="JSON list of frame group sizes for CLIP's 12 visual layers",
    )
    parser.add_argument(
        "--top1_frame_attention_heads",
        type=int,
        default=8,
        help="Number of heads in the global frame attention layer",
    )
    parser.add_argument("--nce_frame_topk", type=int, default=4)
    parser.add_argument("--nce_frame_topk_temperature", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.do_train == args.do_eval:
        parser.error(
            "Select exactly one mode: --do_train 1 --do_eval 0 or --do_train 0 --do_eval 1"
        )
    if args.do_train and not args.prvr_train_annos:
        parser.error("--prvr_train_annos is required for training")
    if args.do_eval and not args.resume:
        parser.error("--resume is required for evaluation")
    try:
        args.frame_attn_groups = json.loads(args.frame_attn_groups)
    except json.JSONDecodeError:
        parser.error("--frame_attn_groups must be a JSON list")
    if (
        not isinstance(args.frame_attn_groups, list)
        or len(args.frame_attn_groups) != 12
        or any(type(g) is not int or g < 1 for g in args.frame_attn_groups)
    ):
        parser.error("--frame_attn_groups must contain 12 positive integers")
    positive = (
        "epochs",
        "batch_size",
        "batch_size_val",
        "lr",
        "eps",
        "clip_grad_norm",
        "n_display",
        "Nf",
        "lora_rank",
        "lora_alpha",
        "top1_frame_attention_heads",
        "nce_frame_topk",
        "nce_frame_topk_temperature",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if any(args.Nf % g for g in args.frame_attn_groups):
        parser.error("Each frame group size must divide --Nf")
    if args.nce_frame_topk > args.Nf:
        parser.error("--nce_frame_topk must not exceed --Nf")
    if 512 % args.top1_frame_attention_heads:
        parser.error("--top1_frame_attention_heads must divide 512")
    if not 2 <= args.max_words <= 77:
        parser.error("--max_words must be between 2 and CLIP's context length of 77")
    if not 0 <= args.warmup_proportion < 1:
        parser.error("--warmup_proportion must be in [0, 1)")
    if args.num_thread_reader < 0 or args.wd < 0:
        parser.error("Worker count and weight decay must be nonnegative")
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error("AdamW beta values must be in [0, 1)")
    return args


def save_hp_to_json(directory, args):
    with (Path(directory) / "hparams_train.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=4, sort_keys=True)
