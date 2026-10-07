#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Override paths with environment variables or command-line arguments.
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
ANNOTATION_DIR="${ANNOTATION_DIR:-$ROOT_DIR/annotations}"
PRETRAINED_DIR="${PRETRAINED_DIR:-$ROOT_DIR/CLIP_weights}"
VIDEO_DIR="${VIDEO_DIR:-$DATA_ROOT/ActivityNet_compressed}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/ita_activitynet}"

exec "${PYTHON:-python}" main_prvr.py \
    --do_train 1 \
    --do_eval 0 \
    --datatype activity \
    --video_dir "$VIDEO_DIR" \
    --prvr_train_annos "$ANNOTATION_DIR/activitynet_train.jsonl" \
    --prvr_val_annos "$ANNOTATION_DIR/activitynet_val.jsonl" \
    --pretrained_dir "$PRETRAINED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lr 2e-4 \
    --wd 0.01 \
    --epochs 10 \
    --batch_size 48 \
    --batch_size_val 16 \
    --num_thread_reader 8 \
    --max_words 64 \
    --Nf 32 \
    --precision amp \
    --n_display 40 \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --frame_attn_groups '[1,1,1,1,1,1,1,1,1,1,2,4]' \
    --top1_frame_attention_heads 8 \
    --nce_frame_topk 4 \
    --nce_frame_topk_temperature 0.05 \
    "$@"
