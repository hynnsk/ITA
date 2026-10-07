# Intrinsic Temporal Adaptation of CLIP for Partially Relevant Video Retrieval (EMNLP 2026)

Hyun Seok Seong, Woojin Jun, SuBeen Lee, and Jae-Pil Heo

ITA adapts a frozen CLIP ViT-B/32 backbone for retrieval from untrimmed videos. It combines:

- **Backbone-Internal Temporal Adaptation:** grouped attention in the final visual layers, temporal positional embeddings, relative temporal bias, and global frame attention.
- **Affinity-Weighted Gradient Propagation:** query-dependent, soft aggregation of the top-k frames for bidirectional contrastive learning.
- **Parameter-efficient adaptation:** LoRA on the QKV input projections in both encoders; the pretrained backbone stays frozen.

The four supported benchmarks are ActivityNet Captions, TVR, Charades-STA, and QVHighlights.

## Installation

Use Linux, Bash, and a CUDA-capable GPU. The paper reports experiments on one NVIDIA RTX A6000. The dependency setup below follows the [upstream installation recipe](https://github.com/BUAAPY/ProPy#installation).

```bash
conda create -n ita python=3.10 -y
conda activate ita
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
conda install ffmpeg -c conda-forge
pip install -r requirements.txt
```

Run the following commands from the repository root. FFmpeg must be on `PATH` for video preparation.

## Dataset preparation

Video preprocessing and annotation formats follow the [upstream preparation procedure](https://github.com/BUAAPY/ProPy#preparation). Use its converted annotations so the splits and video IDs match the training loader.

### 1. Download videos and annotations

| Dataset | Video source | Train annotations | Validation annotations |
| --- | --- | --- | --- |
| ActivityNet Captions | [ActivityNet](http://activity-net.org/download.html) | `activitynet_train.jsonl` | `activitynet_val.jsonl` |
| TVR | [TVQA frames/videos](https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa.html) | `tvr_train_release.jsonl` | `tvr_val_release.jsonl` |
| Charades-STA | [Charades videos](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip) | `charades_train.jsonl` | `charades_val.jsonl` |
| QVHighlights | [QVHighlights videos](https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz) | `qvhighlights_train.jsonl` | `qvhighlights_val.jsonl` |

Obtain the converted annotation archive from [Google Drive](https://drive.google.com/drive/folders/1HZk1JnW50ZhYdFIcDXOscKe3QaSmGRoj?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1BzQAaBhPOTH7d0pyAsbw8A?pwd=s2w5) and extract the JSONL files into `annotations/`. These are upstream data resources, not ITA checkpoints. Follow the respective dataset providers' access procedures for TVQA and ActivityNet.

Each JSONL line contains a query-video pair, for example:

```json
{"desc_id": 0, "vid_name": "video_id", "desc": "A person opens a door.", "duration": 30.0, "ts": [4.0, 8.0]}
```

ITA uses `desc_id`, `vid_name`, and `desc`; temporal annotations are not used for training. Multiple queries can refer to the same video.

### 2. Prepare videos

`preprocess/compress_video.py` recursively reads videos, converts them to MP4 at **3 fps**, and resizes the shorter side to **224 pixels**, preserving the aspect ratio. It writes a flat output directory, skips existing output files, and reports FFmpeg failures. Input and output directories must be separate. Basenames must be unique within a dataset.

ActivityNet videos may be spread across `v1-2/train`, `v1-2/val`, and `v1-3/train_val`; compress each into the same output directory:

```bash
python preprocess/compress_video.py \
    --input_root /path/to/ActivityNet/v1-2/train \
    --output_root data/ActivityNet_compressed
python preprocess/compress_video.py \
    --input_root /path/to/ActivityNet/v1-2/val \
    --output_root data/ActivityNet_compressed
python preprocess/compress_video.py \
    --input_root /path/to/ActivityNet/v1-3/train_val \
    --output_root data/ActivityNet_compressed
```

For ActivityNet, the loader maps annotation ID `v_<id>` to filename `<id>.mp4`, matching the original data convention. Ensure compressed filenames follow this convention; the compression tool preserves the input basename.

If TVQA is distributed as frame folders, first convert each video's folder into a video. Frame folders can be nested under show directories; each video's folder must have a unique name matching its annotation ID.

```bash
python preprocess/image_to_mp4.py \
    --frame_dir /path/to/TVQA/frames \
    --video_dir data/TVR_raw_videos --fps 3
python preprocess/compress_video.py \
    --input_root data/TVR_raw_videos \
    --output_root data/TVR_compressed
```

For Charades and QVHighlights:

```bash
python preprocess/compress_video.py \
    --input_root /path/to/Charades_v1 \
    --output_root data/Charades_compressed
python preprocess/compress_video.py \
    --input_root /path/to/QVHighlights/videos \
    --output_root data/QVHighlights_compressed
```

Use `--num_workers` to control parallel compression. For TVR, Charades, and QVHighlights, a video named `<id>.mp4` must correspond to annotation `vid_name: <id>`.

### 3. Download CLIP weights

Download the [OpenAI CLIP ViT-B/32 weights](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt):

```bash
mkdir -p CLIP_weights
curl -L --fail \
    https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
    -o CLIP_weights/ViT-B-32.pt
```

Expected layout:

```text
ITA/
├── CLIP_weights/ViT-B-32.pt
├── annotations/
│   ├── activitynet_{train,val}.jsonl
│   ├── tvr_{train,val}_release.jsonl
│   ├── charades_{train,val}.jsonl
│   └── qvhighlights_{train,val}.jsonl
├── data/
│   ├── ActivityNet_compressed/
│   ├── TVR_compressed/
│   ├── Charades_compressed/
│   └── QVHighlights_compressed/
├── scripts/
└── main_prvr.py
```

## Training

Each script launches **one dataset only**. Run the desired command independently:

| Dataset | Command | Learning rate | Output directory |
| --- | --- | --- | --- |
| ActivityNet Captions | `bash scripts/train_activitynet.sh` | `2e-4` | `logs/ita_activitynet` |
| TVR | `bash scripts/train_tvr.sh` | `1e-3` | `logs/ita_tvr` |
| Charades-STA | `bash scripts/train_charades.sh` | `2e-4` | `logs/ita_charades` |
| QVHighlights | `bash scripts/train_qvhighlights.sh` | `1e-4` | `logs/ita_qvhighlights` |

The scripts share the paper settings: 10 epochs, batch size 48, 32 sampled frames, maximum text length 64, AdamW with weight decay 0.01, AMP, LoRA rank 8 and alpha 16, frame groups `[1,1,1,1,1,1,1,1,1,1,2,4]`, 8 heads for global frame attention, top-k = 4, and affinity temperature 0.05. Training uses random segment sampling; evaluation uses evenly spaced segment centers.

Override paths and the visible GPU without editing the script:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/DATA/01_PRVR_raw_video \
ANNOTATION_DIR=/path/to/annotations \
PRETRAINED_DIR=/path/to/CLIP_weights \
bash scripts/train_activitynet.sh
```

`VIDEO_DIR` overrides a single dataset's video directory, `OUTPUT_DIR` selects the log/checkpoint directory, and `PYTHON` selects the Python executable. For an existing Charades folder:

```bash
VIDEO_DIR=/DATA/01_PRVR_raw_video/Charades_compressed_debugged \
bash scripts/train_charades.sh
```

Additional CLI arguments override the script defaults:

```bash
bash scripts/train_tvr.sh --batch_size 24 --num_thread_reader 4
```

Changing the training batch size changes the experimental setup. See `python main_prvr.py --help` for the retained arguments. Training writes `log.txt`, `hparams_train.json`, the latest `ckpt.pth.tar`, and `ckpt.best.pth.tar`, selected by the weighted retrieval **SumR** on validation.

## Evaluation and resuming

Evaluate a trained checkpoint using the corresponding dataset script:

```bash
bash scripts/train_activitynet.sh \
    --do_train 0 --do_eval 1 \
    --resume logs/ita_activitynet/ckpt.best.pth.tar
```

Evaluation reports R@1, R@5, R@10, R@100, SumR, median rank, and mean rank for affinity-weighted retrieval and the top-1 diagnostic. The weighted score is ITA's main retrieval score. Evaluation loads only validation videos and queries; it does not read training annotations or construct an optimizer.

Resume from the latest checkpoint with the same model and dataset settings:

```bash
bash scripts/train_activitynet.sh --resume logs/ita_activitynet/ckpt.pth.tar
```

ITA model parameter names are preserved for checkpoints from the previous training script with the same architecture. The loader also accepts a `module.` prefix and the earlier optimizer layout containing frozen CLIP parameter groups. The original CLIP weights are required to construct the model before loading a checkpoint.

## Code layout

- `main_prvr.py`: training, validation, and checkpoint handling.
- `params.py`: ITA training and evaluation arguments.
- `modules/clip_ita.py`: CLIP encoders, LoRA, and grouped temporal attention.
- `modules/ita.py`: global frame attention, affinity-weighted aggregation, and bidirectional contrastive loss.
- `dataloaders/`: video decoding, frame sampling, tokenization, and batches.
- `preprocess/`: video compression and TVQA frame conversion.
- `utils/`: optimizer groups, learning-rate schedule, logging, and retrieval metrics.

## Acknowledgements

This implementation builds on the [ProPy](https://github.com/BUAAPY/ProPy), including its dataset preparation and converted annotations, and uses components from [CLIP](https://github.com/openai/CLIP) and [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip).
