"""Single-GPU ITA training and PRVR evaluation."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dataloaders.dataset import (
    get_train_dataloader,
    get_val_txt_dataloader,
    get_val_vis_dataloader,
)
from modules import ITA, SimpleTokenizer
from params import get_args, save_hp_to_json
from utils.log import set_logger
from utils.lr_scheduler import CosineWarmupScheduler
from utils.metrics import AverageMeter, compute_metrics_prvr
from utils.misc import save_checkpoint, set_random_seed
from utils.optimization import prep_optim_params_groups


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(str(Path(args.output_dir) / "log.txt"), logging.INFO)
    set_random_seed(args.seed)
    if args.do_train:
        save_hp_to_json(args.output_dir, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning(
            "Using CPU; full training requires a CUDA GPU for practical runtimes"
        )
    tokenizer = SimpleTokenizer()
    model = ITA.from_pretrained(args).float().to(device)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logging.info("Loaded ITA checkpoint: %s", args.resume)

    val_vis, num_videos = get_val_vis_dataloader(args)
    val_txt, num_texts = get_val_txt_dataloader(args, tokenizer)
    logging.info("Validation: %d videos, %d queries", num_videos, num_texts)
    if args.do_eval:
        prvr_eval_epoch(model, val_vis, val_txt, device)
        return

    train_loader, num_train = get_train_dataloader(args, tokenizer)
    if not len(train_loader):
        raise ValueError(
            "No complete training batch; reduce --batch_size or check training annotations"
        )
    total_steps = len(train_loader) * args.epochs
    optimizer = torch.optim.AdamW(
        prep_optim_params_groups(args, model),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.wd,
    )
    scheduler = CosineWarmupScheduler(
        args.lr, total_steps, args.warmup_proportion * total_steps
    )
    scaler = GradScaler() if args.precision == "amp" and device.type == "cuda" else None
    start_epoch, global_step = 0, 0
    best_weighted, best_top1 = 0.0, 0.0
    if checkpoint is not None:
        if "optimizer" in checkpoint:
            state = checkpoint["optimizer"]
            # Earlier ITA runs also stored two groups of frozen CLIP parameters.
            if len(state["param_groups"]) == 4:
                state["param_groups"] = state["param_groups"][2:]
                active_ids = {
                    p for group in state["param_groups"] for p in group["params"]
                }
                state["state"] = {
                    k: v for k, v in state["state"].items() if k in active_ids
                }
            optimizer.load_state_dict(state)
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", start_epoch * len(train_loader))
        best_weighted = checkpoint.get(
            "best_weighted_acc1", checkpoint.get("best_acc1", 0.0)
        )
        best_top1 = checkpoint.get("best_top1_acc1", 0.0)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(
        "Parameters: %d trainable / %d total (%.2f%%)",
        trainable,
        total,
        100 * trainable / total,
    )
    logging.info(
        "Training: %d videos, batch size %d, %d steps",
        num_train,
        args.batch_size,
        total_steps,
    )
    for epoch in range(start_epoch, args.epochs):
        loss, global_step = train_epoch(
            epoch,
            args,
            model,
            train_loader,
            device,
            optimizer,
            global_step,
            scheduler,
            scaler,
        )
        logging.info("Epoch %d/%d: train loss %.6f", epoch + 1, args.epochs, loss)
        weighted, top1, _ = prvr_eval_epoch(model, val_vis, val_txt, device)
        is_best = weighted["SumR"] >= best_weighted
        best_weighted = max(best_weighted, weighted["SumR"])
        best_top1 = max(best_top1, top1["SumR"])
        state = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "arch": "ITA",
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc1": best_weighted,
            "best_weighted_acc1": best_weighted,
            "best_top1_acc1": best_top1,
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        save_checkpoint(state, is_best, args.output_dir, filename="ckpt.pth.tar")
    logging.info("Best SumR: weighted %.4f; top-1 %.4f", best_weighted, best_top1)


def train_epoch(
    epoch,
    args,
    model,
    train_dataloader,
    device,
    optimizer,
    global_step,
    scheduler,
    scaler=None,
):
    model.train()
    loss_meter = AverageMeter()
    for step, batch in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader), desc="train"
    ):
        optimizer.zero_grad()
        scheduler(optimizer, global_step)
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(batch)
            loss = output["loss"].mean()
        loss_meter.update(loss.item(), n=len(batch["video_name"]))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        with torch.no_grad():
            model.clip.logit_scale.clamp_(0.1, 4.6052)
        global_step += 1
        if global_step % args.n_display == 0:
            logging.info(
                "Epoch %d [%d/%d] loss %.4f; lr %.2e",
                epoch + 1,
                step + 1,
                len(train_dataloader),
                loss_meter.avg,
                optimizer.param_groups[0]["lr"],
            )
    return loss_meter.avg, global_step


def prvr_eval_epoch(model, val_vis_dataloader, val_txt_dataloader, device):
    model.eval()
    GT_TXT2VID = val_txt_dataloader.dataset.txt2vid
    torch.cuda.empty_cache()
    with torch.no_grad():
        text_names, video_names = [], []
        batch_sequence_output_list, batch_visual_output_list = [], []

        logging.info("calculating text features...")
        for bid, batch in tqdm(
            enumerate(val_txt_dataloader), total=len(val_txt_dataloader)
        ):
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(device=device, non_blocking=True)
            sequence_output = model.encode_text(batch["text_ids"])
            batch_sequence_output_list.append(sequence_output.cpu())
            torch.cuda.empty_cache()
            text_names += batch["text_name"]

        logging.info("calculating video features...")
        for bid, batch in tqdm(
            enumerate(val_vis_dataloader), total=len(val_vis_dataloader)
        ):
            batch = tuple(
                t.to(device=device, non_blocking=True) if type(t) is not tuple else t
                for t in batch
            )
            video_name, video = batch
            visual_output = model.encode_video(video)
            visual_output = model.get_all_fused_frame_features(visual_output)
            batch_visual_output_list.append(visual_output.cpu())
            torch.cuda.empty_cache()
            video_names += video_name
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        weighted_sim_matrix, top1_sim_matrix = _prvr_run_on_single_gpu(
            model,
            batch_sequence_output_list,
            batch_visual_output_list,
            device=device,
            visual_output_is_fused=True,
        )

    logging.info(
        "weighted sim matrix size: {}, {}".format(
            weighted_sim_matrix.shape[0], weighted_sim_matrix.shape[1]
        )
    )
    logging.info(
        "top1 sim matrix size: {}, {}".format(
            top1_sim_matrix.shape[0], top1_sim_matrix.shape[1]
        )
    )
    weighted_metrics = compute_metrics_prvr(
        weighted_sim_matrix, text_names, video_names, gt_txt2vid=GT_TXT2VID
    )
    top1_metrics = compute_metrics_prvr(
        top1_sim_matrix, text_names, video_names, gt_txt2vid=GT_TXT2VID
    )
    logging.info(
        "\t Length-T: {}, Length-V:{}".format(
            len(weighted_sim_matrix), len(weighted_sim_matrix[0])
        )
    )

    info_str = []
    info_str.append("Text-to-Video Weighted:")
    info_str.append(
        " (metric) >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@100: {:.1f} - SumR: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
            weighted_metrics["R1"],
            weighted_metrics["R5"],
            weighted_metrics["R10"],
            weighted_metrics["R100"],
            weighted_metrics["SumR"],
            weighted_metrics["MR"],
            weighted_metrics["MeanR"],
        )
    )
    info_str.append("Text-to-Video Top1:")
    info_str.append(
        " (metric) >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@100: {:.1f} - SumR: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".format(
            top1_metrics["R1"],
            top1_metrics["R5"],
            top1_metrics["R10"],
            top1_metrics["R100"],
            top1_metrics["SumR"],
            top1_metrics["MR"],
            top1_metrics["MeanR"],
        )
    )
    for info in info_str:
        logging.info(info)

    return weighted_metrics, top1_metrics, info_str


def _prvr_run_on_single_gpu(
    model,
    batch_sequence_output_list,
    batch_visual_output_list,
    device=None,
    visual_output_is_fused=False,
):

    if device is None:
        device = next(model.parameters()).device

    weighted_sim_matrix = []
    top1_sim_matrix = []

    for sequence_output in batch_sequence_output_list:
        sequence_output = sequence_output.to(device=device, non_blocking=True)
        weighted_row = []
        top1_row = []
        for visual_output in batch_visual_output_list:
            visual_output = visual_output.to(device=device, non_blocking=True)
            weighted_logits = model.get_prvr_similarity_logits(
                sequence_output,
                visual_output,
                visual_output_is_fused=visual_output_is_fused,
            )
            top1_logits = model.get_prvr_similarity_logits(
                sequence_output,
                visual_output,
                visual_output_is_fused=visual_output_is_fused,
                force_frame_max=True,
            )
            weighted_row.append(weighted_logits.cpu().detach().numpy())
            top1_row.append(top1_logits.cpu().detach().numpy())
            del visual_output, weighted_logits, top1_logits
        weighted_row = np.concatenate(tuple(weighted_row), axis=-1)
        top1_row = np.concatenate(tuple(top1_row), axis=-1)
        weighted_sim_matrix.append(weighted_row)
        top1_sim_matrix.append(top1_row)
        del sequence_output
    weighted_sim_matrix = np.concatenate(tuple(weighted_sim_matrix), axis=0)
    top1_sim_matrix = np.concatenate(tuple(top1_sim_matrix), axis=0)
    return weighted_sim_matrix, top1_sim_matrix


if __name__ == "__main__":
    main(get_args())
