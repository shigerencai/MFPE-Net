import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.shanghai_folder import ShanghaiFolderDataset, PIXEL_SCALE, THRESHOLDS
from models.mfpe_net import MFPENet
from utils.checkpoint import load_model_weights
from utils.metrics import Evaluator, format_metrics
from utils.visualize import gray_to_color, save_sequence


def parse_args():
    parser = argparse.ArgumentParser("Test MFPENet on Shanghai folder-format PNG data")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "valid", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--frames_in", type=int, default=10)
    parser.add_argument("--frames_out", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--spec_num", type=int, default=20)
    parser.add_argument("--phy_base_c", type=int, default=48)
    parser.add_argument("--max_motion", type=float, default=8.0)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--disable_phy", action="store_true")
    parser.add_argument("--use_high_refiner", dest="use_high_refiner", action="store_true", default=True)
    parser.add_argument("--disable_high_refiner", dest="use_high_refiner", action="store_false")
    parser.add_argument("--refiner_hidden_dim", type=int, default=96)
    parser.add_argument("--refiner_blocks", type=int, default=4)
    parser.add_argument("--refiner_threshold", type=float, default=35.0)
    parser.add_argument("--refiner_gate_sharpness", type=float, default=1.0)
    parser.add_argument("--refiner_max_delta", type=float, default=0.15)

    parser.add_argument("--save_csi40_threshold", type=float, default=0.5)
    parser.add_argument("--disable_save_by_csi40", action="store_true")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument("--save_input", action="store_true")
    parser.add_argument("--grid_gap", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--csi40_bin", type=float, default=0.1)
    parser.add_argument("--top_k_csi40", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--progress_width", type=int, default=34)
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--show_cuda_mem", action="store_true")
    return parser.parse_args()


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def color(text, code, enabled = True):
    return f"{code}{text}{C.RESET}" if enabled else text


def fmt_time(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def make_bar(done, total, width, enabled_color):
    ratio = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "█" * filled + "░" * max(0, width - filled)
    if ratio < 0.34:
        return color(bar, C.YELLOW, enabled_color)
    if ratio < 0.67:
        return color(bar, C.CYAN, enabled_color)
    return color(bar, C.GREEN, enabled_color)


def clear_line_print(text):
    sys.stdout.write("\r" + text + "\033[K")
    sys.stdout.flush()


def cuda_mem_string(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f" | mem={alloc:.2f}/{reserved:.2f}G"

def build_model(args):
    return MFPENet(
        pre_seq_length=args.frames_in,
        aft_seq_length=args.frames_out,
        input_shape=(args.img_size, args.img_size),
        input_dim=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        spec_num=args.spec_num,
        use_phy=not args.disable_phy,
        phy_base_c=args.phy_base_c,
        max_motion=args.max_motion,
        residual_scale=args.residual_scale,
        use_high_refiner=args.use_high_refiner,
        refiner_hidden_dim=args.refiner_hidden_dim,
        refiner_blocks=args.refiner_blocks,
        refiner_threshold=args.refiner_threshold,
        refiner_gate_sharpness=args.refiner_gate_sharpness,
        refiner_max_delta=args.refiner_max_delta,
    )


def split_batch(batch, frames_in, frames_out):
    total = frames_in + frames_out
    if batch.shape[1] < total:
        raise ValueError(f"Need at least {total} frames, got {batch.shape[1]}")
    batch = batch[:, :total]
    return batch[:, :frames_in], batch[:, frames_in:total]


def amp_context(args, device):
    if not (args.amp and device.type == "cuda"):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def compute_sample_csi_at_threshold(
    pred_one,
    gt_one,
    threshold,
    value_scale = PIXEL_SCALE,
):
    pred_np = pred_one.detach().float().cpu().numpy()
    gt_np = gt_one.detach().float().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0) * float(value_scale)
    gt_np = np.clip(gt_np, 0.0, 1.0) * float(value_scale)

    pred_mask = pred_np >= float(threshold)
    gt_mask = gt_np >= float(threshold)

    hits = int(np.logical_and(pred_mask, gt_mask).sum())
    misses = int(np.logical_and(~pred_mask, gt_mask).sum())
    false_alarms = int(np.logical_and(pred_mask, ~gt_mask).sum())
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 0.0


def build_csi40_distribution(sample_scores, bin_width = 0.1):
    bin_width = float(bin_width)
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"--csi40_bin must be in (0, 1], got {bin_width}")

    values = [float(item["CSI40"]) for item in sample_scores]
    distribution: List[Dict[str, object]] = []
    n_bins = int(round(1.0 / bin_width))
    edges = [round(i * bin_width, 10) for i in range(n_bins + 1)]
    edges[-1] = 1.0

    for idx in range(n_bins - 1, -1, -1):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == n_bins - 1:
            matched = [v for v in values if low <= v <= high]
            label = f"[{low:.1f}, {high:.1f}]" if bin_width == 0.1 else f"[{low:.3f}, {high:.3f}]"
        else:
            matched = [v for v in values if low <= v < high]
            label = f"[{low:.1f}, {high:.1f})" if bin_width == 0.1 else f"[{low:.3f}, {high:.3f})"
        distribution.append({
            "range": label,
            "low": low,
            "high": high,
            "count": len(matched),
        })
    return distribution


def print_csi40_distribution(distribution, total_samples, enabled_color = True):
    print(color("\n[CSI40 DISTRIBUTION] per-sample CSI40, descending bins", C.BOLD + C.CYAN, enabled_color))
    print(color("-" * 58, C.GRAY, enabled_color))
    max_count = max([int(item["count"]) for item in distribution] + [1])
    for item in distribution:
        count = int(item["count"])
        ratio = count / max(total_samples, 1) * 100.0
        bar_len = int(round(30 * count / max_count)) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"CSI40 {item['range']:>12} : {count:5d} samples | {ratio:6.2f}% | {color(bar, C.GREEN, enabled_color)}")
    print(color("-" * 58, C.GRAY, enabled_color))


def save_csi40_records(
    out_dir,
    sample_scores,
    distribution,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_scores = sorted(sample_scores, key=lambda d: float(d["CSI40"]), reverse=True)

    json_path = out_dir / "sample_csi40_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_scores, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "sample_csi40_scores.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "CSI40"])
        writer.writeheader()
        for row in sorted_scores:
            writer.writerow({"sample_id": int(row["sample_id"]), "CSI40": f"{float(row['CSI40']):.8f}"})

    dist_path = out_dir / "csi40_distribution.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False, indent=2)

    return json_path, csv_path, dist_path

def _seq_to_rgb_frames(seq, value_scale = PIXEL_SCALE):
    arr = seq.detach().float().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected sequence shape [T,1,H,W] or [T,H,W], got {arr.shape}")
    frames: List[Image.Image] = []
    for frame in arr:
        rgba = gray_to_color(frame, value_scale=value_scale)
        frames.append(Image.fromarray(rgba, mode="RGBA").convert("RGB"))
    return frames


def save_gt_pred_grid(
    gt,
    pred,
    out_path,
    value_scale = PIXEL_SCALE,
    gap = 2,
):
    gt_frames = _seq_to_rgb_frames(gt, value_scale=value_scale)
    pred_frames = _seq_to_rgb_frames(pred, value_scale=value_scale)
    n = min(len(gt_frames), len(pred_frames))
    if n == 0:
        raise ValueError("No frames to save in gt/pred comparison grid.")
    gt_frames = gt_frames[:n]
    pred_frames = pred_frames[:n]

    cell_w, cell_h = gt_frames[0].size
    gap = max(0, int(gap))
    canvas_w = n * cell_w + (n - 1) * gap
    canvas_h = 2 * cell_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for i, img in enumerate(gt_frames):
        canvas.paste(img, (i * (cell_w + gap), 0))
    for i, img in enumerate(pred_frames):
        canvas.paste(img, (i * (cell_w + gap), cell_h + gap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_sample_outputs(
    x,
    y,
    pred,
    sample_id,
    sample_csi40,
    out_dir,
    args,
):
    sample_dir = out_dir / f"sample_{sample_id}_CSI40_{sample_csi40:.4f}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    grid_path = sample_dir / f"sample_{sample_id}_CSI40_{sample_csi40:.4f}_gt_pred_grid.png"
    save_gt_pred_grid(y, pred, grid_path, value_scale=PIXEL_SCALE, gap=args.grid_gap)

    if args.save_individual:
        if args.save_input:
            save_sequence(x, sample_dir, "input", value_scale=PIXEL_SCALE, color=True)
        save_sequence(y, sample_dir, "gt", value_scale=PIXEL_SCALE, color=True)
        save_sequence(pred, sample_dir, "pred", value_scale=PIXEL_SCALE, color=True)

    return grid_path


@torch.no_grad()
def main():
    args = parse_args()
    enabled_color = not args.no_color
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(color("\n[TEST] MFPENet Shanghai evaluation", C.BOLD + C.CYAN, enabled_color))
    print(color("=" * 92, C.GRAY, enabled_color))
    print(f"data_root   : {args.data_root}")
    print(f"split       : {args.split}")
    print(f"checkpoint  : {args.ckpt}")
    print(f"output_dir  : {out_dir}")
    print(f"device      : {device} | amp={args.amp}")
    print(f"batch_size  : {args.batch_size} | num_workers={args.num_workers}")
    print(f"img_size    : {args.img_size} | frames_in={args.frames_in} | frames_out={args.frames_out}")
    print(f"metrics     : SSIM=True | MSE=paper-scale | MSE_norm also saved | PIXEL_SCALE={PIXEL_SCALE} | THRESHOLDS={THRESHOLDS}")
    print(color(f"save rule   : per-sample CSI40 > {args.save_csi40_threshold:.4f}", C.GREEN, enabled_color))
    print(color(f"histogram   : CSI40 bins, width={args.csi40_bin}", C.GREEN, enabled_color))
    print(color("=" * 92, C.GRAY, enabled_color))

    dataset = ShanghaiFolderDataset(
        args.data_root,
        split=args.split,
        img_size=args.img_size,
        seq_len=args.seq_len,
        limit=args.limit or None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args).to(device)
    param_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(color(f"[MODEL] parameters={param_m:.2f}M", C.MAGENTA, enabled_color))

    missing, unexpected = load_model_weights(model, args.ckpt, device, strict=False)
    print(color(f"[CKPT] Loaded weights from: {args.ckpt}", C.GREEN, enabled_color))
    if missing:
        print(color(f"[INFO] Missing keys: {len(missing)}. Example: {missing[:8]}", C.YELLOW, enabled_color))
    if unexpected:
        print(color(f"[INFO] Unexpected keys: {len(unexpected)}. Example: {unexpected[:8]}", C.YELLOW, enabled_color))
    model.eval()

    evaluator = Evaluator(thresholds=THRESHOLDS, value_scale=PIXEL_SCALE, compute_ssim=True)
    total_samples = len(dataset)
    total_batches = len(loader)
    processed_samples = 0
    saved_samples: List[int] = []
    saved_paths: List[str] = []
    saved_sample_csi40: Dict[str, float] = {}
    sample_scores: List[Dict[str, float]] = []
    last_sample_csi40 = 0.0
    start_time = time.time()

    print(color(f"[DATA] samples={total_samples} | batches={total_batches}", C.BLUE, enabled_color))

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=True)
        x, y = split_batch(batch, args.frames_in, args.frames_out)

        with amp_context(args, device):
            pred, _ = model.predict(x, compute_loss=False)
        pred = pred.detach()
        evaluator.update(pred, y)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            sample_id = processed_samples + i + 1
            sample_csi40 = compute_sample_csi_at_threshold(pred[i], y[i], threshold=40, value_scale=PIXEL_SCALE)
            last_sample_csi40 = sample_csi40
            sample_scores.append({"sample_id": int(sample_id), "CSI40": float(sample_csi40)})

            if (not args.disable_save_by_csi40) and sample_csi40 > float(args.save_csi40_threshold):
                grid_path = save_sample_outputs(x[i], y[i], pred[i], sample_id, sample_csi40, out_dir, args)
                saved_samples.append(sample_id)
                saved_paths.append(str(grid_path))
                saved_sample_csi40[str(sample_id)] = float(sample_csi40)

        processed_samples += batch_size

        need_log = (
            batch_idx == 1
            or batch_idx == total_batches
            or (args.log_interval > 0 and batch_idx % args.log_interval == 0)
        )
        if need_log:
            last_metrics = evaluator.compute_paper_metrics()
            elapsed = time.time() - start_time
            speed = processed_samples / max(elapsed, 1e-8)
            eta = (total_samples - processed_samples) / max(speed, 1e-8)
            percent = processed_samples / max(total_samples, 1) * 100.0
            bar = make_bar(processed_samples, total_samples, args.progress_width, enabled_color)
            metric_text = (
                f"CSI-M={last_metrics.get('CSI-M', 0):.4f} | "
                f"CSI20={last_metrics.get('CSI20', 0):.4f} | "
                f"CSI40={last_metrics.get('CSI40', 0):.4f} | "
                f"HSS-M={last_metrics.get('HSS-M', 0):.4f} | "
                f"SSIM={last_metrics.get('SSIM', 0):.4f} | "
                f"MSE={last_metrics.get('MSE', 0):.6f} | "
                f"MSE_norm={last_metrics.get('MSE_norm', 0):.6f} | "
                f"last_sample_CSI40={last_sample_csi40:.4f} | "
                f"saved={len(saved_samples)}"
            )
            mem_text = cuda_mem_string(device) if args.show_cuda_mem else ""
            line = (
                f"{color(f'Test {args.split}', C.BOLD + C.CYAN, enabled_color)} | "
                f"{bar} | "
                f"{processed_samples:5d}/{total_samples:<5d} | "
                f"{percent:6.2f}% | "
                f"[{fmt_time(elapsed)}<{fmt_time(eta)}, {speed:.2f}sample/s] | "
                f"{metric_text}{mem_text}"
            )
            clear_line_print(line)

    sys.stdout.write("\n")
    sys.stdout.flush()

    metrics = evaluator.compute_paper_metrics()
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    distribution = build_csi40_distribution(sample_scores, bin_width=args.csi40_bin)
    print_csi40_distribution(distribution, total_samples=len(sample_scores), enabled_color=enabled_color)
    csi40_json_path, csi40_csv_path, csi40_dist_path = save_csi40_records(out_dir, sample_scores, distribution)

    sorted_scores = sorted(sample_scores, key=lambda d: float(d["CSI40"]), reverse=True)
    if args.top_k_csi40 and args.top_k_csi40 > 0:
        print(color(f"\n[TOP {args.top_k_csi40} CSI40 SAMPLES]", C.BOLD + C.MAGENTA, enabled_color))
        for row in sorted_scores[: args.top_k_csi40]:
            print(f"sample_{int(row['sample_id'])}: CSI40={float(row['CSI40']):.6f}")

    save_record = {
        "split": args.split,
        "save_rule": f"per-sample CSI40 > {args.save_csi40_threshold}",
        "save_csi40_threshold": float(args.save_csi40_threshold),
        "saved_samples": saved_samples,
        "saved_sample_csi40": saved_sample_csi40,
        "saved_paths": saved_paths,
        "metrics_path": str(metrics_path),
        "sample_csi40_scores_json": str(csi40_json_path),
        "sample_csi40_scores_csv": str(csi40_csv_path),
        "csi40_distribution_json": str(csi40_dist_path),
    }
    saved_samples_path = out_dir / "saved_samples.json"
    with open(saved_samples_path, "w", encoding="utf-8") as f:
        json.dump(save_record, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(color("\n[RESULT]", C.BOLD + C.GREEN, enabled_color), format_metrics(metrics))
    print(color(f"[TIME] total={fmt_time(total_time)} | speed={total_samples / max(total_time, 1e-8):.2f} sample/s", C.BLUE, enabled_color))
    print(color(f"[SAVE] metrics: {metrics_path}", C.GREEN, enabled_color))
    print(color(f"[SAVE] sample CSI40 json: {csi40_json_path}", C.GREEN, enabled_color))
    print(color(f"[SAVE] sample CSI40 csv : {csi40_csv_path}", C.GREEN, enabled_color))
    print(color(f"[SAVE] CSI40 distribution: {csi40_dist_path}", C.GREEN, enabled_color))
    print(color(f"[SAVE] saved_samples: {saved_samples_path}", C.GREEN, enabled_color))
    print(color(f"[SAVE] CSI40-filtered comparison images: {len(saved_samples)}", C.GREEN, enabled_color))
    if saved_paths:
        print(color("[SAVE] examples:", C.GREEN, enabled_color))
        for path in saved_paths[:10]:
            print(f"  {path}")
        if len(saved_paths) > 10:
            print(f"  ... and {len(saved_paths) - 10} more")


if __name__ == "__main__":
    main()
