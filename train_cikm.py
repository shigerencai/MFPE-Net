import argparse
import json
import os

for _env_name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    _env_value = os.environ.get(_env_name, "").strip()
    if _env_value == "" or _env_value == "0":
        os.environ[_env_name] = "1"

import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.cikm_folder import CIKMFolderDataset, PIXEL_SCALE, THRESHOLDS
from models.mfpe_net import MFPENet
from utils.checkpoint import load_model_weights, resume_checkpoint, save_checkpoint
from utils.metrics import Evaluator, format_metrics
from utils.visualize import save_gt_pred_grid


def autocast_context(args, device):
    enabled = bool(args.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


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
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


def enable_windows_ansi():
    if os.name == "nt":
        try:
            os.system("")
        except Exception:
            pass


def color(text, code, enabled = True):
    if not enabled:
        return text
    return f"{code}{text}{C.RESET}"


def tag(name, code, enabled = True):
    return color(f"[{name}]", code, enabled)


def now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def gpu_mem(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(idx) / 1024 ** 2
    return f"gpu={allocated:.0f}/{reserved:.0f}MB"


def fmt_count(num):
    if num is None:
        return "N/A"
    num = float(num)
    units = ["", "K", "M", "G", "T", "P"]
    idx = 0
    while abs(num) >= 1000.0 and idx < len(units) - 1:
        num /= 1000.0
        idx += 1
    return f"{num:.2f}{units[idx]}"


class _ForwardOnlyWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred, _ = self.model.predict(x, compute_loss=False)
        return pred


def estimate_model_flops(model, args, device):
    dummy = torch.zeros(1, args.frames_in, 1, args.img_size, args.img_size, device=device)
    wrapper = _ForwardOnlyWrapper(model).to(device).eval()
    result = {"flops_per_sample": None, "method": "unavailable", "note": ""}

    was_training = model.training
    try:
        with torch.no_grad():
            try:
                from thop import profile as thop_profile  # type: ignore
                flops, _params = thop_profile(wrapper, inputs=(dummy,), verbose=False)
                result["flops_per_sample"] = float(flops)
                result["method"] = "thop"
                result["note"] = "approximate; some FFT/custom ops may be missed"
                return result
            except Exception as e_thop:
                result["note"] = f"thop failed: {type(e_thop).__name__}"

            try:
                from torch.profiler import ProfilerActivity, profile
                activities = [ProfilerActivity.CPU]
                if device.type == "cuda" and torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                    torch.cuda.synchronize()

                with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                    _ = wrapper(dummy)

                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()

                flops = 0.0
                for item in prof.key_averages():
                    item_flops = getattr(item, "flops", 0) or 0
                    flops += float(item_flops)

                if flops > 0:
                    result["flops_per_sample"] = flops
                    result["method"] = "torch.profiler"
                    result["note"] = "approximate; profiler only counts supported ops"
                else:
                    result["method"] = "unavailable"
                    result["note"] = "FLOPs profiler returned 0; install thop for another estimate"
            except Exception as e_prof:
                old_note = result.get("note", "")
                result["method"] = "unavailable"
                result["note"] = (old_note + "; " if old_note else "") + f"torch.profiler failed: {type(e_prof).__name__}"
    finally:
        model.train(was_training)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


@torch.no_grad()
def measure_inference_fps(model, args, device):
    was_training = model.training
    model.eval()

    profile_bs = max(int(args.profile_batch_size), 1)
    warmup = max(int(args.profile_warmup), 0)
    iters = max(int(args.profile_iters), 1)
    dummy = torch.zeros(profile_bs, args.frames_in, 1, args.img_size, args.img_size, device=device)

    try:
        for _ in range(warmup):
            with autocast_context(args, device):
                _ = model.predict(dummy, compute_loss=False)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        for _ in range(iters):
            with autocast_context(args, device):
                _ = model.predict(dummy, compute_loss=False)

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(time.time() - start, 1e-12)

        sample_fps = profile_bs * iters / elapsed
        output_frame_fps = sample_fps * args.frames_out
        latency_ms_per_batch = elapsed / iters * 1000.0
        latency_ms_per_sample = latency_ms_per_batch / profile_bs

        return {
            "profile_batch_size": float(profile_bs),
            "sample_fps": sample_fps,
            "output_frame_fps": output_frame_fps,
            "latency_ms_per_batch": latency_ms_per_batch,
            "latency_ms_per_sample": latency_ms_per_sample,
        }
    finally:
        model.train(was_training)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_model_profile(model, args, device, color_enabled = True):
    if not args.profile_model:
        return {}

    print(f"\n{tag('PROFILE', C.CYAN + C.BOLD, color_enabled)} Measuring model FLOPs and inference FPS ...")
    profile_result: Dict[str, object] = {}

    flops_info = estimate_model_flops(model, args, device)
    fps_info = measure_inference_fps(model, args, device)

    flops_per_sample = flops_info.get("flops_per_sample")
    flops_per_batch = None
    if flops_per_sample is not None:
        flops_per_batch = float(flops_per_sample) * max(int(args.profile_batch_size), 1)

    profile_result.update({
        "flops_per_sample": flops_per_sample,
        "flops_method": flops_info.get("method", "unavailable"),
        "flops_note": flops_info.get("note", ""),
        **fps_info,
    })

    print_box("Model Speed / FLOPs", {
        "FLOPs per sample": f"{fmt_count(flops_per_sample)} FLOPs",
        "FLOPs per profile batch": f"{fmt_count(flops_per_batch)} FLOPs  (batch={max(int(args.profile_batch_size), 1)})",
        "FLOPs method": flops_info.get("method", "unavailable"),
        "FLOPs note": flops_info.get("note", ""),
        "inference FPS": f"{fps_info['sample_fps']:.2f} samples/s",
        "output frame FPS": f"{fps_info['output_frame_fps']:.2f} frames/s  ({args.frames_out} output frames/sample)",
        "latency/sample": f"{fps_info['latency_ms_per_sample']:.2f} ms",
        "latency/profile batch": f"{fps_info['latency_ms_per_batch']:.2f} ms  (batch={max(int(args.profile_batch_size), 1)})",
        "profile warmup/iters": f"{max(int(args.profile_warmup), 0)}/{max(int(args.profile_iters), 1)}",
    }, color_enabled)
    return profile_result


class SimpleProgress:
    def __init__(
        self,
        total,
        prefix,
        width = 34,
        color_enabled = True,
        bar_color = C.GREEN,
        min_interval = 0.15,
    ):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = max(int(width), 8)
        self.color_enabled = color_enabled
        self.bar_color = bar_color
        self.min_interval = min_interval
        self.start = time.time()
        self.last_print = 0.0
        self.finished = False

    def _bar(self, current):
        ratio = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * ratio)
        raw_bar = "█" * filled + "░" * (self.width - filled)
        return color(raw_bar, self.bar_color, self.color_enabled)

    @staticmethod
    def _speed_text(elapsed, current):
        if current <= 0 or elapsed <= 0:
            return "0.00it/s"
        sec_per_it = elapsed / current
        if sec_per_it >= 1.0:
            return f"{sec_per_it:.2f}s/it"
        return f"{1.0 / sec_per_it:.2f}it/s"

    def update(
        self,
        current,
        metrics = None,
        lr = None,
        extra = "",
        force = False,
    ):
        current = min(max(int(current), 0), self.total)
        now = time.time()
        if not force and current < self.total and (now - self.last_print) < self.min_interval:
            return

        ratio = current / self.total
        elapsed = now - self.start
        eta = elapsed / ratio - elapsed if ratio > 0 else 0.0
        percent = ratio * 100.0
        speed = self._speed_text(elapsed, current)

        desc = f"{self.prefix}, Step {current}/{self.total}"
        time_part = f"[{fmt_seconds(elapsed)}<{fmt_seconds(eta)}, {speed}]"

        parts = [
            desc,
            self._bar(current),
            color(f"{percent:6.2f}%", C.YELLOW, self.color_enabled),
            color(time_part, C.CYAN, self.color_enabled),
        ]

        if lr is not None:
            parts.append(color(f"lr={lr:.2e}", C.BLUE, self.color_enabled))

        if metrics:
            # Keep AlphaPre-like postfix key=value style. Show common keys first.
            show_keys = ["total_loss", "ampli_loss", "intensity_loss", "physical_loss", "refinement_loss"]
            for k in show_keys:
                if k in metrics:
                    parts.append(f"{k}={metrics[k]:.4f}")

        if extra:
            parts.append(extra)

        line = " | ".join(parts)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        self.last_print = now

        if current >= self.total and not self.finished:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.finished = True

    def close(self):
        if not self.finished:
            self.update(self.total, force=True)


def print_box(title, rows, color_enabled = True):
    title_text = color(f" {title} ", C.BOLD + C.WHITE + C.BG_BLUE, color_enabled)
    print("\n" + title_text)
    key_width = max([len(str(k)) for k in rows.keys()] + [8])
    for k, v in rows.items():
        print(f"  {color(str(k).ljust(key_width), C.CYAN, color_enabled)} : {v}")


def parse_args():
    parser = argparse.ArgumentParser("Train MFPENet on CIKM folder-format PNG data")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--frames_in", type=int, default=5)
    parser.add_argument("--frames_out", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--spec_num", type=int, default=20)
    parser.add_argument("--phy_base_c", type=int, default=48)
    parser.add_argument("--max_motion", type=float, default=8.0)
    parser.add_argument("--residual_scale", type=float, default=0.25)

    parser.add_argument("--use_high_refiner", dest="use_high_refiner", action="store_true", default=True)
    parser.add_argument("--disable_high_refiner", dest="use_high_refiner", action="store_false")
    parser.add_argument("--refiner_hidden_dim", type=int, default=96)
    parser.add_argument("--refiner_blocks", type=int, default=4)
    parser.add_argument("--refiner_threshold", type=float, default=35.0)
    parser.add_argument("--refiner_gate_sharpness", type=float, default=1.0)
    parser.add_argument("--refiner_max_delta", type=float, default=0.15)
    parser.add_argument("--refiner_weight", type=float, default=0.05)

    parser.add_argument("--lead_loss_weight", type=float, default=0.10)
    parser.add_argument("--lead_weight_start", type=float, default=1.0)
    parser.add_argument("--lead_weight_end", type=float, default=2.0)
    parser.add_argument("--lead_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--temporal_weight", type=float, default=0.05)
    parser.add_argument("--temporal_loss_type", type=str, default="l1", choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--disable_phy", action="store_true")
    parser.add_argument("--pha_weight", type=float, default=0.01)
    parser.add_argument("--amp_weight", type=float, default=0.01)
    parser.add_argument("--anet_weight", type=float, default=0.1)
    parser.add_argument("--phy_weight", type=float, default=0.1)
    parser.add_argument("--aw_stop_step", type=int, default=5000)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--no_resume_optimizer", action="store_true")
    parser.add_argument("--strict_load", action="store_true")

    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_train_vis", dest="save_train_vis", action="store_true", default=True)
    parser.add_argument("--no_save_train_vis", dest="save_train_vis", action="store_false")
    parser.add_argument("--early_stop", type=int, default=0)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_log_interval", type=int, default=50)
    parser.add_argument("--progress_width", type=int, default=34)
    parser.add_argument("--no_color", action="store_true")
    parser.add_argument("--show_cuda_mem", action="store_true")
    parser.add_argument("--show_grad", action="store_true")
    parser.add_argument("--show_iter_time", action="store_true")
    parser.add_argument("--profile_model", dest="profile_model", action="store_true", default=True)
    parser.add_argument("--no_profile_model", dest="profile_model", action="store_false")
    parser.add_argument("--profile_batch_size", type=int, default=1)
    parser.add_argument("--profile_warmup", type=int, default=5)
    parser.add_argument("--profile_iters", type=int, default=20)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(args):
    return MFPENet(
        pre_seq_length=args.frames_in,
        aft_seq_length=args.frames_out,
        input_shape=(args.img_size, args.img_size),
        input_dim=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        spec_num=args.spec_num,
        pha_weight=args.pha_weight,
        anet_weight=args.anet_weight,
        amp_weight=args.amp_weight,
        phy_weight=args.phy_weight,
        aweight_stop_steps=args.aw_stop_step,
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
        refiner_weight=args.refiner_weight,
    )


def split_batch(batch, frames_in, frames_out):
    total = frames_in + frames_out
    if batch.shape[1] < total:
        raise ValueError(f"Need at least {total} frames, got {batch.shape[1]}")
    batch = batch[:, :total]
    return batch[:, :frames_in], batch[:, frames_in:total]


def _framewise_loss(pred, target, loss_type = "mse"):
    if loss_type == "l1":
        loss_map = torch.abs(pred - target)
    elif loss_type == "smooth_l1":
        loss_map = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        loss_map = (pred - target) ** 2
    return loss_map.flatten(start_dim=2).mean(dim=2)


def lead_time_weighted_loss(
    pred,
    target,
    weight_start = 1.0,
    weight_end = 2.0,
    loss_type = "mse",
):
    if pred.shape[1] <= 0:
        return pred.new_tensor(0.0)
    per_frame = _framewise_loss(pred, target, loss_type=loss_type)  # [B, T]
    t = pred.shape[1]
    weights = torch.linspace(
        float(weight_start),
        float(weight_end),
        steps=t,
        device=pred.device,
        dtype=pred.dtype,
    )
    weights = weights / weights.mean().clamp_min(1e-8)
    return (per_frame * weights.view(1, t)).mean()


def temporal_consistency_loss(
    pred,
    target,
    loss_type = "l1",
):
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    if loss_type == "mse":
        return F.mse_loss(pred_delta, target_delta)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred_delta, target_delta)
    return F.l1_loss(pred_delta, target_delta)


@torch.no_grad()
def validate(model, loader, device, args, epoch, vis_dir = None):
    model.eval()
    evaluator = Evaluator(thresholds=THRESHOLDS, value_scale=PIXEL_SCALE)
    saved_vis = False

    color_enabled = not args.no_color
    prefix = color(f"Valid Epoch {epoch}/{args.epochs}", C.MAGENTA + C.BOLD, color_enabled)
    progress = SimpleProgress(
        total=len(loader),
        prefix=prefix,
        width=args.progress_width,
        color_enabled=color_enabled,
        bar_color=C.MAGENTA,
    )

    start = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=True)
        x, y = split_batch(batch, args.frames_in, args.frames_out)
        pred, _ = model.predict(x, compute_loss=False)
        evaluator.update(pred, y)

        if vis_dir is not None and not saved_vis:
            sample_dir = vis_dir / f"epoch_{epoch:04d}"
            grid_path = sample_dir / "gt_pred_grid.png"
            save_gt_pred_grid(
                gt_seq=y[0],
                pred_seq=pred[0],
                save_path=str(grid_path),
                value_scale=PIXEL_SCALE,
            )
            print("\n" + tag("VIS", C.YELLOW, color_enabled) + f" Saved validation visualization to: {grid_path}")
            saved_vis = True

        need_print = step == 1 or step % max(args.val_log_interval, 1) == 0 or step == len(loader)
        extra = gpu_mem(device) if args.show_cuda_mem else ""
        progress.update(step, extra=extra, force=need_print)

    progress.close()
    metrics = evaluator.compute()
    metrics["val_time_sec"] = time.time() - start
    return metrics


def main():
    args = parse_args()
    enable_windows_ansi()
    color_enabled = not args.no_color

    seed_everything(args.seed)
    device = torch.device(args.device)

    run_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = run_dir / "checkpoints"
    vis_dir = run_dir / "valid_vis" if args.save_vis else None
    train_vis_dir = run_dir / "train_vis" if args.save_train_vis else None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(color("\n" + "=" * 92, C.GREEN, color_enabled))
    print(color(" MFPENet-CIKM Training  |  AlphaPre-style colored progress, no tqdm", C.GREEN + C.BOLD, color_enabled))
    print(color("=" * 92, C.GREEN, color_enabled))
    print(f"{tag('TIME', C.CYAN, color_enabled)} {now_time()}")

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "validation"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    train_set = CIKMFolderDataset(args.data_root, split="train", img_size=args.img_size,
                                  seq_len=args.seq_len, limit=args.train_limit or None)
    val_set = CIKMFolderDataset(args.data_root, split="validation", img_size=args.img_size,
                                seq_len=args.seq_len, limit=args.val_limit or None)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    if len(train_loader) == 0:
        raise RuntimeError("train_loader is empty. Please check batch_size, train_limit and dataset path.")
    if len(val_loader) == 0:
        raise RuntimeError("val_loader is empty. Please check batch_size, val_limit and dataset path.")

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device_name = str(device)
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        device_name = f"cuda:{idx} | {torch.cuda.get_device_name(idx)}"

    print_box("Run Info", {
        "data_root": data_root,
        "run_dir": run_dir,
        "checkpoints": ckpt_dir,
        "train_vis": train_vis_dir if train_vis_dir is not None else "disabled",
        "valid_vis": vis_dir if vis_dir is not None else "disabled",
        "device": device_name,
        "amp": args.amp and device.type == "cuda",
        "seed": args.seed,
    }, color_enabled)

    print_box("Dataset Info", {
        "train_samples": len(train_set),
        "valid_samples": len(val_set),
        "train_batches": len(train_loader),
        "valid_batches": len(val_loader),
        "frames_in/out": f"{args.frames_in}/{args.frames_out}",
        "seq_len": args.seq_len,
        "img_size": f"{args.img_size} x {args.img_size}",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }, color_enabled)

    print_box("Model Info", {
        "model": "MFPE-Net (five-module revised)",
        "use_phy_branch": not args.disable_phy,
        "intensity_module": "Multi-scale Intensity Dynamics",
        "position_module": "Local-frequency Phase Modulation",
        "transport_module": "Coarse-to-fine Physical Transport",
        "fusion_module": "Confidence-guided Evolution Fusion",
        "refiner_module": "High-Echo Temporal Residual Refiner",
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "spec_num": args.spec_num,
        "phy_base_c": args.phy_base_c,
        "max_motion": args.max_motion,
        "residual_scale": args.residual_scale,
        "parameters": f"{total_params / 1e6:.2f}M",
        "trainable_parameters": f"{trainable_params / 1e6:.2f}M",
    }, color_enabled)

    print_box("Loss Weights", {
        "pha_weight": args.pha_weight,
        "amp_weight": args.amp_weight,
        "anet_weight": args.anet_weight,
        "phy_weight": args.phy_weight,
        "aw_stop_step": args.aw_stop_step,
        "lead_loss_weight": args.lead_loss_weight,
        "lead_weight_start/end": f"{args.lead_weight_start}/{args.lead_weight_end}",
        "lead_loss_type": args.lead_loss_type,
        "temporal_weight": args.temporal_weight,
        "temporal_loss_type": args.temporal_loss_type,
    }, color_enabled)

    start_epoch = 0
    global_step = 0
    best_score = -1.0

    if args.pretrained:
        missing, unexpected = load_model_weights(model, args.pretrained, device, strict=args.strict_load)
        print(f"\n{tag('LOAD', C.GREEN, color_enabled)} Loaded pretrained weights: {args.pretrained}")
        if missing:
            print(f"{tag('INFO', C.YELLOW, color_enabled)} Missing keys: {len(missing)}. Example: {missing[:8]}")
        if unexpected:
            print(f"{tag('INFO', C.YELLOW, color_enabled)} Unexpected keys: {len(unexpected)}. Example: {unexpected[:8]}")

    if args.resume:
        start_epoch, global_step, best_score, missing, unexpected = resume_checkpoint(
            args.resume, model, optimizer, scheduler, device,
            load_optimizer=not args.no_resume_optimizer,
            strict=args.strict_load,
        )
        print(f"\n{tag('RESUME', C.GREEN, color_enabled)} Resumed from: {args.resume}")
        print(f"{tag('RESUME', C.GREEN, color_enabled)} start_epoch={start_epoch}, global_step={global_step}, best_score={best_score:.5f}")
        print(f"{tag('RESUME', C.GREEN, color_enabled)} load_optimizer={not args.no_resume_optimizer}")
        if missing:
            print(f"{tag('INFO', C.YELLOW, color_enabled)} Missing keys: {len(missing)}. Example: {missing[:8]}")
        if unexpected:
            print(f"{tag('INFO', C.YELLOW, color_enabled)} Unexpected keys: {len(unexpected)}. Example: {unexpected[:8]}")

    model.set_global_step(global_step)
    profile_info = print_model_profile(model, args, device, color_enabled)
    model.set_global_step(global_step)

    patience = 0
    all_start_time = time.time()

    print(color("\n" + "-" * 92, C.GREEN, color_enabled))
    print(color(" Start Training", C.GREEN + C.BOLD, color_enabled))
    print(color("-" * 92, C.GREEN, color_enabled))

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()

        loss_smooth: Dict[str, float] = {}
        loss_sum: Dict[str, float] = {}
        batch_count = 0
        saved_train_vis = False

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"\n{tag('EPOCH', C.BLUE + C.BOLD, color_enabled)} "
            f"{color(str(epoch), C.YELLOW + C.BOLD, color_enabled)}/{args.epochs - 1} "
            f"| lr={lr_now:.2e} | global_step={global_step}"
        )

        prefix = color(f"Epoch {epoch}/{args.epochs}", C.GREEN + C.BOLD, color_enabled)
        progress = SimpleProgress(
            total=len(train_loader),
            prefix=prefix,
            width=args.progress_width,
            color_enabled=color_enabled,
            bar_color=C.GREEN,
        )

        for step, batch in enumerate(train_loader, start=1):
            iter_start = time.time()
            batch = batch.to(device, non_blocking=True)
            x, y = split_batch(batch, args.frames_in, args.frames_out)

            model.set_global_step(global_step)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(args, device):
                pred, loss_dict = model.predict(x, y, compute_loss=True)
                base_loss = loss_dict["total_loss"]
                lead_loss_raw = lead_time_weighted_loss(
                    pred,
                    y,
                    weight_start=args.lead_weight_start,
                    weight_end=args.lead_weight_end,
                    loss_type=args.lead_loss_type,
                ) if args.lead_loss_weight > 0 else pred.new_tensor(0.0)
                temporal_loss_raw = temporal_consistency_loss(
                    pred,
                    y,
                    loss_type=args.temporal_loss_type,
                ) if args.temporal_weight > 0 else pred.new_tensor(0.0)

                lead_loss = args.lead_loss_weight * lead_loss_raw
                temporal_loss = args.temporal_weight * temporal_loss_raw
                loss = base_loss + lead_loss + temporal_loss

                loss_dict = dict(loss_dict)
                loss_dict["base_loss"] = base_loss
                loss_dict["lead_loss"] = lead_loss
                loss_dict["lead_loss_raw"] = lead_loss_raw
                loss_dict["temporal_loss"] = temporal_loss
                loss_dict["temporal_loss_raw"] = temporal_loss_raw
                loss_dict["total_loss"] = loss

            scaler.scale(loss).backward()

            grad_norm = None
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if train_vis_dir is not None and not saved_train_vis:
                sample_dir = train_vis_dir / f"epoch_{epoch:04d}"
                grid_path = sample_dir / "gt_pred_grid.png"
                save_gt_pred_grid(
                    gt_seq=y[0],
                    pred_seq=pred[0],
                    save_path=str(grid_path),
                    value_scale=PIXEL_SCALE,
                )
                print("\n" + tag("TRAIN_VIS", C.YELLOW, color_enabled) + f" Saved train visualization to: {grid_path}")
                saved_train_vis = True

            batch_count += 1
            for k, v in loss_dict.items():
                value = to_float(v)
                loss_smooth[k] = loss_smooth.get(k, value) * 0.98 + value * 0.02 if k in loss_smooth else value
                loss_sum[k] = loss_sum.get(k, 0.0) + value

            global_step += 1

            need_print = (
                step == 1
                or step % max(args.log_interval, 1) == 0
                or step == len(train_loader)
            )

            extra_items = []
            if args.show_grad and grad_norm is not None:
                extra_items.append(f"grad_norm={to_float(grad_norm):.3f}")
            if args.show_iter_time:
                extra_items.append(f"iter_time={time.time() - iter_start:.2f}s")
            if args.show_cuda_mem:
                m = gpu_mem(device)
                if m:
                    extra_items.append(m)

            progress.update(
                step,
                metrics=loss_smooth,
                lr=optimizer.param_groups[0]["lr"],
                extra=" | ".join(extra_items),
                force=need_print,
            )

        progress.close()

        train_avg = {k: v / max(batch_count, 1) for k, v in loss_sum.items()}
        scheduler.step()

        train_time = time.time() - epoch_start_time
        train_sample_fps = (batch_count * args.batch_size) / max(train_time, 1e-12)
        train_output_frame_fps = train_sample_fps * args.frames_out
        avg_text = " | ".join(
            f"{k}={v:.5f}" for k, v in train_avg.items()
            if k in ["total_loss", "ampli_loss", "intensity_loss", "physical_loss", "refinement_loss"]
        )
        print(
            f"{tag('TRAIN', C.GREEN, color_enabled)} epoch={epoch} "
            f"| time={fmt_seconds(train_time)} "
            f"| train_fps={train_sample_fps:.2f} samples/s "
            f"| train_frame_fps={train_output_frame_fps:.2f} frames/s "
            f"| {avg_text}"
        )

        metrics = validate(model, val_loader, device, args, epoch, vis_dir=vis_dir)

        score = metrics.get("CSI-M", -1.0)
        val_text = format_metrics({k: v for k, v in metrics.items() if k != "val_time_sec"})
        improved = score > best_score

        if improved:
            status = color("NEW BEST", C.GREEN + C.BOLD, color_enabled)
        else:
            status = color(f"no improve ({patience + 1})", C.YELLOW, color_enabled)

        print(
            f"{tag('VALID', C.MAGENTA, color_enabled)} epoch={epoch} "
            f"| time={fmt_seconds(metrics.get('val_time_sec', 0.0))} "
            f"| {val_text} | {status}"
        )

        # Full epoch log: training summary + validation metrics.
        with open(run_dir / "metrics_log.jsonl", "a", encoding="utf-8") as f:
            log_obj = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "train_time_sec": train_time,
                "train_sample_fps": train_sample_fps,
                "train_output_frame_fps": train_output_frame_fps,
                "train_avg": train_avg,
                "profile_info": profile_info,
                **metrics,
            }
            f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        with open(run_dir / "metrics_val_log.jsonl", "a", encoding="utf-8") as f:
            val_log_obj = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": scheduler.get_last_lr()[0],
                **metrics,
            }
            f.write(json.dumps(val_log_obj, ensure_ascii=False) + "\n")

        save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, scheduler,
                        epoch, global_step, best_score, vars(args))
        print(f"{tag('SAVE', C.CYAN, color_enabled)} last checkpoint -> {ckpt_dir / 'last.pt'}")

        if improved:
            best_score = score
            patience = 0
            save_checkpoint(str(ckpt_dir / "best.pt"), model, optimizer, scheduler,
                            epoch, global_step, best_score, vars(args))
            print(f"{tag('BEST', C.GREEN + C.BOLD, color_enabled)} best checkpoint -> {ckpt_dir / 'best.pt'} | CSI-M={best_score:.5f}")
        else:
            patience += 1

        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"{tag('STOP', C.YELLOW + C.BOLD, color_enabled)} Early stopping: no CSI-M improvement for {patience} epochs.")
            break

    total_time = time.time() - all_start_time
    print(color("\n" + "=" * 92, C.GREEN, color_enabled))
    print(color(" Training Finished", C.GREEN + C.BOLD, color_enabled))
    print(color("=" * 92, C.GREEN, color_enabled))
    print(f"{tag('TIME', C.CYAN, color_enabled)} total_time={fmt_seconds(total_time)}")
    print(f"{tag('CKPT', C.CYAN, color_enabled)} last: {ckpt_dir / 'last.pt'}")
    print(f"{tag('CKPT', C.CYAN, color_enabled)} best: {ckpt_dir / 'best.pt'}")
    print(f"{tag('LOG', C.CYAN, color_enabled)} metrics: {run_dir / 'metrics_log.jsonl'}")
    print(f"{tag('LOG', C.CYAN, color_enabled)} val metrics: {run_dir / 'metrics_val_log.jsonl'}")
    if train_vis_dir is not None:
        print(f"{tag('VIS', C.CYAN, color_enabled)} train images: {train_vis_dir}")
    if vis_dir is not None:
        print(f"{tag('VIS', C.CYAN, color_enabled)} valid images: {vis_dir}")


if __name__ == "__main__":
    main()
