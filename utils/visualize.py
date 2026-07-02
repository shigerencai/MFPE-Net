from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


COLOR_MAP = np.array([
    [0, 0, 0, 255],          # dBZ -10 -- -5  无色 / black
    [0, 245, 255, 255],      # dBZ  -5 --  0  light blue
    [30, 144, 255, 255],     # dBZ   0 --  5  medium blue
    [0, 0, 255, 255],        # dBZ   5 -- 10  deep blue
    [0, 255, 0, 255],        # dBZ  10 -- 15  green
    [50, 205, 50, 255],      # dBZ  15 -- 20  medium green
    [34, 139, 34, 255],      # dBZ  20 -- 25  deep green
    [255, 255, 0, 255],      # dBZ  25 -- 30  yellow
    [218, 165, 32, 255],     # dBZ  30 -- 35  dark golden yellow
    [255, 69, 0, 255],       # dBZ  35 -- 40  orange red
    [255, 0, 0, 255],        # dBZ  40 -- 45  red
    [205, 38, 38, 255],      # dBZ  45 -- 50  medium red
    [139, 26, 26, 255],      # dBZ  50 -- 55  deep red
    [255, 20, 147, 255],     # dBZ  55 -- 60  pink
    [139, 10, 80, 255],      # dBZ  60 -- 65  deep pink
    [255, 255, 255, 255],    # dBZ >65         white
], dtype=np.uint8)
DBZ_BOUNDS = np.array([-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65], dtype=np.float32)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def normalized_to_dbz(x, value_scale = 80.0, dbz_min = -10.0):
    x = np.clip(x, 0.0, 1.0)
    return x * float(value_scale) + float(dbz_min)


def gray_to_color(x, value_scale = 80.0):
    dbz = normalized_to_dbz(x, value_scale=value_scale, dbz_min=-10.0)
    idx = np.searchsorted(DBZ_BOUNDS, dbz, side="right") - 1
    idx = np.clip(idx, 0, len(COLOR_MAP) - 1)
    return COLOR_MAP[idx]


def save_sequence(seq, out_dir, prefix, value_scale = 80.0, color = True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.clip(_to_numpy(seq), 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[:, 0]
    for i, frame in enumerate(arr, start=1):
        if color:
            img = Image.fromarray(gray_to_color(frame, value_scale=value_scale), mode="RGBA")
        else:
            img = Image.fromarray((frame * 255).astype(np.uint8), mode="L")
        img.save(out / f"{prefix}_{i:02d}.png")


def _prepare_sequence(seq):
    arr = np.clip(_to_numpy(seq), 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected sequence shape [T,1,H,W] or [T,H,W], got {arr.shape}")
    return arr


def save_gt_pred_grid(
    gt_seq,
    pred_seq,
    save_path,
    value_scale = 80.0,
    pad = 4,
    add_labels = True,
    background=(255, 255, 255, 255),
):
    gt = _prepare_sequence(gt_seq)
    pred = _prepare_sequence(pred_seq)
    T = min(gt.shape[0], pred.shape[0])
    if T <= 0:
        raise ValueError("Empty GT/Pred sequence, cannot save grid.")
    gt = gt[:T]
    pred = pred[:T]

    gt_imgs = [Image.fromarray(gray_to_color(frame, value_scale=value_scale), mode="RGBA") for frame in gt]
    pred_imgs = [Image.fromarray(gray_to_color(frame, value_scale=value_scale), mode="RGBA") for frame in pred]

    w, h = gt_imgs[0].size
    label_h = 22 if add_labels else 0
    row_h = h + label_h
    canvas_w = T * w + (T - 1) * pad
    canvas_h = 2 * row_h + pad

    canvas = Image.new("RGBA", (canvas_w, canvas_h), background)
    draw = ImageDraw.Draw(canvas)

    # Row 1: ground truth
    for i, img in enumerate(gt_imgs):
        x = i * (w + pad)
        y = 0
        canvas.paste(img, (x, y))
        if add_labels:
            draw.text((x + 4, h + 3), f"GT T+{i + 1}", fill=(0, 0, 0, 255))

    # Row 2: prediction
    pred_y = row_h + pad
    for i, img in enumerate(pred_imgs):
        x = i * (w + pad)
        canvas.paste(img, (x, pred_y))
        if add_labels:
            draw.text((x + 4, pred_y + h + 3), f"Pred T+{i + 1}", fill=(0, 0, 0, 255))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)

