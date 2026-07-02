import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


PIXEL_SCALE = 90.0
THRESHOLDS = [20, 30, 35, 40]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _numeric_key(path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def resize_or_crop(x, size, mode = "resize"):
    _, _, h, w = x.shape
    if h == size and w == size:
        return x
    if mode == "resize":
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if mode != "center_crop":
        raise ValueError(f"Unsupported resize_mode={mode}; choose resize or center_crop")

    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        x = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)
        _, _, h, w = x.shape
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return x[:, :, top:top + size, left:left + size]


class ShanghaiFolderDataset(Dataset):
    def __init__(
        self,
        root,
        split = "train",
        img_size = 128,
        seq_len = 20,
        limit = None,
        strict = True,
        resize_mode = "resize",
    ):
        super().__init__()
        self.root = Path(root)
        split_alias = {"val": "validation", "valid": "validation"}
        self.split = split_alias.get(split, split)
        self.img_size = int(img_size)
        self.seq_len = int(seq_len)
        self.strict = bool(strict)
        self.resize_mode = resize_mode

        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Cannot find split directory: {split_dir}\n"
                f"Expected: {self.root}/train, {self.root}/validation, {self.root}/test"
            )

        sample_dirs = [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")]
        sample_dirs = sorted(sample_dirs, key=_numeric_key)
        if limit is not None and limit > 0:
            sample_dirs = sample_dirs[: int(limit)]
        if len(sample_dirs) == 0:
            raise RuntimeError(f"No sample_* folders found under {split_dir}")
        self.sample_dirs: List[Path] = sample_dirs

    def __len__(self):
        return len(self.sample_dirs)

    def _list_frames(self, sample_dir):
        files = [p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        files = sorted(files, key=_numeric_key)
        if self.strict and len(files) < self.seq_len:
            raise FileNotFoundError(
                f"Sample {sample_dir} has only {len(files)} image files, need seq_len={self.seq_len}."
            )
        return files[: self.seq_len]

    def _load_frame(self, path):
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        frame_paths = self._list_frames(sample_dir)
        if len(frame_paths) == 0:
            raise RuntimeError(f"No image files found in {sample_dir}")

        frames = [self._load_frame(p) for p in frame_paths]
        x = torch.stack(frames, dim=0)
        x = resize_or_crop(x, self.img_size, self.resize_mode)
        return x.contiguous()
