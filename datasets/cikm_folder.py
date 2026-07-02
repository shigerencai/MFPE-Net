import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


def _numeric_key(path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def pad_or_center_crop(x, size):
    _, _, h, w = x.shape
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


class CIKMFolderDataset(Dataset):
    def __init__(
        self,
        root,
        split = "train",
        img_size = 128,
        seq_len = 15,
        limit = None,
        strict = True,
    ):
        super().__init__()
        self.root = Path(root)
        split_alias = {"val": "validation", "valid": "validation"}
        self.split = split_alias.get(split, split)
        self.img_size = img_size
        self.seq_len = seq_len
        self.strict = strict

        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Cannot find split directory: {split_dir}\n"
                f"Expected: {self.root}/train, {self.root}/validation, {self.root}/test"
            )

        sample_dirs = [p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")]
        sample_dirs = sorted(sample_dirs, key=_numeric_key)
        if limit is not None and limit > 0:
            sample_dirs = sample_dirs[:limit]
        if len(sample_dirs) == 0:
            raise RuntimeError(f"No sample_* folders found under {split_dir}")
        self.sample_dirs: List[Path] = sample_dirs

    def __len__(self):
        return len(self.sample_dirs)

    def _load_frame(self, path):
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

    def __getitem__(self, index):
        sample_dir = self.sample_dirs[index]
        frames = []
        for t in range(1, self.seq_len + 1):
            img_path = sample_dir / f"img_{t}.png"
            if not img_path.exists():
                if self.strict:
                    raise FileNotFoundError(f"Missing frame: {img_path}")
                break
            frames.append(self._load_frame(img_path))
        if len(frames) == 0:
            raise RuntimeError(f"No frames found in {sample_dir}")
        x = torch.stack(frames, dim=0)  # [T, 1, H, W]
        x = pad_or_center_crop(x, self.img_size)
        return x.contiguous()


PIXEL_SCALE = 80.0
THRESHOLDS = [20, 30, 35, 40]
