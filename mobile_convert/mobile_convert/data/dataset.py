from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from .schema import COUNT_COLS, MEASURE_COLS, RICE_TYPE_MAP
from .tiling import split_to_grid


class RiceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        tile_size: int,
        grid_rows: int,
        grid_cols: int,
        measure_stats: tuple[np.ndarray, np.ndarray] | None,
        train_mode: bool = True,
        require_targets: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.measure_stats = measure_stats
        self.require_targets = require_targets

        if train_mode:
            self.transform = A.Compose(
                [
                    A.Resize(tile_size, tile_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose([A.Resize(tile_size, tile_size), A.Normalize(), ToTensorV2()])

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_id: str) -> np.ndarray:
        path = self.image_dir / f"{image_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return np.array(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_image(row["ID"])
        tiles = split_to_grid(img, self.grid_rows, self.grid_cols)
        tile_tensors = [self.transform(image=t)["image"] for t in tiles]
        stack = torch.stack(tile_tensors)

        rice_type = RICE_TYPE_MAP.get(str(row["Comment"]), 0)
        meta = torch.zeros(3, dtype=torch.float32)
        meta[rice_type] = 1.0

        if not self.require_targets:
            counts = torch.zeros(len(COUNT_COLS), dtype=torch.float32)
            measures = torch.zeros(len(MEASURE_COLS), dtype=torch.float32)
            return stack, meta, counts, measures, rice_type, row["ID"]

        counts = torch.tensor(row[COUNT_COLS].astype("float32").values, dtype=torch.float32)
        measures_raw = row[MEASURE_COLS].astype("float32").values

        if self.measure_stats is None:
            measures = torch.tensor(measures_raw, dtype=torch.float32)
        else:
            mean, std = self.measure_stats
            measures = torch.tensor((measures_raw - mean) / (std + 1e-8), dtype=torch.float32)

        return stack, meta, counts, measures, rice_type, row["ID"]
