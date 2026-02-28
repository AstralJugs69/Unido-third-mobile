from __future__ import annotations

import numpy as np


def split_to_grid(image: np.ndarray, grid_rows: int, grid_cols: int) -> list[np.ndarray]:
    h, w, _ = image.shape
    step_h = h // grid_rows
    step_w = w // grid_cols
    tiles: list[np.ndarray] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * step_h
            x1 = c * step_w
            y2 = (r + 1) * step_h if r < grid_rows - 1 else h
            x2 = (c + 1) * step_w if c < grid_cols - 1 else w
            tiles.append(image[y1:y2, x1:x2])
    return tiles
