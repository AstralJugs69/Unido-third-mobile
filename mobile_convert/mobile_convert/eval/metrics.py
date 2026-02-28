from __future__ import annotations

import numpy as np


def mae_per_target(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - target), axis=0)


def total_mae(count_mae: np.ndarray, measure_mae: np.ndarray) -> float:
    return float((count_mae.mean() * len(count_mae) + measure_mae.mean() * len(measure_mae)) / (len(count_mae) + len(measure_mae)))
