import numpy as np

from mobile_convert.data.tiling import split_to_grid


def test_tiling_count_and_shape():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    tiles = split_to_grid(img, 6, 8)
    assert len(tiles) == 48
    assert tiles[0].ndim == 3
