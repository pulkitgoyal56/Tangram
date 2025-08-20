"""Grid class."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Grid:
    """Grid class."""
    def __init__(self, start=(0, 1), end=(0, 1)):
        self.boundaries = np.asarray([start, end])

    def __contains__(self, pos):
        return np.all((pos >= self.boundaries[0]) & (pos < self.boundaries[1]))

    def __repr__(self):
        return f"Grid({self.boundaries[0]} : {self.boundaries[1]})"

    @property
    def area(self):
        return np.prod(np.diff(self.boundaries, axis=0))

    @staticmethod
    def normalize(boundaries, steps):
        boundaries = np.asarray(boundaries)
        if boundaries.dtype == int:
            boundaries = boundaries / steps
        assert np.all(boundaries <= 1), f"Boundaries must be homogenous, got {boundaries}."
        return boundaries

    @staticmethod
    def xy2rc(xy, *, shape, size=[1., 1.]):
        # Shape is r_c
        # Size  is x_y
        xy = np.asarray(xy)
        if np.any(xy > size):
            logger.warning(f"> {xy} is outside size {size}.")
        return (np.floor(xy / size * shape[::-1]).astype(int) * [1, -1] + [0, shape[0] - 1]).T[::-1].T

    @staticmethod
    def rc2xy(rc, *, shape, size=[1., 1.]):
        # Shape is r_c
        # Size  is x_y
        rc = np.asarray(rc)
        if np.any(rc > shape):
            logger.warning(f"> {rc} is outside shape {shape}.")
        return (([1., 0] - (rc + [1, 0]) / shape) * [1, -1]).T[::-1].T * size
