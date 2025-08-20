"""Polygon list class."""

from __future__ import annotations
import logging
from typing import List
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg

# from skimage.draw import polygon as sk_polygon
from cv2 import fillPoly, fillConvexPoly

from polygon import Polygon
from self import Self
# from grid import Grid
import constants

logger = logging.getLogger(__name__)


class PolygonList(Iterable):
    """Polygon list class."""
    def __init__(self, polygons: List[Polygon]):
        self._polygons = polygons

        dims = self.get("dim")
        assert len(set(dims)) == 1, f"> Incompatible dimensions {dims}."
        self.dim = dims[0]

    def _validate(self):
        for i in range(len(self)):
            # if not isinstance(self.polygons[i], Polygon):
            #     raise ValueError("> Polygons must be `Polygon` objects.")
            for j in range(i + 1, len(self)):
                if Polygon.collides(self.polygons[i].vertices, self.polygons[j].vertices):
                    return False
        return True

    @property
    def polygons(self):
        return self._polygons

    @polygons.setter
    def polygons(self, polygons):
        self._polygons = polygons
        # if not self._validate():
        #     raise ValueError("> Invalid polygons.")

    def get(self, attr):
        if hasattr(Polygon, attr) and attr not in ['__deepcopy__']:
            return [getattr(polygon, attr) for polygon in self.polygons]
        else:
            raise AttributeError(f"> PolygonList or Polygon has no attribute {attr}.")
    __getattr__ = get

    def set(self, attr, values):
        for polygon, value in zip(self.polygons, values):
            assert hasattr(polygon, attr), f"> Polygon does not have attribute {attr}."
            setattr(polygon, attr, value)

    @property
    def colors(self):
        return [polygon.color for polygon in self.polygons]

    @colors.setter
    def colors(self, colors):
        self.set("color", colors)

    @property
    def positions(self):
        return np.asarray(self.get("position"))

    @positions.setter
    def positions(self, positions):
        self.set("position", positions)

    @property
    def angles(self):
        return np.asarray(self.get("angle"))

    @angles.setter
    def angles(self, angles):
        self.set("angle", angles)

    @property
    def flips(self):
        return np.asarray(self.get("flipped"))

    @flips.setter
    def flips(self, flips):
        self.set("flipped", flips)

    @property
    def areas(self):
        return np.asarray(self.get("area"))

    @property
    def centroids(self):
        return np.asarray(self.get("centroid"))

    @property
    def centers(self):
        return np.asarray(self.get("center"))

    @property
    def pivots(self):
        return np.asarray(self.get("pivot"))

    @property
    def min(self):
        return min(self.get("min"))

    @property
    def max(self):
        return max(self.get("max"))

    @property
    def num_vertices(self):
        return np.asarray(self.get("num_vertices"))

    @property
    def size(self):
        return len(self)

    @property
    def n(self):
        return sum(self.num_vertices)

    @property
    def shape(self):
        return (self.n, self.dim)

    @property
    def vertices(self):
        return np.r_[(*[polygon.vertices for polygon in self.polygons],)]

    @vertices.setter
    def vertices(self, vertices):
        # raise ValueError("> Setting vertices of PolygonList directly not allowed!")
        if not isinstance(vertices, np.ndarray):
            vertices = np.asarray(vertices)
        if vertices.shape == (self.dim * self.n,):
            vertices = vertices.reshape(-1, self.dim)
        # if vertices.shape == (self.dim * self.n + 2,):
        #     vertices = vertices[:-2].reshape(-1, self.dim)
        if vertices.shape == (self.n, self.dim):
            vertices = np.split(vertices, np.cumsum(self.num_vertices)[:-1])
        for polygon, vertices_i in zip(self.polygons, vertices):
            assert len(vertices_i) == polygon.num_vertices, f"> Invalid number of vertices {len(vertices_i)} for polygon {polygon} with {polygon.num_vertices} vertices."
            polygon.vertices = vertices_i

    @property
    def rel_vertices(self):
        return np.r_[(*[polygon.rel_vertices for polygon in self.polygons],)]

    @rel_vertices.setter
    def rel_vertices(self, rel_vertices):
        raise ValueError("> Setting vertices of PolygonList directly not allowed!")

    def transform(self, obj_id: int, dx, dy, xy, ax, *, inplace=True):
        return self.polygons[obj_id].transform(dx, dy, xy, ax, inplace=inplace)

    def __repr__(self):
        return f"PolygonList({len(self)})"

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return PolygonList(self.polygons[idx])
        elif isinstance(idx, int):
            return self.polygons[idx]
        elif isinstance(idx, (tuple, list, np.ndarray)):
            if all(isinstance(i, bool) for i in idx):
                return PolygonList([p for p, m in zip(self.polygons, idx) if m])
            elif all(isinstance(i, int) for i in idx):
                return PolygonList([self.polygons[i] for i in idx])
            else:
                raise ValueError(f"> Invalid index {idx}.")

    def __iter__(self):
        return iter(self.polygons)

    def __contains__(self, point):
        return any(point in polygon for polygon in self.polygons)

    def __add__(self, other: PolygonList):
        return PolygonList(self.polygons + other.polygons)

    def __matmul__(self, T: np.ndarray):
        # NOTE All transformations are applied on pivot-relative vertices.
        if len(T) == self.dim:
            return self.rel_vertices @ T
        elif len(T) == self.dim + 1:
            return (np.c_[self.rel_vertices, np.ones(self.n)] @ T)
        else:
            raise ValueError(f"> Invalid transformation matrix of shape {T.shape}.")

    def patches(self, *, color=True, **kwargs):
        return PatchCollection([polygon.patch(**kwargs) for polygon in self.polygons], facecolor=(0, 0, 0) if not color else [[c / 255 for c in colors] for colors in self.colors], **kwargs)

    def render_fast(self, size=(224, 224), *, color=True, lineType=8, **kwargs):
        if color:
            canvas = np.full((*size[::-1], 3), 255, dtype=np.uint8)
            for polygon in self.polygons:
                # cc, rr = sk_polygon(*(polygon.vertices * size).T, (*size, 3))  # **kwargs
                # canvas[size[1] - 1 - rr, cc] = polygon.color
                fillConvexPoly(canvas, [0, size[1] - 1] + (polygon.vertices * [size[0], -size[1]]).astype(int), color=polygon.color, lineType=lineType)  # **kwargs
        else:
            canvas = np.full(size[::-1], 255, dtype=np.uint8)
            fillPoly(canvas, [[0, size[1] - 1] + (polygon.vertices * [size[0], -size[1]]).astype(int) for polygon in self.polygons], color=0, lineType=lineType)  # **kwargs
        return canvas

    def render(self, figsize=constants.FIG_SIZE, dpi=constants.DPI, title='', *, xlim=(0, 1), ylim=(0, 1), fast=True, **kwargs):
        if fast:
            size = [int(s * dpi / (lim[1] - lim[0])) for s, lim in zip(figsize, (xlim, ylim))]
            return self.render_fast(
                size, **kwargs
            )[
                int((1 - ylim[1]) * size[1]) : int((1 - ylim[0]) * size[1]),
                int(xlim[0] * size[0]) : int(xlim[1] * size[0]),
            ]
        else:
            ax = kwargs.pop('ax', None)
            figure = kwargs.pop('figure', None)

            if ax is None:
                if figure is None:
                    figure = plt.Figure(figsize=figsize, dpi=dpi)  # CURI using `plt.figure`/`plt.subplots` leads to plot display without call!
                ax = figure.gca()
            elif figure is None:
                figure = ax.get_figure()

            ax.add_collection(self.patches(**kwargs))
            ax.set_title(title)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.axis("off")
            figure.tight_layout(pad=0)

            (canvas := FigureCanvasAgg(figure)).draw()
            return np.frombuffer(bytearray(canvas.buffer_rgba()), dtype=np.uint8).reshape(*reversed(canvas.get_width_height()), -1)

    @Self("vertices", 'Polygon')
    def collides(self, vertices, obj_id=-1):
        for i in range(len(self)):
            if i != obj_id and Polygon.collides(self.polygons[i], vertices):
                return True
        return False
