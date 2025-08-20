"""Polygon class."""

import logging
from typing import Iterable
from cmath import phase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
# https://github.com/matplotlib/matplotlib/blob/v3.8.1/lib/matplotlib/path.py
from matplotlib.patches import Polygon as PolygonPatch

from self import Self
import constants

logger = logging.getLogger(__name__)


class Polygon(path.Path):
    """Polygon class."""
    def __init__(self, vertices, angle=0, flipped=False, *, scale=(1, 1), shear=(0, 0), name=None, color=None, **kwargs):
        super().__init__(vertices, readonly=True)
        # self.vertices = vertices

        # self.vertices_ = self.vertices.copy()

        self._angle = angle
        self._flipped = flipped
        self._position = self.get_pivot()

        self.name = name
        self.color = color if color else constants.DEFAULT_COLOR

        self.scale(*scale)
        self.shear(*shear)

        # self._validate(**kwargs)

    def _validate(self, convex=True):
        if convex and not self.is_convex():
            raise ValueError("> Vertices do not form a convex polygon. Vertices must be in order.")
        if self.dim != 2:
            raise NotImplementedError("> Only 2D polygons supported.")

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # if self._vertices is not None:
        #     raise ValueError("> Setting vertices not allowed!")
        self._vertices = np.asarray(vertices)
        self._angle = 0
        self._flipped = False
        self._position = self.get_pivot()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if not all(self.position == position):
            self.translate(*np.asarray(position) - self.position)
    pivot = position

    @Self("_angle", "Polygon")
    def get_angle(angle):
        return (angle + 1) % 2 - 1
    angle = property(get_angle)

    @angle.setter
    def angle(self, angle):
        self.rotate(angle - self.angle)

    @property
    def flipped(self):
        return self._flipped

    @flipped.setter
    def flipped(self, flipped):
        if flipped != self.flipped:
            self.flip()

    @property
    def num_vertices(self):
        return len(self)
    n = size = num_vertices

    @property
    def shape(self):
        return self.vertices.shape

    @property
    def dim(self):
        return self.vertices.shape[1]

    @property
    def min(self):
        return np.min(self.vertices, axis=0)

    @property
    def max(self):
        return np.max(self.vertices, axis=0)

    @Self("vertices", "Polygon")
    def get_area(vertices):
        """Returns area of polygon."""
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1] - vertices[i][1] * vertices[j][0]
        return abs(area) / 2.0
    area = property(get_area)

    @Self("vertices", "Polygon")
    def get_centroid(vertices):
        """Returns centroid of polygon."""
        # https://en.wikipedia.org/wiki/Polygon#Centroid
        n = len(vertices)
        if len(vertices) == 3:
            return np.mean(vertices, axis=0)
        cx = cy = 0
        for i in range(n):
            j = (i + 1) % n
            cx += (vertices[i][0] + vertices[j][0]) * (vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1])
            cy += (vertices[i][1] + vertices[j][1]) * (vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1])
        area = Polygon.get_area(vertices)
        return np.asarray([cx / 6 * area, cy / 6 * area])
    centroid = property(get_centroid)

    @Self("vertices", "Polygon")
    def get_center(vertices):
        """Returns center of polygon."""
        return np.mean(vertices, axis=0)
    center = property(get_center)

    @Self("vertices", "Polygon")
    def get_pivot(vertices, *, method=get_center):  # get_centroid
        """Returns pivot of polygon."""
        return method(vertices)
    # get_position = get_pivot

    @property
    def rel_vertices(self):
        return self.vertices - self.pivot

    @rel_vertices.setter
    def rel_vertices(self, rel_vertices):
        self._vertices = self.pivot + rel_vertices

    def __repr__(self):
        return f"{len(self)}-Polygon({self.name + ' | ' if self.name else ''}{tuple(self.position)}, {self.angle * 180:.0f}°)"

    # def __len__(self):
    #     return len(self.vertices)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.vertices[idx]
        elif isinstance(idx, (tuple, list, np.ndarray)):
            if all(isinstance(i, bool) for i in idx):
                return [v for v, m in zip(self.vertices, idx) if m]
            elif all(isinstance(i, int) for i in idx):
                return [self.vertices[i] for i in idx]
            else:
                raise ValueError(f"> Invalid index {idx}.")

    def __iter__(self):
        return iter(self.vertices)

    def __contains__(self, point):
        """Checks if point is inside polygon."""
        # https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
        # x, y = point

        # min_x, min_y = self.min
        # max_x, max_y = self.max
        # if not (min_x <= x < max_x and min_y <= y < max_y):
        #     return False

        # ex, ey = self.vertices.T
        # ex_, ey_ = np.roll(self.vertices, -1, axis=0).T
        # return bool((((ey > y) != (ey_ > y)) & (x < (ex_ - ex) * (y - ey) / (ey_ - ey) + ex)).sum() % 2)

        return self.contains_point(point)

    def __add__(self, d):
        if isinstance(d, Iterable) and len(d) == self.dim:
            return Polygon(self.translate(*d, inplace=False))
        else:
            raise ValueError(f"> Invalid operand {(type(d), len(d))}.")

    def __sub__(self, d):
        if isinstance(d, Iterable) and len(d) == self.dim:
            return Polygon(self.translate(*[-o for o in d], inplace=False))
        else:
            raise ValueError(f"> Invalid operand {(type(d), len(d))}.")

    def __mul__(self, s):
        if isinstance(s, (int, float, complex)):
            Polygon((self.rotate(phase(s), normalized=False, inplace=False) - self.pivot) * abs(s) + self.pivot)
        else:
            raise ValueError(f"> Invalid operand {type(s)}.")

    def __truediv__(self, s):
        if isinstance(s, (int, float, complex)):
            Polygon((self.rotate(-phase(s), normalized=False, inplace=False) - self.pivot) / abs(s) + self.pivot)
        else:
            raise ValueError(f"> Invalid operand {type(s)}.")

    def __matmul__(self, T: np.ndarray):
        # NOTE All transformations are applied on pivot-relative vertices.
        if len(T) == self.dim:
            return self.rel_vertices @ T
        elif len(T) == self.dim + 1:
            return (np.c_[self.rel_vertices, np.ones(self.num_vertices)] @ T)
        else:
            raise ValueError(f"> Invalid transformation matrix of shape {T.shape}.")

    @Self.inplace("_vertices")
    def scale(self, sx, sy=None, *, inplace=True):
        """Scales polygon via pivot."""
        # WARN Not tracked; use only in constructor.
        return self.pivot + self.rel_vertices * (sx, sy if sy else sx)

    @Self.inplace("_vertices")
    def shear(self, thx=0, thy=0, *, normalized=True, inplace=True):
        """Shears polygon via pivot."""
        # WARN Not tracked; use only in constructor.
        if normalized:
            thx, thy = thx * np.pi, thy * np.pi
        return self.pivot + self @ [[np.cos(thx), np.sin(thx)], [-np.sin(thy), np.cos(thy)]]

    def translate(self, dx, dy, *, inplace=True):
        """Translates polygon."""
        # assert abs(dx) <= 1 and abs(dy) <= 1, f"> Translation by ({dx, dy}) > 1."
        vertices = self.vertices + [dx, dy]
        if inplace:
            self._vertices = vertices
            self._position += (dx, dy)
            return self
        else:
            return vertices

    def rotate(self, xy, *, normalized=True, inplace=True):
        """Rotates polygon via pivot."""
        # xy is the normalized angle.
        if xy:
            xy_ = xy * np.pi if normalized else xy
            cth, sth = np.cos(xy_), np.sin(xy_)
            vertices = self.pivot + self @ [[cth, sth], [-sth, cth]]
            if inplace:
                self._vertices = vertices
                self._angle += Polygon.get_angle(xy)
                return self
            else:
                return vertices
        else:
            if inplace:
                return self
            else:
                return self.vertices

    def flip(self, ax=True, *, inplace=True):
        """Flips polygon via pivot."""
        if ax:
            # vertices = self.vertices - 2 * self.rel_vertices * [1, 0]  # (np.arange(self.dim) == ax)
            c2th, s2th = np.cos(2 * self.angle * np.pi), np.sin(2 * self.angle * np.pi)
            vertices = self.pivot + self @ [[-c2th, -s2th], [-s2th, c2th]]
            if inplace:
                self._vertices = vertices
                self._flipped = not self.flipped
                return self
            else:
                return vertices
        else:
            if inplace:
                return self
            else:
                return self.vertices

    def transform(self, dx, dy, xy=0, ax=False, *, inplace=True):
        """Transforms (translates, rotates, and flips) polygon via pivot."""
        rel_vertices = self.rel_vertices
        if ax:
            c2th, s2th = np.cos(2 * self.angle * np.pi), np.sin(2 * self.angle * np.pi)
            rel_vertices = rel_vertices @ [[-c2th, -s2th], [-s2th, c2th]]
        if xy:
            xy_ = xy * np.pi
            cth, sth = np.cos(xy_), np.sin(xy_)
            rel_vertices = rel_vertices @ [[cth, sth], [-sth, cth]]
        vertices = self.pivot + [dx, dy] + rel_vertices

        if inplace:
            self._vertices = vertices
            self._position += (dx, dy)
            if xy:
                self._angle += Polygon.get_angle(xy)
            if ax:
                self._flipped = not self.flipped
            return self
        else:
            return vertices

    def is_convex(self):
        """Checks if polygon is convex."""
        # https://stackoverflow.com/questions/471962/how-do-determine-if-a-polygon-is-complex-convex-nonconvex
        def direction(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        d = direction(self.vertices[0], self.vertices[1], self.vertices[2])
        for i in range(1, self.num_vertices):
            if direction(self.vertices[i], self.vertices[(i + 1) % self.num_vertices], self.vertices[(i + 2) % self.num_vertices]) * d < 0:
                return False
        return True

    def within_bounds(self, boundaries, *, criteria=None):
        if criteria is None:
            pass
        elif criteria == "complete":
            if np.any((self.vertices < boundaries[0]) | (self.vertices >= boundaries[1])):
                return False
        elif criteria == "partial":
            if np.any((self.vertices >= boundaries[0]) & (self.vertices < boundaries[1])):
                return True
            return False
        elif hasattr(self, criteria):
            attr = getattr(self, criteria)
            if np.any((attr < boundaries[0]) | (attr >= boundaries[1])):
                return False
        else:
            raise ValueError(f"> Unknown criteria - {criteria}.")
        return True

    def patch(self, **kwargs):
        return PolygonPatch(self.to_polygons()[0], facecolor=[c / 255 for c in self.color], **kwargs)

    def show(self, axes: plt.Axes = None, figure: plt.Figure = None, figsize=(5, 5), title='', **kwargs):
        show = False
        if axes is None:
            if figure is None:
                figure, axes = plt.subplots(1, 1, figsize=figsize)
                show = True
            else:
                axes = figure.gca()
        elif figure is None:
            figure = axes.get_figure()

        axes.add_patch(self.patch(**kwargs))
        axes.set_title(title if title else self)
        # axes.axis("off")
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

        if show:
            plt.show()
            return figure, axes

    @Self("vertices", "Polygon")
    def get_edges(vertices):
        return np.roll(vertices, -1, axis=0) - vertices
    edges = property(get_edges)

    @Self("vertices", "Polygon")
    def get_normals(vertices):
        return Polygon.get_edges(vertices) @ [[0, 1], [-1, 0]]

    @Self("vertices", "Polygon")
    def collides(vert1, vert2):
        """Checks whether two polygons collide."""
        # https://stackoverflow.com/questions/10962379/how-to-check-intersection-between-2-rotated-rectangles
        # for edge in chain(Polygon.get_normals(p), Polygon.get_normals(q)):
        #     r = p @ edge
        #     s = q @ edge
        #     if np.max(r) <= np.min(s) or np.max(s) <= np.min(r):
        #         return False
        # return True

        normals = np.r_[Polygon.get_normals(vert1), Polygon.get_normals(vert2)].T
        r = vert1 @ normals
        s = vert2 @ normals
        return np.all((np.max(r, axis=0) > np.min(s, axis=0)) & (np.max(s, axis=0) > np.min(r, axis=0)))

    @staticmethod
    def move_center_to_origin(vertices) -> np.ndarray:
        return np.asarray(vertices) - np.mean(vertices, axis=0)


class Triangle(Polygon):
    __VERTICES = Polygon.move_center_to_origin([(0, 0), (0, 1), (1, 0)]).tolist()

    def __init__(self, *args, **kwargs):
        super().__init__(self.__VERTICES, *args, **kwargs)

    def __repr__(self):
        return f"Triangle({self.name + ' | ' if self.name else ''}{tuple(self.position)}, {self.angle * 180:.0f}°)"


class Quadrilateral(Polygon):
    __VERTICES = Polygon.move_center_to_origin([(0, 0), (0, 1), (1, 1), (1, 0)]).tolist()

    def __init__(self, *args, **kwargs):
        super().__init__(self.__VERTICES, *args, **kwargs)

    def __repr__(self):
        return f"Quadrilateral({self.name + ' | ' if self.name else ''}{tuple(self.position)}, {self.angle * 180:.0f}°)"
