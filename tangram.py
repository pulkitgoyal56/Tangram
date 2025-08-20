"""Gymnasium environment for Tangram."""

import logging
from functools import partial

import numpy as np
from matplotlib.collections import PatchCollection

from PIL import Image
# from skimage.draw import polygon as sk_polygon
from cv2 import rectangle, fillPoly, fillConvexPoly

import gymnasium as gym

from polygon import Polygon, Triangle, Quadrilateral
from polygonlist import PolygonList
from grid import Grid
import constants

logger = logging.getLogger(__name__)


class Tangram(PolygonList, gym.Env):
    OBJECTS = lambda scale: [
        Triangle(name="triangle-large-1", scale=(scale / 2**0.5,), color=[255, 0, 0]),                                              # 2 large right triangles  # (1.00, 0.00, 0.00, 1.)
        Triangle(name="triangle-large-2", scale=(scale / 2**0.5,), color=[255, 127, 66]),                                           # "                        # (1.00, 0.50, 0.26, 1.)
        Triangle(name="triangle-medium-1", scale=(scale / 2,), color=[211, 221, 127]),                                              # 1 medium right triangle  # (0.83, 0.87, 0.50, 1.)
        Triangle(name="triangle-small-1", scale=(scale / 8**0.5,), color=[127, 255, 178]),                                          # 2 small right triangles  # (0.50, 1.00, 0.70, 1.)
        Triangle(name="triangle-small-2", scale=(scale / 8**0.5,), color=[43, 221, 221]),                                           # "                        # (0.17, 0.87, 0.87, 1.)
        Quadrilateral(name="square-small-1", scale=(scale / 8**0.5,), color=[43, 127, 247]),                                        # 1 small square           # (0.17, 0.50, 0.97, 1.)
        Quadrilateral(name="parallelogram-small-1", scale=(scale / 2, scale / 8**0.5), shear=(0, 0.25), color=[127, 0, 255]),       # 1 small parallelogram    # (0.50, 0.00, 1.00, 1.)
        # [(0, 0), (0.25, 0.25), (0.75, 0.25), (0.5, 0)]
    ]

    """Gymnasium environment for Tangram (2D)."""
    def __init__(
        self,
        *,
        scale=constants.DEFAULT_SCALE,   # scale of objects
        x_size=28,                       # number of steps in x
        y_size=None,                     # number of steps in y
        r_size=4,                        # number of steps in pi
        rotate=True,                     # whether or not to allow rotation of objects
        flip=False,                      # whether or not to allow flipping objects
        x_step=1,                        # maximum steps in x translation actions
        y_step=None,                     # maximum steps in y translation actions
        r_step=1,                        # maximum steps in xy rotation actions
        object_persistency=3,            # setting this to zero is the same as controlling all objects together
        max_dist=-1,                     # maximum allowed number of steps from initial position for resetting  # setting this to negative is the same as having no constraints in randomization of initial positions on reset
        control="all",                   # "limited" or "all"
        control_boundaries=None,         # control area ((lower_x, lower_y), (upper_x, upper_y))
        control_criteria="center",       # None, "complete", "partial", or polygon attribute (e.g. "center", "centroid", "pivot", "position"), "random", float
        staging_boundaries=None,         # staging area ((lower_x, lower_y), (upper_x, upper_y))
        staging_criteria="center",       # None, "complete", "partial", or polygon attribute (e.g. "center", "centroid", "pivot", "position")
        seed=None,
        **kwargs,
    ):
        self._scale = scale
        PolygonList.__init__(self, Tangram.OBJECTS(self._scale))

        self.name = "Tangram"

        self.x_size = x_size
        self.y_size = y_size if y_size is not None else self.x_size
        self.r_size = r_size

        self.x_step = x_step if x_step >= 1 else self.x_size * x_step
        self.y_step = (y_step if y_step >= 1 else self.y_size * y_step) if y_step is not None else self.x_step
        self.r_step = r_step if r_step >= 1 else self.r_size * r_step

        self.np_random = None
        self.game = None
        self.target = None
        self.collisions = True

        self.max_dist = max_dist
        # self.num_objects = num_objects
        self.object_persistency = object_persistency

        self.control_boundaries = Grid.normalize(control_boundaries, (self.x_size, self.y_size)) if control_boundaries is not None else np.asarray(((0, 0), (1., 1.)))
        self.control_criteria = control_criteria
        self.control = control

        self.staging_boundaries = Grid.normalize(staging_boundaries, (self.x_size, self.y_size)) if staging_boundaries is not None else np.asarray(((0, 0), (1., 1.)))
        self.staging_criteria = staging_criteria

        self.colors = list(map(lambda polygon: polygon.color, self.polygons))

        self.seed(seed)

        self.num_actions = self.dof = 2 + bool(rotate) + bool(flip)

        # Initialize to pos outside of env for easier collision resolution.
        # self.objects = [[-1, -1, -1, -1]] * len(self.polygons)
        if not self.object_persistency and self.control == "all":
            self.num_actions = self.dof * self.num_objects

        # if "set_params" in kwargs:
        #     logger.info("> Setting Tangram")
        #     for param, value in kwargs["set_params"].items():
        #         logger.debug(f" > {param=}")
        #         setattr(self, param, value)

        self.mask()
        self.checkpoint()
        self.reorder()
        self.reset(constrained=False)

    @property
    def objects(self):
        objects = np.asarray([(*polygon.position, polygon._angle, polygon._flipped) for polygon in self.polygons], dtype=np.float32)
        objects.flags.writeable = False
        return objects

    @objects.setter
    def objects(self, objects):
        if len(objects) != self.num_objects:
            raise ValueError(f"> Only {len(self.polygons)} polygons are setup; expected {self.num_objects} positions, got {len(objects)}.")
        # else:
        #     logger.debug("> Resetting polygon positions.")
        for polygon, (x, y, xy, ax) in zip(self.polygons, objects):
            polygon.position = (x, y)
            polygon.angle = xy
            polygon.flipped = bool(ax)
        if not self.object_persistency and self.control == "all":
            self.num_actions = self.dof * self.num_objects
        if not hasattr(self, "colors") or len(self.colors) != self.num_objects:
            self.colors += [constants.DEFAULT_COLOR] * (self.num_objects - len(self.colors))

    def object(self, obj_id: int, *, scale=False, dtype=np.float32):
        object = np.asarray([*self.polygons[obj_id].position, self.polygons[obj_id]._angle, self.polygons[obj_id]._flipped])
        if scale:
            object = object * [self.x_size, self.y_size, self.r_size, 1]
        return object.astype(dtype)

    @property
    def num_objects(self):
        return len(self.polygons)

    @property
    def observation_space(self):
        # TODO `observation_space` is a mutable object. It should have a setter and be a member.
        # return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_objects * self.num_actions + (2 if self.object_persistency else 0),), dtype=np.float32)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n * self.dim + (2 if self.object_persistency else 0),), dtype=np.float32)

    @property
    def num_actions(self):
        return self._num_actions

    @num_actions.setter
    def num_actions(self, num_actions):
        if not hasattr(self, "num_actions") or self.num_actions != num_actions:
            self._num_actions = num_actions
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_actions,), dtype=np.float32)
            # self.action_space.np_random = self.np_random

    @property
    def state_obs(self):
        # if self.object_persistency:
        #     return np.hstack((np.asarray(self.objects, dtype=np.float32).flatten(), self.current_object, self.current_object_t))
        # else:
        #     return np.asarray(self.objects, dtype=np.float32).flatten()

        if self.object_persistency:
            return np.hstack((np.asarray(self.vertices, dtype=np.float32).flatten(), self.current_object, self.current_object_t))
        else:
            return np.asarray(self.vertices, dtype=np.float32).flatten()

    def objects_from_state(self, state_obs):
        return state_obs[:-2 if self.object_persistency else None].reshape(-1, self.dim)  # .tolist()

    def attr_from_state(self, state_obs, *, attr="position", i=None):
        vertices = self.objects_from_state(state_obs)
        assert len(vertices) == self.n, f"> Expected {self.n} vertices, got {len(vertices)}."
        vertices = np.split(vertices, np.cumsum(self.num_vertices)[:-1])
        assert len(vertices) == len(self), f"> Expected {len(self)} set of vertices, got {len(vertices)}."
        if i is not None:
            return getattr(Polygon(vertices[i]), attr)
        else:
            return PolygonList(vertices).get(attr)

    def __repr__(self):
        return f"{self.name}({self.num_objects} | [{self.x_size}{'.' if self.x_size == 1 else ''} x {self.y_size}{'.' if self.y_size == 1 else ''} x {self.r_size}{'.' if self.r_size == 1 else ''}] @ [{self.x_step} x {self.y_step} x {self.r_step}])"

    def __contains__(self, obj):
        if isinstance(obj, Polygon):
            return obj.within_bounds(self.staging_boundaries, criteria=self.staging_criteria)
        else:
            return any(obj in polygon for polygon in self.polygons if polygon in self)

    def mask(self, mask=None):
        self._mask = np.asarray(self.mask_from_boundaries(self.control_boundaries, criteria=self.control_criteria) if mask is None else mask, dtype=bool)
        if not self.object_persistency and self.control == "limited":
            self.num_actions = self.dof * sum(self._mask)
        return self

    def mask_from_boundaries(self, boundaries=None, *, criteria=None):
        if criteria == "random":
            return np.random.choice([0, 1], self.num_objects, p=[0.5, 0.5])

        if isinstance(criteria, float):
            return np.random.choice([0, 1], self.num_objects, p=[1 - boundaries, boundaries])

        def polygon_within_active_bounds(polygon: Polygon):
            return np.any(polygon.position < 0) or polygon.within_bounds(boundaries, criteria=criteria)

        return list(map(polygon_within_active_bounds, self.polygons))

    def reorder(self):
        # t = False  # if any locked objects were found
        # for m in self._mask:
        #     if not m:
        #         t = True
        #     elif t:
        #         logger.debug(">> Readjusting object order ...")
        #         break

        # if m and t:
        #     if self.object_persistency:
        #         assert self.current_object == 0, f"Expected current_object to be 0 for reordering, got {self.current_object}."
        #         assert self.current_object_t == 0, f"Expected current_object_t to be 0 for reordering, got {self.current_object_t}."
        #     # current_object = self.objects[self.current_object]

        #     _reorder = lambda l: np.r_[np.asarray(l)[self._mask], np.asarray(l)[~self._mask]]

        #     self._polygons = [self.polygons[i] for i in range(len(self)) if self._mask[i]] + [self.polygons[i] for i in range(len(self)) if not self._mask[i]]
        #     # self.objects = _reorder(self.objects)
        #     self.objects_ = _reorder(self.objects_)
        #     TODO: replace objects_ with vertices_
        #     # self.colors = _reorder(self.colors).tolist()
        #     self._mask = _reorder(self._mask)
        #     # self.current_object = self.objects.index(current_object)
        return self

    def seed(self, seed=None):
        self.np_random, seed = np.random.default_rng(seed), seed
        # self.action_space.np_random = self.np_random
        self._seed = seed
        return [seed]

    def patches(self, *, sift=True, color=True, **kwargs):
        sift = [not sift or polygon in self for polygon in self.polygons]
        patches = PatchCollection([self.polygons[i].patch(**kwargs) for i in range(len(self)) if sift[i]], **kwargs)
        patches.set_facecolors((0, 0, 0) if not color else [[c / 255 for c in self.colors[i]] for i in range(len(self)) if sift[i]])
        return patches

    def render_fast(self, size=(224, 224), *, sift=True, color=True, lineType=8, **kwargs):
        if color:
            canvas = np.full((*size[::-1], 3), 255, dtype=np.uint8)
            for polygon in self.polygons:
                if not sift or polygon in self:
                    # cc, rr = sk_polygon(*(polygon.vertices * size).T, (*size, 3))  # **kwargs
                    # canvas[size[1] - 1 - rr, cc] = polygon.color
                    fillConvexPoly(canvas, [0, size[1] - 1] + (polygon.vertices * [size[0], -size[1]]).astype(int), color=polygon.color, lineType=lineType)  # **kwargs
        else:
            canvas = np.full(size[::-1], 255, dtype=np.uint8)
            fillPoly(
                canvas,
                [[0, size[1] - 1] + (polygon.vertices * [size[0], -size[1]]).astype(int) for polygon in self.polygons if not sift or polygon in self],
                color=0,
                lineType=lineType,
            )  # **kwargs
        return canvas

    def render_image(self, *, format="image", highlight_active=False, invert=False, crop=True, color=True, **kwargs):
        image = self.render(xlim=self.staging_boundaries.T[0] if crop else [0, 1], ylim=self.staging_boundaries.T[1] if crop else [0, 1], sift=crop, color=color, **kwargs)
        if invert:
            image = 255 - image
        if highlight_active:
            image = self.highlight_active(image, v=[int(invert) * 255] * (3 if color else 1), boundaries=not crop, vertices=False)
        if format == "image":
            return Image.fromarray(image)
        return image

    image = property(partial(render_image, crop=False))

    def _ipython_display_(self):
        from IPython.display import display
        display(self.image)

    def highlight_active(self, image: np.ndarray, v: list, *, boundaries: bool = False, vertices: bool = False, size: int = 5):
        from imagegrid import Im

        if boundaries:
            rectangle(image, *Grid.xy2rc(self.staging_boundaries, shape=image.shape[:2])[:, ::-1], tuple(v), 1)
            # ImageDraw.Draw(image).rectangle((self.staging_boundaries * image.shape[::-1]).astype(int).ravel().tolist(), outline=tuple(v))

        def highlight(image: np.ndarray, xy: np.ndarray, *, scale: float = 1.):
            return Im.highlight(
                image,
                *Grid.xy2rc(
                    xy - (not boundaries) * self.staging_boundaries[0],
                    shape=image.shape[:2],
                    size=(1., 1.) if boundaries else np.diff(self.staging_boundaries, axis=0)[0],
                ),
                size=1,
                v=v if vertices or image.ndim == 3 else [255 - c for c in v],
                radius=scale * self._scale * size / (1 if boundaries else np.sqrt(Grid(*self.staging_boundaries).area)),
            )

        if self.object_persistency:
            if boundaries or self.polygons[self.current_object] in self:
                if vertices:
                    for vertex in self.polygons[self.current_object].vertices:
                        if boundaries or vertex in Grid(*self.staging_boundaries):
                            image = highlight(image, vertex)
                else:
                    image = highlight(image, self.polygons[self.current_object].pivot)
        elif self.control == "limited":
            for polygon, mask in zip(self.polygons, self._mask):
                if mask and (boundaries or polygon in self):
                    if vertices:
                        for vertex in polygon.vertices:
                            if boundaries or vertex in Grid(*self.staging_boundaries):
                                image = highlight(image, vertex)
                    else:
                        image = highlight(image, polygon.pivot)
        return image

    def checkpoint(self, objects=None):
        # assert objects is None or np.asarray(objects).shape == (self.num_objects, 4), f"Expected objects of shape {self.num_objects, 4}, got {np.asarray(objects).shape}."
        self.objects_ = np.asarray([o[:] for o in objects], dtype=self.objects.dtype) if objects else self.objects.copy()  # Original positions for resetting
        return self

    def restore(self):
        self.positions = self.objects_[:, :self.dim]
        return self

    def reset(self, *, constrained=True):
        # self.objects = [[-1, -1]] * (self.num_objects if self.num_objects else 0)

        if self.objects_ is None:
            logger.warning("> No checkpoint found! | Resetting objects in boundaries without constraints and leaving objects outside boundaries unchanged.")
            self.objects_ = np.where(np.tile(self._mask, (self.objects.shape[1], 1)).T, [[-1] * self.objects.shape[1]] * self.num_objects, self.objects)
        assert self.num_objects == len(self.objects_), f"Checkpoint has {len(self.objects_)} objects, but state has {self.num_objects}."

        # self.objects = np.c_[np.where(np.tile(self._mask, (self.dim, 1)).T, [[-1] * self.dim] * self.num_objects, self.objects_[:, :self.dim]), self.objects_[:, self.dim:]]
        self.positions = np.where(np.tile(self._mask, (self.dim, 1)).T, [[-1] * self.dim] * self.num_objects, self.objects_[:, :self.dim])
        # for i in range(self.num_objects):
        #     if self._mask[i]:
        #         self.objects[i] = [-1, -1, 0, 0]
        #     else:  # if self.control == "all"  # Only required when all objects are allowed to move
        #         self.objects[i] = self.objects_[i]

        # Randomize object positions
        for i in range(self.num_objects):
            # Resample to ensure objects don't fall on same spot.
            if self._mask[i]:
                while not self.valid_pos(self.polygons[i].vertices, i, constrained=constrained):
                    self.polygons[i].position = [self.np_random.choice(size) / size if size > 1 else self.np_random.uniform() for size in [self.x_size, self.y_size]]

        if self.object_persistency:
            self.current_object = 0
            self.current_object_t = 0

        assert self._validate(), "Invalid state after reset!"
        return self.state_obs

    def valid_pos(self, vertices, obj_id, *, constrained=False):
        """Check if position is valid."""
        # for v in vertices:
        #     if not (0 <= v[0] < 1 and 0 <= v[1] < 1):  # not in self
        #         return False
        if np.any((0 > vertices) | (vertices >= 1)):
            return False

        if self.collisions and self.collides(vertices, obj_id):
            return False

        if constrained and self.max_dist >= 0:  # self._mask[obj_id]  # Only required for unlocked objects
            if np.any(
                (self.objects_[obj_id][:self.dim] >= 0)
                & (
                    np.abs(Polygon.get_pivot(vertices) - self.objects_[obj_id][:self.dim])
                    > (self.max_dist if self.max_dist < 1 else [self.max_dist / size if size > 1 else size / self.max_dist for size in [self.x_size, self.y_size]])
                )
            ):
                return False

        return True

    def valid_move(self, obj_id: int, action, *, constrained=False):
        """Check if move is valid."""
        if self.control == "limited" and not self._mask[obj_id] and action != [0, 0, 0, 0]:
            logger.error(f"> Tried moving locked polygon, id={obj_id} ({self.polygons[obj_id]})!")
            # return False
        vertices = self.polygons[obj_id].transform(*action, inplace=False)
        return self.valid_pos(vertices, obj_id, constrained=constrained), vertices

    def transform(self, obj_id: int, dx, dy, xy=0, ax=False):
        """Translate polygon with given ID."""
        valid, vertices = self.valid_move(obj_id, (dx, dy, xy, ax))
        if valid:
            self.polygons[obj_id]._vertices = vertices
            self.polygons[obj_id]._position += (dx, dy)
            if xy:
                self.polygons[obj_id]._angle += Polygon.get_angle(xy)
            if ax:
                self.polygons[obj_id]._flipped = not self.polygons[obj_id].flipped
        # else:
        #     logger.error(f"> Invalid move for polygon {obj_id} @ [{self.current_object, self.current_object_t}]!")

    def discretize_action(self, dx, dy, xy=0, ax=None):
        def discretize(action, step, size=1):
            if size == 1:
                return action / step if step >= 1 else action * step
            return ((action + 1) * (step + 0.5) // 1 - step // 1) / (size if size > 1 else step)
            # return (-1 if action < -1 / 3 else 1 if action > 1 / 3 else 0) / size

        return (discretize(dx, self.x_step, self.x_size), discretize(dy, self.y_step, self.y_size), discretize(xy, self.r_step, self.r_size), bool(ax))

    def step(self, action):
        # directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        if self.object_persistency:
            # logger.debug(f"Stepping for object {obj} in direction {direction}")
            self.transform(self.current_object, *self.discretize_action(*action))

            self.current_object_t += 1

            if self.current_object_t >= self.object_persistency:
                self.current_object = (self.current_object + 1) % self.num_objects
                if self.control == "limited":
                    while not self._mask[self.current_object]:
                        self.current_object = (self.current_object + 1) % self.num_objects
                self.current_object_t = 0
        else:
            assert len(action) == self.num_actions, f"Expected {self.num_actions} actions, got {len(action)}."
            for obj, direction in enumerate(map(self.discretize_action, *np.asarray(action).reshape(-1, self.dof).T)):
                self.transform(obj, *direction)

        return self.state_obs, 0, False, None  # done, reward, None

    def deform(self):
        action = self.action_space.sample()
        return self.step(action), action

    def regularity(self, **kwargs):
        from imagegrid import Im
        return Im.regularity(self.vertices, **kwargs)
        # return Im.regularity(self.objects[:, :2], **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from tangram import Tangram
    from constants import HOUSE

    house = Tangram(x_size=1)
    house.vertices = HOUSE()
    
    house.render_image()
