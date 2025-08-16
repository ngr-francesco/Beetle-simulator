from matplotlib.axes import Axes
from typing import List, Set
import numpy as np
from src.vector import Vec2, EPSILON
from dataclasses import dataclass, field

INFTY = 10


def clockwise_sort(vec: Vec2):
    angle = np.arctan(vec.y / (vec.x + EPSILON)) + np.pi / 2
    if vec.x < 0:
        angle += np.pi
    return angle


def ordered_random_vertices(
    n_points: int, max_size: float, center: Vec2 = Vec2(0, 0), visualize=False
):
    if n_points < 4:
        raise ValueError("PolygonRoom must have at least 4 vertices")
    quadrants = set([])
    while len(quadrants) < 4:
        quadrants = set([])
        pts = []
        for _ in range(n_points):
            pt = Vec2(*np.random.uniform(-max_size / 2, max_size / 2, 2))
            pts.append(pt)
            if pt.x > 0 and pt.y > 0:
                quadrants.add("ne")
            if pt.x < 0 and pt.y > 0:
                quadrants.add("nw")
            if pt.x < 0 and pt.y < 0:
                quadrants.add("sw")
            if pt.x > 0 and pt.y < 0:
                quadrants.add("se")
    if visualize:
        for pt in pts:
            Point.from_vec2(pt)

    pts = sorted(pts, key=clockwise_sort)
    # After everything, add the center offset
    pts = [pt + center for pt in pts]
    return pts


class Shape:
    can_collide = False
    shape_id = 0

    def __init__(self):
        Shape.shape_id += 1
        if ShapeContainer._instance is not None:
            ShapeContainer._instance.add(self)

    def draw(self, axes: Axes):
        raise NotImplementedError

    def compute_fig_points(self):
        raise NotImplementedError


@dataclass
class Polygon(Shape):
    """
    Class for generic polygons. 
    It offloads collision detection and drawing to its edges,
    so it doesn't need explicit implementations for either of these functionalities.
    In practice, it is just a container for edges (Segments).
    """
    # Not required for all polygons
    vertices: list[Vec2] = field(default_factory=list)
    edges: list["Segment"] = field(init=False)
    center: Vec2 = field(init=False)

    def __post_init__(self):
        if not len(self.vertices):
            raise ValueError("Cannot initialize polygon without vertices")
        self.edges = self.edges_from_vertices()
        self.center = self.center_from_vertices()

    def draw(self, axes: Axes):
        """This is handled by the individual edges"""
        return

    def edges_from_vertices(self):
        edges = []
        for i in range(len(self.vertices)):  # 4 edges
            iplus1 = i + 1 if i < len(self.vertices) - 1 else 0
            p1 = self.vertices[i]
            p2 = self.vertices[iplus1]
            edges.append(Segment.from_two_points(p1, p2))
        return edges

    def center_from_vertices(self):
        center = Vec2(0, 0)
        for vertex in self.vertices:
            center += vertex
        return center / len(self.vertices)

    def rotate(self, angle: float):
        new_vertices = []
        for vertex in self.vertices:
            c_to_v = vertex - self.center
            rotated = c_to_v.rotate(angle)
            displ = rotated - c_to_v
            new_vertices.append(vertex + displ)
        self.vertices = new_vertices


@dataclass(kw_only=True)
class RandomPolygon(Polygon):
    n_edges: int
    max_size: float
    center: Vec2 = field(default=Vec2(0, 0))

    def __post_init__(self):
        self.vertices = ordered_random_vertices(
            self.n_edges, self.max_size, self.center
        )
        return super().__post_init__()


@dataclass(kw_only=True)
class Point(Shape):
    can_collide = False
    x: float
    y: float

    def __post_init__(self):
        Shape.__init__(self)

    @staticmethod
    def from_vec2(vec: Vec2):
        return Point(x=vec.x, y=vec.y)

    def to_vec(self):
        return Vec2(self.x, self.y)

    def draw(self, axes: Axes):
        axes.plot([self.x], [self.y], marker="o", color="magenta")


@dataclass(kw_only=True)
class Rectangle(Polygon):
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self):
        Shape.__init__(self)
        if not len(self.vertices):
            self.vertices = [
                Vec2(self.x, self.y),
                Vec2(self.x + self.w, self.y),
                Vec2(self.x + self.w, self.y + self.h),
                Vec2(self.x, self.y + self.h),
            ]
        Polygon.__post_init__(self)

    def collision(self, x: int, y: int):
        return all([x > self.x, x < self.x + self.w, y > self.y, y < self.y + self.h])

    def rotate(self, angle: float):
        super().rotate(angle)
        self.x = self.vertices[0].x
        self.y = self.vertices[0].y


@dataclass
class HemiPlane(Shape):
    can_collide = True
    p0: Vec2
    direction: Vec2
    normal: Vec2 = field(init=False)
    fig_points: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Uses the equation ax + by + c = 0
        """
        Shape.__init__(self)
        self.normal = self.direction.rotate(np.pi / 2)
        self.fig_points = self.compute_fig_points()

    @staticmethod
    def from_two_points(pt_1: Vec2, pt_2: Vec2):
        direction = (pt_2 - pt_1).normalized()
        return HemiPlane(p0=pt_1, direction=direction)

    @staticmethod
    def from_characteristic_equation(a: float, b: float, c: float):
        if b != 0:
            p0 = Vec2(0, -c / b)
            direction = Vec2(1, -a / b).normalized()
        elif a != 0:
            p0 = Vec2(-c / a, 0)
            # In this case b=0, so we can just set the vertical vector
            direction = Vec2(0, 1).normalized()
        else:
            raise ValueError(
                f"Invalid parameters for HemiPlane definition a={a}, b={b}"
            )
        return HemiPlane(p0=p0, direction=direction)

    def compute_fig_points(self):
        """
        In this case the points are ordered [[x1,y1],[x2,y2]]
        Because we're using plt.axline which needs them in this format.
        """
        return np.array(
            [
                [self.p0.x, self.p0.y],
                [
                    INFTY * self.direction.x + self.p0.x,
                    INFTY * self.direction.y + self.p0.y,
                ],
            ]
        )

    def draw(self, axes: Axes):
        axes.axline(*self.fig_points)

    def collision(self, point: Vec2):
        # You check the dot product with the normal to the line.
        return (self.normal.dot(point - self.p0)) <= 0

    def rotate(self, angle: float):
        self.direction = self.direction.rotate(angle)
        self.normal = self.direction.rotate(np.pi / 2)
        self.fig_points = self.compute_fig_points()


@dataclass
class Segment(HemiPlane):
    length: float
    can_collide: bool = field(default=True)
    add_to_shape_container: bool = field(default=True)

    @staticmethod
    def from_two_points(pt_1: Vec2, pt_2: Vec2):
        direction = (pt_2 - pt_1).normalized()
        length = (pt_2 - pt_1).length()
        return Segment(p0=pt_1, direction=direction, length=length)

    def compute_fig_points(self):
        return np.array(
            [
                [self.p0.x, self.length * self.direction.x + self.p0.x],
                [self.p0.y, self.length * self.direction.y + self.p0.y],
            ]
        )

    def draw(self, axes: Axes):
        axes.plot(*self.fig_points, color="r", alpha=0.5)


class Ray:
    def __init__(self, angle: float, origin: Vec2 = Vec2(0, 0), length: int = 10000):
        self.angle = angle
        self.origin = origin
        self.length = length
        self.direction = Vec2(np.cos(angle), np.sin(angle))
        self._draw_point = 0
        self.collisions = {
            HemiPlane: self.collides_line,
            Polygon: self.collides_polygon,
            Segment: self.collides_segment,
        }

    def at(self, coordinate: float):
        return self.origin + self.direction * coordinate

    def to_line(self):
        return Segment(self.origin, self.direction, self.length, can_collide=False)

    def collides_line(self, hemiplane: HemiPlane):
        determinant = self.direction.dot(hemiplane.normal)
        if np.abs(determinant) < EPSILON:
            return None
        t = (hemiplane.p0 - self.origin).dot(hemiplane.normal) / determinant
        if t < EPSILON or t > self.length:
            return None
        return t

    def collides_segment(self, segment: Segment):
        t = self.collides_line(segment)
        if t is None:
            return None

        hit_point = self.at(t)
        proj = (hit_point - segment.p0).dot(segment.direction)
        if proj < 0 - EPSILON or proj > segment.length + EPSILON:
            return None
        return t

    def collides_polygon(self, polygon: Polygon):
        min_t = None
        for edge in polygon.edges:
            t = self.collides_segment(edge)
            if t is not None and (min_t is None or t < min_t):
                min_t = t
        return min_t

    def collide(self, shape: Shape):
        if not shape.can_collide:
            return None
        try:   
            print(type(shape), type(shape) in self.collisions)
            return self.collisions[type(shape)](shape)
        except KeyError:
            raise NotImplementedError(
                f"Ray collision with Shape of type {type(shape)} is not supported yet."
            )


class ShapeContainer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.shapes = {}
        return cls._instance

    def __init__(self):
        self.shapes: dict[int, Shape] = {}

    def draw(self, axes: Axes):
        for shape in self.shapes.values():
            shape.draw(axes)

    def add(self, shape: Shape):
        if Shape.shape_id in self.shapes:
            raise ValueError(
                "Shape already in container. Adding a shape multiple times is not allowed."
            )
        self.shapes[Shape.shape_id] = shape

    def __iter__(self):
        # Makes the container iterable
        return iter(self.shapes.values())

    def __len__(self):
        return len(self.shapes)

    def remove(self, shape_id):
        self.shapes.pop(shape_id)

    def clear(self):
        self.shapes = {}
