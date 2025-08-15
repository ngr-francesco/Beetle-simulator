from matplotlib.axes import Axes
from typing import List, Set
import numpy as np
from src.vector import Vec2, EPSILON
from dataclasses import dataclass, field

INFTY = 10


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
class Point(Shape):
    can_collide = False
    x: float
    y: float
    def __post_init__(self):
        Shape.__init__(self)
    
    @staticmethod
    def from_vec2(vec: Vec2):
        return Point(vec.x, vec.y)
    
    def to_vec(self):
        return Vec2(self.x, self.y)

    def draw(self, axes: Axes):
        axes.plot([self.x], [self.y], marker='o', color='magenta')


@dataclass
class Rectangle(Shape):
    can_collide = True
    x: float
    y: float
    w: float
    h: float
    vertices: list[Vec2] = field(init=False)
    center: Vec2 = field(init=False)
    fig_points: np.ndarray = field(init=False)

    def __post_init__(self):
        Shape.__init__(self)
        self.center = Vec2(self.x+self.w/2, self.y+self.h/2)
        self.vertices = [Vec2(self.x, self.y),
                         Vec2(self.x+self.w, self.y),
                         Vec2(self.x+self.w, self.y+self.h),
                         Vec2(self.x, self.y + self.h)]
        self.fig_points = self.compute_fig_points()

    def collision(self, x: int, y: int):
        return all([x > self.x, x < self.x + self.w, y > self.y, y < self.y + self.h])

    def compute_fig_points(self):
        pts = np.zeros((2, 5))
        # Ordering is important!
        pts[0] = [v.x for v in self.vertices] + [self.vertices[0].x]
        pts[1] = [v.y for v in self.vertices] + [self.vertices[0].y]
        return pts

    def draw(self, axes: Axes):
        axes.plot(*self.fig_points)

    def rotate(self, angle: float):
        new_vertices = []
        for vertex in self.vertices:
            c_to_v = vertex - self.center
            rotated = c_to_v.rotate(angle)
            displ = rotated - c_to_v
            new_vertices.append(vertex+displ)
        self.vertices = new_vertices
        self.x = self.vertices[0].x
        self.y = self.vertices[0].y
        self.fig_points= self.compute_fig_points()



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
        self.normal = self.direction.rotate(np.pi/2)
        self.fig_points = self.compute_fig_points()
    
    @staticmethod
    def from_two_points(pt_1: Vec2, pt_2: Vec2):
        direction = (pt_2 - pt_1).normalized()
        return HemiPlane(p0=pt_1, direction=direction)

    @staticmethod
    def from_characteristic_equation(a: float, b: float, c: float):

        if b != 0:
            p0 = Vec2(0, - c/b)
            direction = Vec2(1, -a/b).normalized()
        elif a != 0:
            p0 = Vec2(-c/a, 0)
            # In this case b=0, so we can just set the vertical vector
            direction= Vec2(0, 1).normalized()
        else:
            raise ValueError(f"Invalid parameters for HemiPlane definition a={a}, b={b}")
        return HemiPlane(p0=p0, direction=direction)

    def compute_fig_points(self):
        """
        In this case the points are ordered [[x1,y1],[x2,y2]]
        Because we're using plt.axline which needs them in this format.
        """
        return np.array(
            [[self.p0.x, self.p0.y], [INFTY*self.direction.x + self.p0.x, INFTY*self.direction.y + self.p0.y]]
        )

    def draw(self, axes: Axes):
        axes.axline(*self.fig_points)

    def collision(self, point: Vec2):
        # You check the dot product with the normal to the line.
        return (self.normal.dot(point-self.p0)) <= 0
    
    def rotate(self, angle: float):
        self.direction = self.direction.rotate(angle)
        self.normal = self.direction.rotate(np.pi/2)
        self.fig_points = self.compute_fig_points()


@dataclass
class Segment(HemiPlane):
    can_collide = False
    length: float

    def compute_fig_points(self):
        return np.array(
            [[self.p0.x, self.length*self.direction.x + self.p0.x], [self.p0.y, self.length*self.direction.y + self.p0.y]]
        )
    
    def draw(self, axes: Axes):
        axes.plot(*self.fig_points, color='r', alpha=0.5)


class Ray:
    def __init__(
        self, angle: float, origin: Vec2 = Vec2(0, 0), length: int = 10000
    ):
        self.angle = angle
        self.origin = origin
        self.length = length
        self.direction = Vec2(np.cos(angle), np.sin(angle))
        self._draw_point = 0
        self.collisions = {
            HemiPlane: self.collides_line,
            Rectangle: self.collides_rectangle,
        }

    def at(self, coordinate: float):
        return self.origin + self.direction * coordinate

    def to_line(self):
        return Segment(self.origin, self.direction, self.length)

    def collides_line(self, hemiplane: HemiPlane):
        determinant = self.direction.dot(hemiplane.normal)
        if np.abs(determinant) < EPSILON:
            return None
        t = (hemiplane.p0 - self.origin).dot(hemiplane.normal)/ determinant
        if t < EPSILON or t > self.length:
            return None
        return t

    def collides_rectangle(self, rectangle: Rectangle):
        min_t = None
        for i in range(4):  # 4 edges
            iplus1= i+1 if i < len(rectangle.vertices)-1 else 0
            p1 = rectangle.vertices[i]
            p2 = rectangle.vertices[iplus1]

            edge_dir = p2 - p1
            edge_normal = edge_dir.rotate(np.pi/2)

            denom = self.direction.dot(edge_normal)
            if abs(denom) < EPSILON:
                continue  # Parallel → no intersection

            t = (p1 - self.origin).dot(edge_normal) / denom
            if t < EPSILON or t > self.length:
                continue  # Behind ray or beyond length

            # Check if intersection point is within segment
            hit_point = self.at(t)
            proj = (hit_point - p1).dot(edge_dir.normalized())
            if 0 - EPSILON <= proj <= edge_dir.length() + EPSILON:
                if min_t is None or t < min_t:
                    min_t = t

        return min_t
    
    def collide(self, shape: Shape):
        if not shape.can_collide: 
            return None
        try:
            return self.collisions[type(shape)](shape)
        except KeyError:
            raise NotImplementedError(f"Ray collision with Shape of type {type(shape)} is not supported yet.")

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
            raise ValueError("Shape already in container. Adding a shape multiple times is not allowed.")
        self.shapes[Shape.shape_id]= shape
    
    def __iter__(self):
        # Makes the container iterable
        return iter(self.shapes.values())

    def __len__(self):
        return len(self.shapes)

    def remove(self, shape_id):
        self.shapes.pop(shape_id)
    
    def clear(self):
        self.shapes = {}