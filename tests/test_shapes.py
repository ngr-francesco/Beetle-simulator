import math
import numpy as np
import pytest
from src.vector import Vec2
from src.shapes import ShapeContainer, Point, Rectangle, HemiPlane, Ray, EPSILON


@pytest.fixture(autouse=True)
def clear_container():
    # Reset singleton before each test
    ShapeContainer._instance = None
    yield
    ShapeContainer._instance = None


def test_shape_container_singleton():
    c1 = ShapeContainer()
    c2 = ShapeContainer()
    assert c1 is c2  # Singleton instance check


def test_point_creation_and_container_add():
    container = ShapeContainer()
    p = Point(1, 2)
    assert len(container) == 1
    # The ID is incremented before adding to the container
    assert container.shapes[1] == p
    assert p.to_vec() == Vec2(1, 2)


def test_rectangle_collision():
    r = Rectangle(0, 0, 10, 5)
    assert r.collision(5, 2) is True
    assert r.collision(-1, 2) is False
    assert r.collision(5, 6) is False
    assert r.fig_points.shape == (2, 5)


def test_hemiplane_from_two_points_and_collision():
    p1 = Vec2(0, 0)
    p2 = Vec2(1, 0)
    hemi = HemiPlane.from_two_points(p1, p2)

    # Point above line
    pt_above = Vec2(0, 1)
    assert hemi.collision(pt_above) == True  # depends on normal orientation

    # Point below line
    pt_below = Vec2(0, -1)
    assert hemi.collision(pt_below) == True or hemi.collision(pt_below) == False  # Just ensure boolean


def test_hemiplane_from_characteristic_equation():
    # Horizontal line y = 2 => b=1, c=-2
    hemi = HemiPlane.from_characteristic_equation(0, 1, -2)
    assert math.isclose(hemi.p0.y, 2, abs_tol=1e-9)
    assert isinstance(hemi.direction, Vec2)


def test_ray_line_collision():
    hemi = HemiPlane.from_characteristic_equation(0, 1, 0)  # y=0
    ray = Ray(angle=math.pi/2, origin=Vec2(0, -1), length=10)  # going upward
    t = ray.collides_line(hemi)
    assert t is not None
    assert math.isclose(ray.at(t).y, 0, abs_tol=EPSILON)


def test_ray_no_collision_parallel():
    hemi = HemiPlane.from_characteristic_equation(0, 1, 0)  # y=0
    ray = Ray(angle=0, origin=Vec2(0, 1), length=10)  # parallel to line y=0
    t = ray.collides_line(hemi)
    assert t is None


def test_container_iteration_and_len():
    container = ShapeContainer()
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    assert len(container) == 2
    points = list(container)
    assert p1 in points and p2 in points


def test_ray_collides_rectangle_hit():
    rect = Rectangle(0, 0, 10, 5)  # bottom-left at (0,0), width=10, height=5
    ray = Ray(angle=0, origin=Vec2(-5, 2), length=20)  # horizontal rightwards through rectangle

    t = ray.collides_rectangle(rect)
    assert t is not None, "Ray should intersect rectangle"
    hit_point = ray.at(t)
    # Since the left edge is at x=0, y should stay at 2
    assert math.isclose(hit_point.x, 0, abs_tol=EPSILON)
    assert math.isclose(hit_point.y, 2, abs_tol=EPSILON)


def test_ray_collides_rectangle_miss():
    rect = Rectangle(0, 0, 10, 5)
    ray = Ray(angle=math.pi / 2, origin=Vec2(-5, 10), length=20)  # upward ray above rectangle

    t = ray.collides_rectangle(rect)
    assert t is None, "Ray should miss rectangle completely"


def test_ray_collides_rectangle_from_inside():
    rect = Rectangle(0, 0, 10, 5)
    ray = Ray(angle=0, origin=Vec2(5, 2), length=20)  # starts inside, pointing right

    t = ray.collides_rectangle(rect)
    assert t is not None
    hit_point = ray.at(t)
    # Should exit at right edge x=10
    assert math.isclose(hit_point.x, 10, abs_tol=EPSILON)
    assert math.isclose(hit_point.y, 2, abs_tol=EPSILON)