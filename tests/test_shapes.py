import math
import unittest
import numpy as np
from src.vector import Vec2
from src.shapes import *


def square_vertices(center=(0, 0), size=2):
    """Helper: return 4 vertices of a square centered at `center`."""
    cx, cy = center
    s = size / 2
    return [
        Vec2(cx - s, cy - s),
        Vec2(cx + s, cy - s),
        Vec2(cx + s, cy + s),
        Vec2(cx - s, cy + s),
    ]


class TestShapes(unittest.TestCase):
    def setUp(self):
        # Reset singleton before each test
        ShapeContainer._instance = None
        ShapeContainer()

    def test_shape_container_singleton(self):
        c1 = ShapeContainer()
        c2 = ShapeContainer()
        self.assertIs(c1, c2)

    def test_point_creation_and_container_add(self):
        container = ShapeContainer()
        p = Point(x=1, y=2)
        print(container.shapes)
        self.assertEqual(len(container), 1)
        self.assertEqual(list(container.shapes.values())[0], p)
        self.assertEqual(p.to_vec(), Vec2(1, 2))

    def test_rectangle_collision(self):
        r = Rectangle(x=0, y=0, w=10, h=5)

        # Inside point → should collide
        self.assertTrue(r.collision(Vec2(5, 2), Vec2(1, 0)))
        # Outside left
        self.assertFalse(r.collision(Vec2(-1, 2), Vec2(1, 0))[0])
        # Outside top
        self.assertFalse(r.collision(Vec2(5, 6), Vec2(0, -1))[0])

    def test_hemiplane_from_two_points_and_collision(self):
        p1 = Vec2(0, 0)
        p2 = Vec2(1, 0)
        hemi = HemiPlane.from_two_points(p1, p2)

        # Point above line
        pt_above = Vec2(0, 1)
        result_above = hemi.collision(pt_above, Vec2(0, -1))  
        self.assertIsInstance(result_above[0], bool)

        # Point below line
        pt_below = Vec2(0, -1)
        result_below = hemi.collision(pt_below, Vec2(0, 1))  
        self.assertIsInstance(result_below[0], bool)

    def test_hemiplane_from_characteristic_equation(self):
        hemi = HemiPlane.from_characteristic_equation(0, 1, -2)
        self.assertTrue(math.isclose(hemi.p0.y, 2, abs_tol=1e-9))
        self.assertIsInstance(hemi.direction, Vec2)

    def test_ray_line_collision(self):
        hemi = HemiPlane.from_characteristic_equation(0, 1, 0)
        ray = Ray(angle=math.pi / 2, origin=Vec2(0, -1), length=10)
        t = ray.collides_line(hemi)
        assert t is not None
        self.assertTrue(math.isclose(ray.at(t).y, 0, abs_tol=EPSILON))

    def test_ray_no_collision_parallel(self):
        hemi = HemiPlane.from_characteristic_equation(0, 1, 0)
        ray = Ray(angle=0, origin=Vec2(0, 1), length=10)
        t = ray.collides_line(hemi)
        self.assertIsNone(t)

    def test_container_iteration_and_len(self):
        container = ShapeContainer()
        p1 = Point(x=0, y=0)
        p2 = Point(x=1, y=1)
        self.assertEqual(len(container), 2)
        points = list(container)
        self.assertIn(p1, points)
        self.assertIn(p2, points)

    def test_ray_collides_rectangle_hit(self):
        rect = Rectangle(x=0, y=0, w=10, h=5)
        ray = Ray(angle=0, origin=Vec2(-5, 2), length=20)

        t = ray.collides_polygon(rect)
        assert t is not None
        hit_point = ray.at(t)
        self.assertTrue(math.isclose(hit_point.x, 0, abs_tol=EPSILON))
        self.assertTrue(math.isclose(hit_point.y, 2, abs_tol=EPSILON))

    def test_ray_collides_rectangle_miss(self):
        rect = Rectangle(x=0, y=0, w=10, h=5)
        ray = Ray(angle=math.pi / 2, origin=Vec2(-5, 10), length=20)
        t = ray.collides_polygon(rect)
        self.assertIsNone(t)

    def test_ray_collides_rectangle_from_inside(self):
        rect = Rectangle(x=0, y=0, w=10, h=5)
        ray = Ray(angle=0, origin=Vec2(5, 2), length=20)
        t = ray.collides_polygon(rect)
        assert t is not None
        hit_point = ray.at(t)
        self.assertTrue(math.isclose(hit_point.x, 10, abs_tol=EPSILON))
        self.assertTrue(math.isclose(hit_point.y, 2, abs_tol=EPSILON))

    def test_polygon_edges_and_center(self):
        vertices = square_vertices()
        poly = Polygon(vertices=vertices)
        self.assertEqual(len(poly.edges), 4)
        self.assertTrue(np.allclose([poly.center.x, poly.center.y], [0, 0]))

    def test_polygon_rotation_preserves_center(self):
        vertices = square_vertices(size=2)
        poly = Polygon(vertices=vertices)
        before_center = poly.center
        poly.rotate(np.pi / 4)
        after_center = poly.center_from_vertices()
        self.assertTrue(
            np.allclose([before_center.x, before_center.y], [after_center.x, after_center.y])
        )

    def test_random_polygon_has_correct_number_of_vertices(self):
        rp = RandomPolygon(n_edges=6, max_size=10)
        self.assertEqual(len(rp.vertices), 6)
        self.assertEqual(len(rp.edges), 6)

    def test_random_polygon_vertices_cover_quadrants(self):
        rp = RandomPolygon(n_edges=8, max_size=20)
        xs = [v.x for v in rp.vertices]
        ys = [v.y for v in rp.vertices]
        self.assertTrue(any(x > 0 for x in xs))
        self.assertTrue(any(x < 0 for x in xs))
        self.assertTrue(any(y > 0 for y in ys))
        self.assertTrue(any(y < 0 for y in ys))

    def test_ray_hits_square_polygon(self):
        poly = Polygon(vertices=square_vertices(size=2))
        ray = Ray(angle=0, origin=Vec2(-5, 0))
        t = ray.collides_polygon(poly)
        assert t is not None
        hit_point = ray.at(t) 
        self.assertTrue(np.allclose([hit_point.x, hit_point.y], [-1, 0], atol=1e-6))

    def test_ray_misses_polygon(self):
        poly = Polygon(vertices=square_vertices(size=2))
        ray = Ray(angle=np.pi / 2, origin=Vec2(-5, 0))
        t = ray.collides_polygon(poly)
        self.assertIsNone(t)

    def test_ray_hits_rectangle_same_as_polygon(self):
        rect = Rectangle(x=-1, y=-1, w=2, h=2)
        ray = Ray(angle=0, origin=Vec2(-5, 0))
        t = ray.collides_polygon(rect)
        assert t is not None
        hit_point = ray.at(t)
        self.assertTrue(np.allclose([hit_point.x, hit_point.y], [-1, 0], atol=1e-6))

    def test_polygon_collision(self):
        vertices = [Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(0, 2)]
        square = Polygon(vertices=vertices)
        inside_points = [Vec2(1, 1), Vec2(0.5, 0.5), Vec2(1.5, 1.5)]
        dir = Vec2(1, 0)
        for pt in inside_points:
            self.assertTrue(square.collision(pt, dir)[0])

        outside_points = [Vec2(-1, -1), Vec2(3, 1), Vec2(1, 3)]
        for pt in outside_points:
            self.assertFalse(square.collision(pt, dir)[0])

        edge_points = [Vec2(0, 1), Vec2(2, 1), Vec2(1, 0), Vec2(1, 2)]
        directions = [Vec2(-1, 0), Vec2(-1, 0), Vec2(0, -1), Vec2(0, 1)]
        for pt, dir in zip(edge_points, directions):
            self.assertFalse(square.collision(pt, dir)[0])


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestShapes("test_point_creation_and_container_add"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
