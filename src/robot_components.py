from src.shapes import Ray, ShapeContainer, Point
from src.vector import Vec2
import numpy as np

DEFAULT_POINT_DENSITY = 20
MAX_DEPTH = 20


class UltrasonicSensor:
    def __init__(self, point_density: int, max_depth: float):
        self.point_density = point_density
        self.max_depth = max_depth

    def scan(self, origin: Vec2, offset_angle: float = 0, visualize: bool = False):
        if ShapeContainer._instance is None:
            raise RuntimeError("No ShapeContainer found. Couldn't scan environment.")
        distances = []
        for k in range(self.point_density + 1):
            angle = np.pi * k / self.point_density + offset_angle
            ray = Ray(angle, origin, length=self.max_depth)
            cur_collision = self.max_depth
            for shape in ShapeContainer._instance:
                collision = ray.collide(shape)
                if collision and collision < cur_collision:
                    cur_collision = collision

            if visualize:
                if cur_collision < self.max_depth:
                    Point.from_vec2(ray.at(cur_collision))
                ray.to_line()
            distances.append((angle, cur_collision))
        return distances


class Robot:
    def __init__(self, origin: Vec2= Vec2(0,0)):
        self.position = origin
        self.rotation = 0.0
        self.ultrasonic_sensor = UltrasonicSensor(DEFAULT_POINT_DENSITY, MAX_DEPTH)
    
    def scan(self, visualize:bool = False):
        self.ultrasonic_sensor.scan(self.position, self.rotation, visualize)
    

    def rotate(self, angle: float):
        self.rotation += angle


    def move(self, direction: Vec2, speed: float):
        self.position += direction.normalized()*speed