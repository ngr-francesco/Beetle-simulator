from src.shapes import Ray, ShapeContainer, Point
from src.vector import Vec2
import numpy as np


class UltrasonicSensor:
    def __init__(self, point_density, max_depth):
        self.point_density = point_density 
        self.max_depth = max_depth
    
    def scan(self, origin: Vec2, visualize: bool = False):
        if ShapeContainer._instance is None:
            raise RuntimeError("No ShapeContainer found. Couldn't scan environment.")
        distances = []
        for k in range(self.point_density+1):
            angle = np.pi * k/self.point_density
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
                
    
            

        
