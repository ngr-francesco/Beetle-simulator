import matplotlib.pyplot as plt
from src.shapes import Rectangle, HemiPlane, Ray, ShapeContainer, Point
from src.vector import Vec2
from src.robot_components import UltrasonicSensor
import numpy as np

"""
Set up one instance of each of the basic shapes. Check line collision of a single ray. Plot
"""
def test_basic_shapes():
    # draw_boundaries(radial_coordinates)
    all_shapes = ShapeContainer()
    Rectangle(10, 1, 10, 2)
    plane = HemiPlane.from_characteristic_equation(-1, 2, 1)
    ray0 = Ray(angle=0)
    intersection = ray0.collides_line(plane)
    if intersection:
        # Plot the ray
        ray0.to_line()
        Point.from_vec2(ray0.at(intersection))
    origin = Point(0,0)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    all_shapes.draw(ax)
    plt.show()

def test_collisions_ultrasonic_sensor():
    # draw_boundaries(radial_coordinates)
    all_shapes = ShapeContainer()
    HemiPlane.from_two_points(Vec2(10,4), Vec2(10,2))
    HemiPlane.from_two_points(Vec2(-9,4), Vec2(10,8)).rotate(np.pi/4)
    Rectangle(2,3,1,5).rotate(np.pi/4)
    origin = Point(0,0)
    sensor = UltrasonicSensor(20, 12)
    distances = sensor.scan(origin.to_vec(), visualize=True)
    print(distances)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    all_shapes.draw(ax)
    plt.show()
