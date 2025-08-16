import matplotlib.pyplot as plt
import numpy as np
from src.vector import Vec2
from src.shapes import (
    Rectangle,
    HemiPlane,
    Shape,
    Ray,
    ShapeContainer,
    Point,
    RandomPolygon,
)
from src.robot_components import UltrasonicSensor


# draw_boundaries(radial_coordinates)
for k in range(3):
    all_shapes = ShapeContainer()
    room = RandomPolygon(n_edges=20, max_size=120)
    origin = Point(x=0, y=0)
    sensor = UltrasonicSensor(20, 120)
    distances = sensor.scan(origin.to_vec(), visualize=True)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    all_shapes.draw(ax)
    plt.show()
    all_shapes.clear()
