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
from src.robot_components import Robot


def draw():
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    all_shapes.draw(ax)
    plt.show()


# draw_boundaries(radial_coordinates)
for k in range(1):
    all_shapes = ShapeContainer()
    room = RandomPolygon(n_edges=20, max_size=120)
    origin = Vec2(x=0, y=0)
    robot = Robot(origin=origin)
    robot.scan(visualize=True)
    draw()
    robot.rotate(np.pi)
    robot.scan(visualize=True)
    draw()
    robot.move(Vec2(-1, 0), speed=20)
    robot.scan(visualize=True)
    draw()
    all_shapes.clear()
