import matplotlib.pyplot as plt
import numpy as np
from src.vector import Vec2
from src.shapes import Rectangle, HemiPlane, Shape, Ray, ShapeContainer, Point
from src.robot_components import UltrasonicSensor
from src.rooms import SquareRoom


# draw_boundaries(radial_coordinates)
all_shapes = ShapeContainer()
room = SquareRoom(10,Vec2(-1,-3))
origin = Point(0,0)
sensor = UltrasonicSensor(20, 12)
distances = sensor.scan(origin.to_vec(), visualize=True)
print(distances)
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_aspect("equal", adjustable="box")
all_shapes.draw(ax)
plt.show()
