import matplotlib.pyplot as plt
import numpy as np


def draw_boundaries(radial_coordinates: list[tuple]):
    cartesian_coordinates = np.zeros((2, len(radial_coordinates)))
    for i, (radius, angle) in enumerate(radial_coordinates):
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        cartesian_coordinates[0][i] = x
        cartesian_coordinates[1][i] = y
    plt.plot(*cartesian_coordinates)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()
