from src.shapes import Rectangle
from src.vector import Vec2

class SquareRoom:
    def __init__(self, size: float, origin: Vec2):
        self.size = size
        self.shapes = Rectangle(origin.x, origin.y, size, size)

