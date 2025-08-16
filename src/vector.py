from dataclasses import dataclass
import numpy as np

EPSILON = 1e-9


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    # --- basic ops ---
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def _matmul(self, matrix: "RotationMatrix"):
        return Vec2(self.dot(matrix.first), self.dot(matrix.second))

    def normalized(self):
        l = self.length()
        if l < EPSILON:
            return Vec2(0.0, 0.0)
        return self / l

    def length(self):
        return np.sqrt(self.dot(self))

    def dot(self, other: "Vec2"):
        return self.x * other.x + self.y * other.y

    def tolist(self):
        return [self.x, self.y]

    def rotate(self, angle_rad: float):
        matrix = RotationMatrix(
            Vec2(np.cos(angle_rad), np.sin(angle_rad)),
            Vec2(-np.sin(angle_rad), np.cos(angle_rad)),
        )
        return self._matmul(matrix)


@dataclass
class RotationMatrix:
    first: Vec2
    second: Vec2


ROTATIONS = {
    90: RotationMatrix(Vec2(0, 1), Vec2(-1, 0)),
    180: RotationMatrix(Vec2(-1, 0), Vec2(0, -1)),
    270: RotationMatrix(Vec2(0, -1), Vec2(1, 0)),
    360: RotationMatrix(Vec2(1, 0), Vec2(0, 1)),
}
