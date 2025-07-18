import math
import random

import numpy as np


class Dot:
    """
        단위구(S²) 상의 점을 나타내는 클래스.
        theta: [0, π] 범위의 colatitude
        phi  : [0, 2π) 범위의 longitude
        """

    # 공용 오차 한계
    EPSILON = 1e-5

    def __init__(self, theta: float | None = None, phi: float | None = None):
        if theta is None or phi is None:
            # 균일 분포로 무작위 점 생성
            self.theta = math.acos(2 * random.random() - 1)
            self.phi = 2 * math.pi * random.random()
        else:
            self.theta = float(theta)
            self.phi = float(phi)

    def __sub__(self, other):
        cos_distance = (
                math.cos(self.theta) * math.cos(other.theta) +
                math.sin(self.theta) * math.sin(other.theta) * math.cos(self.phi - other.phi)
        )

        cos_distance = min(1.0, max(-1.0, cos_distance))
        return math.acos(cos_distance)

    def __eq__(self, other):
        return self - other < self.EPSILON

    def __ne__(self, other):
        epsilon = 1e-5
        return self - other >= epsilon

    def __copy__(self):
        return Dot(theta=self.theta, phi=self.phi)

    def angle(self, other1, other2):
        """두 점 self, other1, other2 사이의 각을 radian으로 반환."""
        a = self - other1
        b = self - other2
        c = other1 - other2

        cos_angle = (math.cos(c) - math.cos(a) * math.cos(b)) / (math.sin(a) * math.sin(b))
        cos_angle = min(1.0, max(-1.0, cos_angle))
        return math.acos(cos_angle)

    def to_cartesian(self):
        x = math.sin(self.theta) * math.cos(self.phi)
        y = math.sin(self.theta) * math.sin(self.phi)
        z = math.cos(self.theta)
        return np.array([x, y, z])

    @staticmethod
    def from_cartesian(P: np.ndarray) -> "Dot":
        """
        3차원 직교좌표계상의 단위구 점 P를 (theta, phi)로 변환하여 Dot 인스턴스로 반환.
        """
        x, y, z = P
        theta = math.acos(z)
        phi = math.atan2(y, x)
        if phi < 0:
            phi += 2 * math.pi
        return Dot(theta, phi)
