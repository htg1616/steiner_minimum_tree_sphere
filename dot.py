import math
import random
import copy

class Dot:
    def __init__(self, theta=None, phi=None):
        if theta is not None and phi is not None:
            self.theta = theta
            self.phi = phi
        else:
            self.theta = math.acos(2 * random.random() - 1) + math.pi/2
            self.phi = 2 * math.pi * random.random()

    def __sub__(self, other):
        cos_distance = (
                math.cos(self.theta) * math.cos(other.theta) +
                math.sin(self.theta) * math.sin(other.theta) * math.cos(self.phi - other.phi)
        )

        cos_distance = min(1.0, max(-1.0, cos_distance))
        return math.acos(cos_distance)

    def __eq__(self, other):  # 실수인데 부동소수점고려 해야함. ㅈ같네
        epsilon = 1e-5
        return self - other < epsilon

    def __ne__(self, other):
        epsilon = 1e-5
        return self - other >= epsilon

    def __copy__(self):
        return Dot(theta=self.theta, phi=self.phi)

    def angle(self, other1, other2):
        # (self, other1)과 (self, other2)가 이루는 각 반환
        a = self - other1
        b = self - other2
        c = other1 - other2

        cos_angle = (math.cos(c) - math.cos(a) * math.cos(b)) / (math.sin(a) * math.sin(b))
        cos_angle = min(1.0, max(-1.0, cos_angle))
        return math.acos(cos_angle)
