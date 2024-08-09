import math
import random


class Dot:
    def __init__(self, theta=None, phi=None):
        if theta and phi:
            self.theta = theta
            self.phi = phi
        else:
            self.theta = 2 * math.pi * random.random()
            self.phi = math.acos(2 * random.random() - 1)

    def __sub__(self, other):
        return math.acos(
            math.cos(self.theta) * math.cos(other.theta) + math.sin(self.theta) * math.sin(other.theta) * math.cos(
                self.phi - other.phi))

    def __eq__(self, other):
        return self.theta == other.theta and self.phi == other.phi

    def __ne__(self, other):
        return self.theta != other.theta or self.phi != other.phi
