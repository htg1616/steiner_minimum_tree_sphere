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

    def angle(self, other1, other2):
        #(self, other1)과 (self, other2)가 이루는 각 반환
        return math.acos(math.cos(other1-other2) - math.cos(other1-self) * math.cos(other2-self)) / (math.sin(other1-self)*math.sin(other2-self))
