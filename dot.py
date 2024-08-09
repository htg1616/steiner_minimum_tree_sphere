import math


class Dot:
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi
        

def distance(dot1: Dot, dot2: Dot):  # 곡면에서 두 점을 잇는 최단 경로
    return math.acos(
        math.cos(dot1.theta) * math.cos(dot2.theta) + math.sin(dot1.theta) * math.sin(dot2.theta) * math.cos(
            dot1.phi - dot2.phi))

