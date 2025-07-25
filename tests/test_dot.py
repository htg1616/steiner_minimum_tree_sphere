import pytest
import math
from geometry.dot import Dot

def test_ne():
    """Dot.__ne__가 올바르게 작동하는지 테스트합니다."""
    d1 = Dot(theta=1, phi=1)
    d2 = Dot(theta=1, phi=1)
    d3 = Dot(theta=2, phi=2)

    # 두 점이 거의 같을 때 False를 반환해야 합니다.
    assert not (d1 != d2)

    # 두 점이 다를 때 True를 반환해야 합니다.
    assert d1 != d3

def test_angle():
    """Dot.angle이 올바르게 작동하는지 테스트합니다."""
    # 북극점
    north_pole = Dot(theta=0, phi=0)
    # 적도 위의 두 점
    equator_point1 = Dot(theta=math.pi / 2, phi=0)
    equator_point2 = Dot(theta=math.pi / 2, phi=math.pi / 2)

    # 북극점에서 두 적도 위의 점을 바라보는 각도는 pi/2 여야 합니다.
    angle = north_pole.angle(equator_point1, equator_point2)
    assert math.isclose(angle, math.pi / 2, rel_tol=1e-9)

    # 적도 위의 한 점에서 북극점과 다른 적도 위의 점을 바라보는 각도
    angle2 = equator_point1.angle(north_pole, equator_point2)
    assert math.isclose(angle2, math.pi / 2, rel_tol=1e-9)

