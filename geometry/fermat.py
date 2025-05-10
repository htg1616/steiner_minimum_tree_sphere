import numpy as np
import math
from .dot import Dot

def fermat_2d(A, B, C):
    """
    2차원에서 세 점 A, B, C (np.array, shape (2,))에 대해 토리첼리(페르마) 점을 구한다.
    만약 삼각형의 어느 한 각이 120° 이상이면 그 꼭짓점을 반환한다.
    그렇지 않으면, 삼각형의 orientation에 따라,
      한 변(B-A)은 -60° (시계 방향)로, 다른 변(C-A)은 +60° (반시계 방향)로 회전하여
    외부에 정삼각형을 구성한 후, 선(C, D)와 선(B, E)의 교점을 반환한다.
    """

    def norm(v):
        return np.linalg.norm(v)

    def angle(P, Q, R):
        # Q를 기준으로 P와 R 사이의 각
        u = P - Q
        v = R - Q
        return math.acos(np.dot(u, v) / (norm(u) * norm(v)))

    angleA = angle(B, A, C)
    angleB = angle(A, B, C)
    angleC = angle(A, C, B)

    if angleA >= 2 * math.pi / 3:
        return A
    if angleB >= 2 * math.pi / 3:
        return B
    if angleC >= 2 * math.pi / 3:
        return C

    # 삼각형의 orientation 결정: det = (B-A)_x*(C-A)_y - (B-A)_y*(C-A)_x
    det = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    # 60° 회전 행렬 정의
    cos60 = 0.5
    sin60 = math.sqrt(3) / 2

    # 삼각형의 외부에 정삼각형이 구성되도록, B-A와 C-A에 대해 반대 방향의 회전을 적용
    if det > 0:  # 삼각형이 반시계 방향이면
        R_B = np.array([[cos60, sin60], [-sin60, cos60]])  # -60° (시계 방향)
        R_C = np.array([[cos60, -sin60], [sin60, cos60]])  # +60° (반시계 방향)
    else:  # 삼각형이 시계 방향이면
        R_B = np.array([[cos60, -sin60], [sin60, cos60]])  # +60° (반시계 방향)
        R_C = np.array([[cos60, sin60], [-sin60, cos60]])  # -60° (시계 방향)

    D = A + R_B.dot(B - A)
    E = A + R_C.dot(C - A)

    # 선(C, D): F = C + t*(D - C)
    # 선(B, E): F = B + s*(E - B)
    M = np.column_stack((D - C, -(E - B)))
    rhs = B - C
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return (A + B + C) / 3.0
    t = sol[0]
    F = C + t * (D - C)
    return F


def fermat_3d(A, B, C):
    """
    3차원에서 세 점 A, B, C (np.array, shape (3,))가 이루는 평면에 사영하여
    2차원 closed-form 해법으로 페르마점을 계산하고, 다시 3차원으로 복원하여 반환한다.
    """
    # 기준: A를 원점으로 하고, e1 = (B-A)/||B-A||, e2 = 정규화(C-A - projection of (C-A) on e1)
    AB = B - A
    e1 = AB / np.linalg.norm(AB)
    CA = C - A
    proj = np.dot(CA, e1) * e1
    e2 = CA - proj
    norm_e2 = np.linalg.norm(e2)
    if norm_e2 < 1e-10:
        raise ValueError("세 점이 거의 일직선상에 있습니다.")
    e2 = e2 / norm_e2

    # 각 3D 점 p를 평면에 사영하여 2D 좌표로 변환: (dot(p-A, e1), dot(p-A, e2))
    def project_to_2d(p):
        v = p - A
        return np.array([np.dot(v, e1), np.dot(v, e2)])

    A2 = project_to_2d(A)
    B2 = project_to_2d(B)
    C2 = project_to_2d(C)

    fermat_2d_pt = fermat_2d(A2, B2, C2)

    # 3D 복원: A + (x*e1 + y*e2)
    fermat_3d_pt = A + fermat_2d_pt[0] * e1 + fermat_2d_pt[1] * e2
    return fermat_3d_pt


def project_to_sphere(F, n):
    """
    3D 점 F와 평면의 단위 법선 벡터 n가 주어졌을 때,
    F + λ n가 단위구에 놓이도록 하는 λ를 구하고, P = F + λ n를 반환.
    즉, ||F + λ n|| = 1 인 λ를 구한다.
    """
    # ||F + λ n||^2 = ||F||^2 + 2λ (F·n) + λ^2 = 1
    # => λ^2 + 2 (F·n) λ + (||F||^2 - 1) = 0
    a_coef = 1.0
    b_coef = 2.0 * np.dot(F, n)
    c_coef = np.dot(F, F) - 1.0
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0:
        if disc > -1e-8:
            disc = 0.0
        else:
            raise ValueError("No real solution for projection onto the sphere.")
    sqrt_disc = math.sqrt(disc)
    # 두 근 중 |λ|가 더 작은 것을 선택 (F에서 최소 이동)
    lambda1 = (-b_coef + sqrt_disc) / (2 * a_coef)
    lambda2 = (-b_coef - sqrt_disc) / (2 * a_coef)
    if abs(lambda1) < abs(lambda2):
        lam = lambda1
    else:
        lam = lambda2
    P = F + lam * n
    return P


def find_projected_fermat_on_sphere(a: Dot, b: Dot, c: Dot):
    """
    입력: Dot 클래스의 인스턴스 3개 (구면상의 점)
    동작:
      1. 각 Dot을 3차원 직교좌표 (np.array, shape(3,))로 변환.
      2. 이 3점이 이루는 평면의 단위 법선 벡터 n을 구한다.
      3. fermat_3d 함수를 이용해 3차원 평면상 페르마점 F를 구한다.
      4. F에 대해, F + λ n가 단위구(구면)에 놓이도록 하는 λ를 구하여 P = F + λ n를 계산.
      5. P를 반환 (P는 구면상에 사영된 페르마점).
    """
    A, B, C = a.to_cartesian(), b.to_cartesian(), c.to_cartesian()

    # 평면의 단위 법선 벡터 n: n = (B-A) x (C-A) / ||(B-A) x (C-A)||
    cross_vec = np.cross(B - A, C - A)
    norm_cross = np.linalg.norm(cross_vec)
    if norm_cross < 1e-10:
        raise ValueError("입력된 점들이 거의 공선상에 있습니다.")
    n = cross_vec / norm_cross

    # 3차원 평면상의 Fermat point F 구하기 (closed-form 방식)
    F = fermat_3d(A, B, C)

    # F는 평면상에 있으나 구면상에 있지 않으므로, 평면 법선 방향으로 λ만큼 이동시켜
    # P = F + λ n가 단위구(구면)에 놓이도록 한다.
    P = project_to_sphere(F, n)
    dot_P = Dot.from_cartesian(P)
    return dot_P