import copy
import math
import itertools
import torch

from geometry.dot import Dot
from geometry.fermat import find_projected_fermat_on_sphere, is_point_in_spherical_triangle
from .mst import MinimalSpanningTree as MST
from .enums import InsertionMode


class SteinerTree:
    def __init__(self, mst: MST, insertion_mode):
        self.fixed_vertices = copy.deepcopy(mst.vertices)  # 고정된 정점들
        self.fixed_count = len(mst.vertices) # 고정된 원래 정점 개수
        self.steiner_vertices = [] # 스타이너 포인트를 저장할 리스트
        self.steiner_count = 0 # 스타이너 포인트 개수
        self.adj = copy.deepcopy(mst.adj)  # 인접 리스트 복사
        self.insertion_mode = insertion_mode

        self.steiner_insertion()

    def get_vertice(self, u: int) -> Dot:
        """정점 u의 좌표를 반환"""
        if u < self.fixed_count:
            return self.fixed_vertices[u]
        else:
            return self.steiner_vertices[u - self.fixed_count]

    def get_distance(self, u: int, v: int) -> float:
        """두 정점 u, v 사이의 거리를 반환"""
        point_u = self.get_vertice(u)
        point_v = self.get_vertice(v)
        return point_u - point_v

    def length(self) -> float:
        """스타이너 트리의 총 길이를 계산"""
        total_length = 0.0
        total_count = self.fixed_count + self.steiner_count
        for u in range(total_count):
            for v in self.adj[u]:
                if u < v:
                    total_length += self.get_distance(u, v)
        return total_length

    def get_vertices_angle_tensor(self, device):
        """모든 정점의 theta, phi 좌표를 텐서로 반환 (optimizer 인터페이스)"""
        vertices = []

        # 고정 정점들의 좌표
        for i in range(self.fixed_count):
            dot = self.fixed_vertices[i]
            vertices.append([dot.theta, dot.phi])

        # 스타이너 정점들의 좌표
        for i in range(self.steiner_count):
            dot = self.steiner_vertices[i]
            vertices.append([dot.theta, dot.phi])

        return torch.tensor(vertices, dtype=torch.float64, device=device)

    def get_vertices_xyz_tensor(self, device):
        """모든 정점의 x, y, z 좌표를 텐서로 반환 (optimizer 인터페이스)"""
        angle_tensor = self.get_vertices_angle_tensor(device)

        # theta, phi를 x, y, z로 변환
        theta = angle_tensor[:, 0]  # theta 좌표
        phi = angle_tensor[:, 1]    # phi 좌표

        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack([x, y, z], dim=1)

    def get_edge_index(self, device):
        """간선 인덱스를 텐서로 반환 (optimizer 인터페이스)"""
        total_count = self.fixed_count + self.steiner_count
        edges = []

        for u in range(total_count):
            for v in self.adj[u]:
                if u < v:  # 중복 제거
                    edges.append([u, v])

        if len(edges) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)

        return torch.tensor(edges, dtype=torch.long, device=device)

    def get_steiner_mask(self, device):
        """스타이너 점 마스크를 불리언 텐서로 반환 (optimizer 인터페이스)"""
        total_count = self.fixed_count + self.steiner_count
        mask = torch.zeros(total_count, dtype=torch.bool, device=device)
        if self.steiner_count > 0:
            mask[self.fixed_count:] = True
        return mask

    def steiner_insertion(self):
        """thompson's method 적용하여 스타이너 포인트 삽입"""

        for i in range(self.fixed_count):
            if len(self.adj[i]) < 2: # 인접한 정점이 1개인 경우 스타이너 포인트 삽입 불가
                continue

            inserted_flag = True
            while inserted_flag:
                inserted_flag = False # 스타이너 포인트 삽입이 일어날때 까지 반복
                smallest_anlge = 2 * math.pi
                smallest_j = -1
                smallest_k = -1
                for j, k in itertools.combinations(self.adj[i], 2):
                    angle = self.get_vertice(i).angle(self.get_vertice(j), self.get_vertice(k))
                    if angle < smallest_anlge:
                        smallest_anlge = angle
                        smallest_j = j
                        smallest_k = k

                if smallest_anlge < 2 * math.pi / 3:  # 120도 미만인 경우
                    inserted_flag = self.add_steiner_vertice(i, smallest_j, smallest_k) # 스타이너 포인트가 삽입되었으므로 다시 반복

    def add_steiner_vertice(self, u: int, v: int, w: int) -> bool:
        """
        스타이너 포인트를 추가하고 인접 리스트 업데이트
        : param u: 스타이너 포인트를 추가할 각의 꼭짓점
        : param v, w: u에 인접한 두 정점
        """
        point_u = self.get_vertice(u)
        point_v = self.get_vertice(v)
        point_w = self.get_vertice(w)

        # 스타이너 점 계산
        steiner_point = find_projected_fermat_on_sphere(point_u, point_v, point_w)

        # 구면삼각형 내부 검증
        is_inside = is_point_in_spherical_triangle(steiner_point, point_u, point_v, point_w)

        if not is_inside:
            raise ValueError("스타이너 점이 구면삼각형 내부에 위치하지 않습니다.")

        # 스타이너 점이 기존 정점과 동일한지 확인
        for vertex in [point_u, point_v, point_w]:
            if steiner_point == vertex:
                raise ValueError("스타이너 점이 기존 정점과 동일합니다.")

        old_len = (point_u - point_v) + (point_u - point_w)
        new_len = (steiner_point - point_v) + (steiner_point - point_w) + (steiner_point - point_u)

        if self.insertion_mode == InsertionMode.DECREASE_ONLY and new_len > old_len:
            return False
        elif self.insertion_mode == InsertionMode.INSERT_WITH_OPTIMIZE:
            raise NotImplementedError("INSERT_WITH_OPTIMIZE 모드는 아직 구현되지 않았습니다.")


        # 스타이너 점을 리스트에 추가
        self.steiner_vertices.append(steiner_point)
        new_index = self.fixed_count + self.steiner_count
        self.steiner_count += 1

        # 인접 리스트 업데이트
        self.adj[u].remove(v)
        self.adj[u].remove(w)
        self.adj[v].remove(u)
        self.adj[w].remove(u)
        self.adj[u].append(new_index)
        self.adj[v].append(new_index)
        self.adj[w].append(new_index)

        # 새로 추가된 스타이너 점과 기존 정점들 간의 연결
        self.adj.append([u, v, w])

        return True