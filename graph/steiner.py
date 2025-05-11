import copy
import math

from geometry.dot import Dot
from geometry.fermat import find_projected_fermat_on_sphere
from .graph_base import GraphBase
from .mst import MinimalSpanningTree as MST


class SteinerTree(GraphBase):
    def __init__(self, mst: MST, si_option: bool = True):
        super().__init__(mst.vertices, mst.adj)
        self.original_count = len(mst.vertices) # 고정된 원래 정점 개수
        self.steiner_insertion(si_option)

    def find_min_angle_point(self, x, y):
        """edge (x, y)와 (y, z)가 가장 작은 각을 이루는 z 찾기"""
        point_x = self.vertices[x]
        point_y = self.vertices[y]

        min_point_index = None
        min_angle = 2 * math.pi

        for z in self.adj[y]:
            point_z = self.vertices[z]
            if point_x == point_y or point_y == point_z or point_z == point_x:  # 자기 자신은 제외
                continue

            angle = point_y.angle(point_x, point_z)

            if angle < min_angle:
                min_angle = angle
                min_point_index = z

        return min_point_index

    def steiner_insertion(self, si_option):
        """스타이너 포인트 삽입 알고리즘"""
        edges = list(self.edges())  # 현재 간선들의 복사본 (반복 중 변경되므로)

        for x, y in edges:
            if y not in self.adj[x]:  # 이미 처리된 간선
                continue

            point_x = self.vertices[x]
            point_y = self.vertices[y]
            z = self.find_min_angle_point(x, y)

            # 조건을 만족하면 스타이너 포인트 추가
            if z is not None:
                point_z = self.vertices[z]
                if point_y.angle(point_x, point_z) < math.pi * 2 / 3:
                    # 기존 간선 제거
                    self.adj[x].remove(y)
                    self.adj[y].remove(x)
                    self.adj[y].remove(z)
                    self.adj[z].remove(y)

                    # 새로운 스타이너 포인트 생성
                    s_idx = len(self.vertices)

                    if si_option:
                        point_s = find_projected_fermat_on_sphere(point_x, point_y, point_z)
                    else:
                        point_s = copy.copy(point_y)

                    self.vertices.append(point_s)
                    self.adj.append([])

                    # 스타이너 포인트와 다른 정점들 연결
                    self.add_edge(x, s_idx)
                    self.add_edge(y, s_idx)
                    self.add_edge(z, s_idx)


def build_smt(vertices: list[Dot], mst=None, si_option: bool = True) -> SteinerTree:
    """스타이너 트리를 생성하는 편의 함수"""
    return SteinerTree(mst, si_option)
