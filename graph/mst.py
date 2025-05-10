from geometry.dot import Dot
from .graph_base import GraphBase


class MinimalSpanningTree(GraphBase):
    def __init__(self, vertices: list[Dot]):
        super().__init__(vertices)
        self.build()

    def build(self):
        # 크루스칼 알고리즘을 이용한 MST 구축
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(x, y):
            x, y = find(x), find(y)
            if x < y:
                parent[x] = y
            else:
                parent[y] = x

        # 모든 간선을 생성하고 거리순으로 정렬
        edges = []
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                edges.append((i, j, self.vertices[i] - self.vertices[j]))
        edges.sort(key=lambda x: x[2])

        # 서로소 집합 초기화
        parent = [i for i in range(len(self.vertices))]

        # 크루스칼 알고리즘 실행
        for u, v, _ in edges:
            if find(u) != find(v):
                self.add_edge(u, v)
                union(u, v)


def build_mst(vertices: list[Dot]) -> MinimalSpanningTree:
    """MST를 생성하는 편의 함수"""
    return MinimalSpanningTree(vertices)
