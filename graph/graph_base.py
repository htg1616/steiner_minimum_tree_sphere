from copy import deepcopy

from geometry.dot import Dot


class GraphBase:
    def __init__(self, vertices: list[Dot], adj: list[list[int]] = None):
        self.vertices = deepcopy(vertices)
        self.adj = deepcopy(adj)
        if not self.adj:
            self.adj = [[] for _ in self.vertices]

    def add_edge(self, u: int, v: int):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def edges(self):
        for u, nbrs in enumerate(self.adj):
            for v in nbrs:
                yield u, v

    def length(self) -> float:
        total = 0
        for u, v in self.edges():
            if u > v: continue
            total += self.vertices[u] - self.vertices[v]
        return total
