from dot import *


# Class to represent a graph
class Graph:
    def __init__(self, vertices: list):
        self.V = vertices
        self.graph = []
        for i in range(len(self.V)):
            for j in range(i + 1, len(self.V)):
                self.graph.append((i, j, i - j))

    def find(self, parent, i):
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent, x, y):
        x = self.find(parent, x)
        y = self.find(parent, y)
        if x < y:
            parent[x] = y
        else:
            parent[y] = x

    def kruskal(self):
        result = []
        total_cost = 0
        parent = [i for i in range(len(self.V))]

        sort(self.graph, key=lambda x: x[2])
        for cost, a, b in self.graph:
            if self.find(parent, a) != self.find(parent, b):
                union(parent, a, b)
                total_cost += cost

        return result, total_cost
