from dot import *
import copy


# Class to represent a graph
class MinimalSpanningTree:
    def __init__(self, vertices: list):
        self.vertices = vertices
        self.edges = []
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                self.edges.append((i, j))

        self.mst_length = None
        self.mst_adj_list = None
        self.mst_edges = None

    def kruskal(self):  # 크루스칼 알고리즘에 의해 생성된 최소 신장트리의 인접리스트를 반환
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(x, y):
            x = find(x)
            y = find(y)
            if x < y:
                parent[x] = y
            else:
                parent[y] = x

        self.mst_adj_list = [[] for _ in range(len(self.vertices))]
        self.mst_edges = []
        total_cost = 0
        parent = [i for i in range(len(self.vertices))]

        self.edges.sort(key=lambda x: x[1] - x[0])
        for cost, a, b in self.edges:
            if find(a) != find(b):
                self.mst_adj_list[a].append(b)
                self.mst_edges.append((a, b))
                union(a, b)
                total_cost += cost

        self.mst_length = total_cost


class SteinerTree(MinimalSpanningTree):
    def __init__(self, vertices: list):
        super().__init__(vertices)
        self.kruskal()
        self.si_adj_list = None
        self.si_edges = None
        self.si_length = None
        self.si_vertices = None

    def find_min_angle_point(self, x, y):
        #edge (t_x, t_y)와 (t_y, t_z)가 가장 작은 각을 이루는 z 찾기, z는 스타이너 포인트도 될 수 있음
        point_x = self.vertices[x]
        point_y = self.vertices[y]

        min_point_index = None
        min_angel = 2 * math.pi
        for z in self.mst_adj_list[point_y]:
            point = self.vertices[z]
            if point != point_x and point_y.angle(point_x, point) < min_angel:
                min_angel = point_y.angle(point_x, point)
                min_point_index = z
        return min_point_index


    def steiner_insertion(self):
        self.si_adj_list = copy.deepcopy(self.mst_adj_list)
        self.si_vertices = []
        for x, y in self.mst_edges:
            point_x = self.vertices[x]
            point_y = self.vertices[y]
            z = self.find_min_angle_point(x, y)
            point_z = self.vertices[z]
            if not z and point_y.angle(point_x, point_z) < math.pi * 2 / 3:
                self.si_adj_list[x].remove(y)
                self.si_adj_list[y].remove(x)
                self.si_adj_list[y].remove(z)
                self.si_adj_list[z].remove(y)

                s_index = len(self.si_vertices)
                s = y.copy()
                self.si_vertices.append(s)
                self.si_adj_list[x].append(s_index)
                self.si_adj_list[y].append(s_index)
                self.si_adj_list[z].append(s_index)

            