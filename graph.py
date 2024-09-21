from contributed.sumopy.agilepy.lib_wx.test_glcanvas import vertices
from dot import *
import copy


# Class to represent a graph
class MinimalSpanningTree:
    def __init__(self, vertices: list):#생성자
        #정점과 간선 초기화
        self.vertices = vertices
        self.edges = []
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                self.edges.append((i, j))

        #mst 길이, 인접리스트, 간선
        self.mst_adj_list = None
        self.mst_edges = None

        self.kruskal()

    def kruskal(self):  # 크루스칼 알고리즘에 의해 생성된 최소 신장트리의 인접리스트를 반환
        def find(i):#부모 노드를 찾는 함수
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(x, y):#두 정점 x,y가 속학 집합을 합치는 함수
            x = find(x)
            y = find(y)
            if x < y:
                parent[x] = y
            else:
                parent[y] = x

        # mst 인접리스트, 간선, 부모노드, 비용 초기화
        self.mst_adj_list = [[] for _ in range(len(self.vertices))]
        self.mst_edges = []
        parent = [i for i in range(len(self.vertices))]
        #크루스칼 알고리즘
        self.edges.sort(key=lambda x: vertices[x[1]] - vertices[x[0]])
        for a, b in self.edges:
            if find(a) != find(b):
                self.mst_adj_list[a].append(b)
                self.mst_adj_list[b].append(a)
                self.mst_edges.append((a, b))
                self.mst_edges.append((b, a))
                union(a, b)

    def length(self):
        length = 0
        for a, b in self.mst_edges:
            length += vertices[a] - vertices[b]
        return length / 2

#MinimalSpanningTree를 상속받은 steinertree class
class SteinerTree(MinimalSpanningTree):
    def __init__(self, vertices: list):
        #MinimalSpanningTree의 생성자를 호출하고 크루스칼 알고리즘을 통해 mst를 생성
        super().__init__(vertices)

        self.si_vertices = None #SteinerTree의 좌표들 앞부분은 fixed points의 좌표, 뒷부분은 steiner points의 좌표
        self.si_adj_list = None #SteinerTree의 인접리스트, si_vertices의 인덱스로 구성됨
        self.si_edges = None #StwinerTree의 간선 표현, si_vertices의 인덱스로 구성됨

        self.steiner_insertion()

    def find_min_angle_point(self, x, y):
        #edge (t_x, t_y)와 (t_y, t_z)가 가장 작은 각을 이루는 z 찾기, z는 스타이너 포인트도 될 수 있음
        point_x = self.vertices[x]
        point_y = self.vertices[y]

        min_point_index = None
        min_angel = 2 * math.pi #각도 최솟값을 2pi로 초기화
        #y의 모든 인접점에 대해  최소각을 가지는 z를 찾음
        for z in self.si_adj_list[point_y]:
            point = self.si_vertices[z]
            if point != point_x and point_y.angle(point_x, point) < min_angel:
                min_angel = point_y.angle(point_x, point)
                min_point_index = z
        return min_point_index


    def steiner_insertion(self):
        #steiner insertion 알고리즘

        self.si_adj_list = copy.deepcopy(self.mst_adj_list)#기존 mst 인접리스트 복사
        self.si_vertices = self.vertices[:]

        for x, y in self.mst_edges: #모든 mst 간선에 대해 반복
            point_x = self.si_vertices[x]
            point_y = self.si_vertices[y]
            z = self.find_min_angle_point(x, y) #x,y 각도가 작은 각을 이루는 z찾기
            point_z = self.si_vertices[z]

            #z가 존재하고, x, y, z가 이루는 각이 120도보다 작으면 steiner point를 추가
            if not z and point_y.angle(point_x, point_z) < math.pi * 2 / 3:
                #기존 간선 제거
                self.si_adj_list[x].remove(y)
                self.si_adj_list[y].remove(x)
                self.si_adj_list[y].remove(z)
                self.si_adj_list[z].remove(y)

                #새로운 steiner point를 y를 복사하여 만든후 간선 연결
                s_index = len(self.si_vertices)
                s = y.copy()
                self.si_vertices.append(s)
                self.si_adj_list[x].append(s_index)
                self.si_adj_list[y].append(s_index)
                self.si_adj_list[z].append(s_index)


class LocalOptimizedGraph(SteinerTree):
    def __init__(self, vertices: list):
        super().__init__(vertices) #mst와 si-insertion을 이용하여 스타이너 트리의 위상 생성
