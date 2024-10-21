import numpy as np

from dot import *


# Class to represent a graph
class MinimalSpanningTree:
    def __init__(self, vertices: list):  # 생성자
        # 정점과 간선 초기화
        self.vertices = vertices
        self.edges = []
        for i in range(len(self.vertices)):
            for j in range(i + 1, len(self.vertices)):
                self.edges.append((i, j))

        # mst 길이, 인접리스트, 간선
        self.adj_list = None

        self.kruskal()

    def kruskal(self):  # 크루스칼 알고리즘에 의해 생성된 최소 신장트리의 인접리스트를 반환
        def find(i):  # 부모 노드를 찾는 함수
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(x, y):  # 두 정점 x,y가 속학 집합을 합치는 함수
            x = find(x)
            y = find(y)
            if x < y:
                parent[x] = y
            else:
                parent[y] = x

        # mst 인접리스트, 서로소 집합 초기화
        self.adj_list = [[] for _ in range(len(self.vertices))]
        parent = [i for i in range(len(self.vertices))]
        # 크루스칼 알고리즘
        self.edges.sort(key=lambda x: self.vertices[x[1]] - self.vertices[x[0]])
        for a, b in self.edges:
            if find(a) != find(b):
                self.adj_list[a].append(b)
                self.adj_list[b].append(a)
                union(a, b)

    def length(self):
        length = 0
        for a, b in self.edges:
            length += self.vertices[a] - self.vertices[b]
        return length / 2


# MinimalSpanningTree를 상속받은 steinertree class
class SteinerTree:
    def __init__(self, vertices: list[Dot], mst_adj_list: list[list[int]]):
        self.vertices = vertices
        self.total_vertices = vertices[:]  # SteinerTree의 좌표들 앞부분은 fixed points의 좌표, 뒷부분은 steiner points의 좌표
        self.adj_list = copy.deepcopy(mst_adj_list) # SteinerTree의 인접리스트, total_vertices의 인덱스로 구성됨, 기존 mst 간접 리스트 복사하여 초기화
        self.mst_edges = [(x, y) for x in range(len(mst_adj_list)) for y in mst_adj_list[x]]

        self.steiner_insertion()
        self.si_vertices = self.total_vertices[len(self.vertices):]

    def find_min_angle_point(self, x, y):
        # edge (t_x, t_y)와 (t_y, t_z)가 가장 작은 각을 이루는 z 찾기, z는 스타이너 포인트도 될 수 있음
        point_x = self.total_vertices[x]
        point_y = self.total_vertices[y]

        min_point_index = None
        min_angel = 2 * math.pi  # 각도 최솟값을 2pi로 초기화
        # y의 모든 인접점에 대해  최소각을 가지는 z를 찾음
        for z in self.adj_list[y]:
            point_z = self.total_vertices[z]
            if z != x and point_y.angle(point_x, point_z) < min_angel:
                min_angel = point_y.angle(point_x, point_z)
                min_point_index = z
        return min_point_index

    def steiner_insertion(self):
        # steiner insertion 알고리즘

        for x, y in self.mst_edges:  # 모든 mst 간선에 대해 반복
            point_x = self.total_vertices[x]
            point_y = self.total_vertices[y]
            z = self.find_min_angle_point(x, y)  # x,y 각도가 작은 각을 이루는 z찾기

            # z가 존재하고, x, y, z가 이루는 각이 120도보다 작으면 steiner point를 추가
            if z is not None:
                point_z = self.total_vertices[z]
                if point_y.angle(point_x, point_z) < math.pi * 2 / 3:
                    # 기존 간선 제거
                    self.adj_list[x].remove(y)
                    self.adj_list[y].remove(x)
                    self.adj_list[y].remove(z)
                    self.adj_list[z].remove(y)

                    # 새로운 steiner point를 y를 복사하여 만든후 간선 연결
                    s_index = len(self.total_vertices)
                    s = copy.copy(point_y)
                    self.total_vertices.append(s)
                    self.adj_list[x].append(s_index)
                    self.adj_list[y].append(s_index)
                    self.adj_list[z].append(s_index)

class LocalOptimizedGraph:
    def __init__(self, vertices: list[Dot], si_vertices: list[Dot], adj_list: list[list[int]]):
        self.vertices = vertices
        self.si_theta = np.array([i.theta for i in si_vertices])
        self.si_phi = np.array([i.phi for i in si_vertices])
        self.adj_list = adj_list
        self.si_vertices = self.optimze()

    @staticmethod
    def great_circle_gradient(v_vertex: Dot, u_vertex: Dot) -> tuple[float, float]:
        """
        Computes the gradient of the great-circle distance between v_vertex and u_vertex
        with respect to theta and phi of v_vertex.

        :param v_vertex: The Dot instance whose position is being optimized.
        :param u_vertex: The other Dot instance.
        :return: Tuple of partial derivatives (dL/dtheta_v, dL/dphi_v).
        """
        theta1 = v_vertex.theta
        phi1 = v_vertex.phi
        theta2 = u_vertex.theta
        phi2 = u_vertex.phi
        grad_theta = (-(-math.sin(theta1) * math.cos(theta2) + math.cos(theta1) * math.sin(theta2) * math.cos(
            phi2 - phi1)) /
                      math.sqrt(1 - (
                                  math.cos(theta1) * math.cos(theta2) + math.sin(theta1) * math.sin(theta2) * math.cos(
                              phi2 - phi1)) ** 2))
        grad_phi = (-(math.sin(theta1) * math.sin(theta2) * math.sin(phi2 - phi1)) /
                    math.sqrt(1 - (math.cos(theta1) * math.cos(theta2) + math.sin(theta1) * math.sin(theta2) * math.cos(
                        phi2 - phi1)) ** 2))
        return grad_theta, grad_phi

    def compute_gradient(self):
        # 각 점들의 gradient값은 각 점들과 이웃한 점들과의 great_circle_gradient의 합으로 구할 수 있다. 이를 이용해보자.
        """
        각 si_vertices의 theta와 phi에 대한 length의 미분값을 계산

        :return: Tuple of gradients (grad_theta, grad_phi) as numpy arrays.
        """
        # theta, phi에 대한 그래디언트 값을 저장할 값이 0인 numpy_array 생성
        n = len(self.si_vertices)
        grad_theta = np.zeros(n)
        grad_phi = np.zeros(n)

        # 각 정점에 대해 그래디언트 계산
        for v_index in range(n):
            v_vertex = self.si_vertices[v_index]
            neighbors = self.adj_list[v_index]

            for u_index in neighbors:
                u_vertex = self.vertices[u_index]

                # 그래디언트를 계산
                grad_theta_v, grad_phi_v = self.great_circle_gradient(v_vertex, u_vertex)

                # 그래디언트를 numpy_array에 누적
                grad_theta[v_index] += grad_theta_v
                grad_phi[v_index] += grad_phi_v

        return grad_theta, grad_phi

    def optimze(self) -> list[Dot]:
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        num_iterations = 1000  # 반복 횟수

        # 1차 및 2차 모멘트 초기화
        m_theta = np.zeros_like(self.si_theta)
        m_phi = np.zeros_like(self.si_phi)
        v_theta = np.zeros_like(self.si_theta)
        v_phi = np.zeros_like(self.si_phi)

        for t in range(1, num_iterations + 1):
            # 그래디언트 계산
            grad_theta, grad_phi = self.compute_gradient()

            # 1차 모멘트 업데이트
            m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
            m_phi = beta1 * m_phi + (1 - beta1) * grad_phi

            # 2차 모멘트 업데이트
            v_theta = beta2 * v_theta + (1 - beta2) * (grad_theta ** 2)
            v_phi = beta2 * v_phi + (1 - beta2) * (grad_phi ** 2)

            # 편향 보정
            m_hat_theta = m_theta / (1 - beta1 ** t)
            m_hat_phi = m_phi / (1 - beta1 ** t)
            v_hat_theta = v_theta / (1 - beta2 ** t)
            v_hat_phi = v_phi / (1 - beta2 ** t)

            # 매개변수 업데이트
            self.si_theta -= learning_rate * m_hat_theta / (np.sqrt(v_hat_theta) + epsilon)
            self.si_phi -= learning_rate * m_hat_phi / (np.sqrt(v_hat_phi) + epsilon)

            # θ와 φ의 범위를 보정
            self.si_theta = np.clip(self.si_theta, 0, np.pi)
            self.si_phi = np.mod(self.si_phi, 2 * np.pi)

            # 진행 상황 출력 (선택 사항)
            if t % 100 == 0 or t == 1:
                current_length = self.length()
                print(f"Iteration {t}, Total Length: {current_length}")

        # 최적화된 si_vertices 생성
        optimized_si_vertices = [Dot(theta, phi) for theta, phi in zip(self.si_theta, self.si_phi)]

        return optimized_si_vertices

    def length(self) -> float:
        length = 0
        total_vertices = self.vertices + self.si_vertices
        for i in range(len(self.adj_list)):
            for j in range(len(self.adj_list[i])):
                length += total_vertices[i] - total_vertices[self.adj_list[i][j]]
        return length / 2
