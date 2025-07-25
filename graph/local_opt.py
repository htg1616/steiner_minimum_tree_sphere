import copy
import math

import numpy as np

from geometry.dot import Dot
from .graph_base import GraphBase
from .steiner import SteinerTree


class LocalOptimizedGraph(GraphBase):
    def __init__(self, steiner_tree: SteinerTree):
        # 부모 클래스 초기화
        vertices = copy.deepcopy(steiner_tree.fixed_vertices) + copy.deepcopy(steiner_tree.steiner_vertices)
        super().__init__(vertices, steiner_tree.adj)

        # 원본 정점과 스타이너 정점 분리
        self.original_count = steiner_tree.fixed_count
        self.original_vertices = copy.deepcopy(steiner_tree.fixed_vertices)
        self.steiner_vertices = copy.deepcopy(steiner_tree.steiner_vertices)

        #최적화된 점 위치 임시 보관용
        self.optimized_steiner_vertices = copy.deepcopy(self.steiner_vertices)

    @staticmethod
    def great_circle_gradient(point_v: Dot, point_u: Dot) -> tuple[float, float]:
        """대원 거리의 그래디언트 계산"""
        epsilon = 1e-8
        theta1, phi1 = point_v.theta, point_v.phi
        theta2, phi2 = point_u.theta, point_u.phi

        # theta에 대한 그래디언트 계산
        theta_numerator = -(-math.sin(theta1) * math.cos(theta2) +
                            math.cos(theta1) * math.sin(theta2) * math.cos(phi2 - phi1))
        theta_denominator = math.sqrt(1 - (math.cos(theta1) * math.cos(theta2) +
                                           math.sin(theta1) * math.sin(theta2) * math.cos(phi2 - phi1)) ** 2)
        if abs(theta_denominator) < epsilon:
            theta_denominator = epsilon if theta_denominator >= 0 else -epsilon
        grad_theta = theta_numerator / theta_denominator

        # phi에 대한 그래디언트 계산
        phi_numerator = -(math.sin(theta1) * math.sin(theta2) * math.sin(phi2 - phi1))
        phi_denominator = math.sqrt(1 - (math.cos(theta1) * math.cos(theta2) +
                                         math.sin(theta1) * math.sin(theta2) * math.cos(phi2 - phi1)) ** 2)
        if abs(phi_denominator) < epsilon:
            phi_denominator = epsilon if phi_denominator >= 0 else -epsilon
        grad_phi = phi_numerator / phi_denominator

        return grad_theta, grad_phi

    def compute_gradient(self):
        """모든 스타이너 정점의 그래디언트 계산"""
        n = self.original_count
        m = len(self.optimized_steiner_vertices)
        grad_theta = np.zeros(m)
        grad_phi = np.zeros(m)

        # 각 스타이너 정점에 대해 그래디언트 계산
        for v_idx, point_v in enumerate(self.optimized_steiner_vertices):
            steiner_idx = n + v_idx

            for u in self.adj[steiner_idx]:
                if u < n:
                    point_u = self.vertices[u]
                else:
                    point_u = self.optimized_steiner_vertices[u - n]

                grad_theta_v, grad_phi_v = self.great_circle_gradient(point_v, point_u)
                grad_theta[v_idx] += grad_theta_v
                grad_phi[v_idx] += grad_phi_v

        return grad_theta, grad_phi

    def optimize(self, num_iterations=1000) -> list[float]:
        """Adam 최적화 알고리즘으로 스타이너 정점 위치 최적화"""
        learning_rate = 0.001
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-3

        # 수치적 안정성을 위해 정점 조정
        for si_point in self.optimized_steiner_vertices:
            si_point.theta += epsilon
            si_point.phi += epsilon

        si_theta = np.array([p.theta for p in self.optimized_steiner_vertices])
        si_phi = np.array([p.phi for p in self.optimized_steiner_vertices])

        # Adam 최적화 변수 초기화
        m_theta, m_phi = np.zeros_like(si_theta), np.zeros_like(si_phi)
        v_theta, v_phi = np.zeros_like(si_theta), np.zeros_like(si_phi)
        results = [self.length()]

        for t in range(1, num_iterations + 1):
            # 그래디언트 계산 및 Adam 업데이트
            grad_theta, grad_phi = self.compute_gradient()

            m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
            m_phi = beta1 * m_phi + (1 - beta1) * grad_phi

            v_theta = beta2 * v_theta + (1 - beta2) * (grad_theta ** 2)
            v_phi = beta2 * v_phi + (1 - beta2) * (grad_phi ** 2)

            m_hat_theta = m_theta / (1 - beta1 ** t)
            m_hat_phi = m_phi / (1 - beta1 ** t)
            v_hat_theta = v_theta / (1 - beta2 ** t)
            v_hat_phi = v_phi / (1 - beta2 ** t)

            # 정점 위치 업데이트
            si_theta -= learning_rate * m_hat_theta / (np.sqrt(v_hat_theta) + epsilon)
            si_phi -= learning_rate * m_hat_phi / (np.sqrt(v_hat_phi) + epsilon)

            # 정점 목록 업데이트
            n = self.original_count
            self.optimized_steiner_vertices = [Dot(theta, phi) for theta, phi in zip(si_theta, si_phi)]
            for i, v in enumerate(self.optimized_steiner_vertices):
                self.vertices[n + i] = v

            results.append(self.length())

        return results

