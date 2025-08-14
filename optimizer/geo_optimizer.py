import torch
import geoopt
from typing import Callable

from optimizer.optimizer_base import OptimizerBase

def length_xyzs(xyz: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    3D 단위벡터를 사용하여 구면상의 대원 거리를 계산합니다.

    Args:
        xyz: (N, 3) 단위벡터 텐서
        edge_index: (E, 2) long tensor, 0-기반 정점 인덱스
        eps: 수치적 안정성을 위한 클램프 값 (기본값 1e-7)

    Returns:
        스칼라 - 모든 간선을 따라 대원 거리(라디안)의 합
    """
    # 간선 끝점 벡터 수집
    u = xyz[edge_index[:, 0]]  # (E, 3)
    v = xyz[edge_index[:, 1]]  # (E, 3)

    # 수치적 안정성 위해 arccos 대신 다른 공식 사용
    a = torch.linalg.norm(u - v, dim=-1)
    b = torch.linalg.norm(u + v, dim=-1)
    distances = 2.0 * torch.atan2(a, b)
    #더 정밀한 수치적 안정성 필요시 보상합 구현 필요

    return distances.sum()

def make_geoopt_factory(name: str, hyper_param: dict) -> Callable[[torch.nn.Parameter], torch.optim.Optimizer]:
    """geoopt 옵티마이저 팩토리 함수"""
    name = name.lower()

    if name == "radam":
        return lambda p: geoopt.optim.RiemannianAdam([p], **hyper_param)
    elif name == "rsgd":
        return lambda p: geoopt.optim.RiemannianSGD([p], **hyper_param)
    else:
        raise ValueError(f"지원하지 않는 geoopt optimizer: {name}. 지원되는 optimizer는 'radam', 'rsgd'입니다.")

class GeoOptimizer(OptimizerBase):
    """구면 스타이너 트리를 위한 3D 벡터 표현 지역최적화기 (geoopt 사용)"""

    def __init__(self, vertices: torch.Tensor,
                 edge_index: torch.Tensor,
                 steiner_mask: torch.Tensor,
                 optim_name: str,
                 hyper_param: dict | None = None,
                 max_iter: int = 10000,
                 tolerance: float = 1e-6,
                 scheduler_name: str | None = None,
                 scheduler_param: dict | None = None):
        """
        Args:
            vertices: 3D 단위벡터 (N, 3) 텐서
            edge_index: 간선 인덱스 (2, E) 형태의 텐서
            steiner_mask: 스타이너 점 마스크 (N,) 형태의 불리언 텐서
            optim_name: geoopt optimizer 이름 ('radam', 'rsgd')
            hyper_param: optimizer 하이퍼파라미터 딕셔너리
            max_iter: 최대 반복 횟수
            tolerance: 수렴 판단 기준
            scheduler_name: 스케줄러 이름
            scheduler_param: 스케줄러 하이퍼파라미터 딕셔너리
        """
        # 입력이 3D 벡터인지 확인
        if vertices.shape[1] != 3:
            raise ValueError(f"vertices는 (N, 3) 3D 벡터여야 합니다. 현재 shape: {vertices.shape}")

        # geoopt factory 생성
        optimizer_factory = make_geoopt_factory(optim_name, hyper_param if hyper_param is not None else {})

        # 기본 클래스 초기화
        super().__init__(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            objective=length_xyzs,
            optimizer_factory=optimizer_factory,
            max_iter=max_iter,
            tolerance=tolerance,
            scheduler_name=scheduler_name,
            scheduler_params=scheduler_param
        )

    def _create_train_param(self) -> geoopt.ManifoldParameter:
        """구면 매니폴드 파라미터 생성 및 optimizer 설정"""
        sphere = geoopt.Sphere()
        steiner_vertices = self.vertices[self.steiner_mask].clone()

        # 입력이 이미 단위벡터라고 가정하되, 정규화로 안전성 확보
        steiner_vertices = steiner_vertices / steiner_vertices.norm(dim=-1, keepdim=True)

        manifold_param = geoopt.ManifoldParameter(steiner_vertices, manifold=sphere, requires_grad=True)

        return manifold_param
