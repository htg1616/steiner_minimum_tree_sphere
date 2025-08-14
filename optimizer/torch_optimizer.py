import torch
from typing import Callable

from optimizer.optimizer_base import OptimizerBase

def length_angles(theta_phi: torch.Tensor,
                  edge_index: torch.Tensor,
                  eps: float = 1e-12) -> torch.Tensor:
    """
    구면 좌표계 (theta, phi)로 주어진 트리의 전체 대원 거리를 계산합니다.
        theta: [0, pi] 범위의 colatitude
        phi  : [0, 2pi) 범위의 longitude

    Args:
        theta_phi : (N, 2) tensor, [theta, phi] 라디안 형태
        edge_index: (E, 2) long tensor, 0-기반 정점 인덱스
        eps       : 수치적 안정성을 위한 클램프 값 (기본값 1e-12)

    Returns:
        스칼라 - 모든 간선을 따라 대원 거리(라디안)의 합
    """
    # 간선 끝점 좌표 수집 - shape (E,)
    theta_i, phi_i = theta_phi[edge_index[:, 0]].T  # 간선 시작점의 여위도
    theta_j, phi_j = theta_phi[edge_index[:, 1]].T  # 간선 끝점의 여위도

    # 각도 차이
    dtheta = theta_j - theta_i
    dphi   = phi_j - phi_i

    # 하버사인 구성 요소 (여위도 사용)
    sin_half_dtheta = torch.sin(dtheta * 0.5)
    sin_half_dphi   = torch.sin(dphi   * 0.5)

    # 참고: 위도 = pi/2 − theta  ⇒  cos(위도) = sin(theta)
    sin_theta_i = torch.sin(theta_i)
    sin_theta_j = torch.sin(theta_j)

    hav = sin_half_dtheta.pow(2) + sin_theta_i * sin_theta_j * sin_half_dphi.pow(2)
    hav = torch.clamp(hav, eps, 1.0 - eps)  # asin 인수를 (0,1) 범위로 유지

    # 대원 거리 δ = 2·asin(√hav)
    dist = 2.0 * torch.asin(torch.sqrt(hav))

    return dist.sum()

def make_torch_factory(name: str, hyper_param: dict) -> Callable[[torch.nn.Parameter], torch.optim.Optimizer]:
    name = name.lower()
    if name == "adam":
        return lambda p: torch.optim.Adam([p], **hyper_param)
    if name == "sgd":
        return lambda p: torch.optim.SGD([p], **hyper_param)
    raise ValueError(f"지원하지 않는 optimizer: {name}. 지원되는 optimizer는 'adam', 'sgd'입니다.")

class TorchOptimizer(OptimizerBase):
    """구면 스타이너 트리를 위한 각도 표현(theta, phi) 지역최적화기"""

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
            vertices: 각도 텐서 (theta, phi) 형태의 (N, 2) 텐서
            edge_index: 간선 인덱스 (2, E) 형태의 텐서
            steiner_mask: 스타이너 점 마스크 (N,) 형태의 불리언 텐서
            optim_name: optimizer 이름 ('adam', 'sgd' 등)
            hyper_param: optimizer 하이퍼파라미터 딕셔너리
            max_iter: 최대 반복 횟수
            tolerance: 수렴 판단 기준
            scheduler_name: 스케줄러 이름
            scheduler_param: 스케줄러 하이퍼파라미터 딕셔너리
        """

        optimizer_factory = make_torch_factory(optim_name, hyper_param if hyper_param is not None else {})

        # 기본 클래스 초기화
        super().__init__(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            objective=length_angles,
            optimizer_factory=optimizer_factory,
            max_iter=max_iter,
            tolerance=tolerance,
            scheduler_name=scheduler_name,
            scheduler_params=scheduler_param
        )

    def post_step(self):
        """각 최적화 스텝 후 호출되는 후처리 메소드.
        theta, phi 좌표계 제약 조건 적용: theta ∈ [0, pi], phi ∈ [0, 2*pi)
        theta가 [0, pi] 범위를 벗어날 때는 반사 변환을 적용하여 올바른 구면 좌표로 변환
        """
        with torch.no_grad():
            data = self.train_param.data  # (M, 2) tensor
            theta = data[:, 0]
            phi = data[:, 1]

            # --- theta를 [0, pi] 범위로 반사 변환 ---------------------------
            over_mask = theta > torch.pi
            under_mask = theta < 0

            # theta > pi -> (2pi − theta, phi + pi)
            theta[over_mask] = 2 * torch.pi - theta[over_mask]
            phi[over_mask] += torch.pi

            # theta < 0 -> (−theta, phi + pi)
            theta[under_mask] = -theta[under_mask]
            phi[under_mask] += torch.pi

            # --- phi를 [0, 2pi) 범위로 mod 변환 ---------------------------------
            phi.remainder_(2 * torch.pi)
