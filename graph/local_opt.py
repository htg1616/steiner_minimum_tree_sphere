import torch

from optimizer.geo_optimizer import GeoOptimizer
from optimizer.torch_optimizer import TorchOptimizer


def make_local_optimizer(
    backend: str,
    steiner_tree,
    optim_name: str,
    hyper_param: dict | None = None,
    max_iter: int = 10000,
    tolerance: float = 1e-6,
    device: str | torch.device = "cpu",
):
    """
    로컬 최적화기를 생성하는 팩토리 함수

    Args:
        backend: 최적화 백엔드 ('torch', 'geo')
        steiner_tree: SteinerTree 같은 그래프 객체
        optim_name: optimizer 이름 ('adam', 'sgd' 등)
        hyper_param: optimizer 하이퍼파라미터 딕셔너리
        max_iter: 최대 반복 횟수
        tolerance: 수렴 판단 기준
        device: 연산 장치 ('cpu', 'cuda' 등)

    Returns:
         optimizer 객체 (TorchOptimizer 또는 GeoOptimizer)
    """
    edges = steiner_tree.get_edge_index(device)
    mask = steiner_tree.get_steiner_mask(device)

    if backend.lower() == "torch":
        verts = steiner_tree.get_vertices_angle_tensor(device)
        return TorchOptimizer(
            vertices=verts,
            edge_index=edges,
            steiner_mask=mask,
            optim_name=optim_name,
            hyper_param=hyper_param,
            max_iter=max_iter,
            tolerance=tolerance
        )
    elif backend.lower() == "geo":
        verts = steiner_tree.get_vertices_xyz_tensor(device)
        return GeoOptimizer(
            vertices=verts,
            edge_index=edges,
            steiner_mask=mask,
            optim_name=optim_name,
            hyper_param=hyper_param,
            max_iter=max_iter,
            tolerance=tolerance
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")