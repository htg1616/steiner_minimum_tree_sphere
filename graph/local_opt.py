import torch

from .steiner import SteinerTree
from optimizer.geo_optimizer import GeoOptimizer
from optimizer.torch_optimizer import TorchOptimizer


def make_local_optimizer(backend: str, steiner_tree: SteinerTree, optim_name: str, optim_param: dict | None = None,
                         scheduler_name: str = None, scheduler_param: dict = None,
                         max_iter: int = 10000, tolerance: float = 1e-6,
                         device: str | torch.device = "cpu",
                         early_stopper=None):
    """
    로컬 최적화기를 생성하는 팩토리 함수

    Args:
        backend: 최적화 백엔드 ('torch', 'geo')
        steiner_tree: SteinerTree 같은 그래프 객체
        optim_name: optimizer 이름 ('adam', 'sgd' 등)
        optim_param: optimizer 하이퍼파라미터 딕셔너리
        scheduler_name: 스케줄러 이름
        scheduler_param: 스케줄러 하이퍼파라미터 딕셔너리
        max_iter: 최대 반복 횟수
        tolerance: 수렴 판단 기준
        device: 연산 장치 ('cpu', 'cuda' 등)
        early_stopper: EarlyStopper 인스턴스 (옵션)

    Returns:
         optimizer 객체 (TorchOptimizer 또는 GeoOptimizer)
    """
    edges = steiner_tree.get_edge_index(device)
    mask = steiner_tree.get_steiner_mask(device)

    if backend.lower() == "torch":
        verts = steiner_tree.get_vertices_angle_tensor(device)
        opt = TorchOptimizer(
            vertices=verts,
            edge_index=edges,
            steiner_mask=mask,
            optim_name=optim_name,
            hyper_param=optim_param,
            max_iter=max_iter,
            tolerance=tolerance,
            scheduler_name=scheduler_name,
            scheduler_param=scheduler_param
        )
        if early_stopper is not None:
            opt.early_stopper = early_stopper
        return opt
    elif backend.lower() == "geo":
        verts = steiner_tree.get_vertices_xyz_tensor(device)
        opt = GeoOptimizer(
            vertices=verts,
            edge_index=edges,
            steiner_mask=mask,
            optim_name=optim_name,
            hyper_param=optim_param,
            max_iter=max_iter,
            tolerance=tolerance,
            scheduler_name=scheduler_name,
            scheduler_param=scheduler_param
        )
        if early_stopper is not None:
            opt.early_stopper = early_stopper
        return opt

    else:
        raise ValueError(f"Unknown backend: {backend}")