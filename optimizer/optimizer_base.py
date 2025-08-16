from __future__ import annotations
from typing import Callable, Union, Optional, Any
import torch

from .scheduler import make_scheduler
from .early_stopper import EarlyStopper

class OptimizerBase:
    """Steiner 정점만 업데이트하는 공통 최적화 루프."""

    def __init__(self, vertices: torch.Tensor,
        edge_index: torch.Tensor, steiner_mask: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_factory: Callable[[torch.nn.Parameter], torch.optim.Optimizer],  # 생성자 콜백
        max_iter: int = 10000, tolerance: float = 1e-6,
        scheduler_name: str | None = None,
        scheduler_params: dict | None = None
    ) -> None:
        # 입력 검증
        assert vertices.device == edge_index.device == steiner_mask.device, "모든 텐서는 동일 디바이스에 있어야 합니다."
        assert vertices.ndim == 2, "vertices는 (N, d) 형태여야 합니다."
        assert edge_index.ndim == 2, "edge_index는 (E, 2) 형태여야 합니다."
        assert steiner_mask.ndim == 1, "steiner_mask는 (N,) 형태여야 합니다."
        assert steiner_mask.dtype == torch.bool, "steiner_mask는 bool 타입이어야 합니다."
        assert vertices.shape[0] == steiner_mask.shape[0], "vertices와 steiner_mask의 첫 번째 차원은 같아야 합니다."
        assert edge_index.shape[1] == 2, "edge_index는 (E, 2) 형태여야 합니다."
        assert callable(objective), "objective는 호출 가능한 함수여야 합니다."
        assert callable(optimizer_factory), "optimizer_factory는 호출 가능한 함수여야 합니다."
        assert max_iter > 0, "max_iter는 양수여야 합니다."
        assert tolerance > 0, "tolerance는 양수여야 합니다."

        self.edge_index = edge_index # (2, E)
        self.objective = objective # 손실 함수 (vertices, get_edge_index) -> loss

        #학습-고정 정점 분리
        self.vertices = vertices  # 원본 vertices 저장
        self.steiner_mask = steiner_mask  # 스타이너 마스크 저장

        # 서브클래스에서 오버라이드할 수 있도록 _create_train_param 호출
        self.train_param = self._create_train_param()  # (K, d) 또는 ManifoldParameter
        self.fixed_buffer = vertices[~steiner_mask].clone().detach()  # (N-K, d)
        self.fixed_idx = (~steiner_mask).nonzero(as_tuple=False).squeeze()
        self.train_idx = steiner_mask.nonzero(as_tuple=False).squeeze()

        #optimizer 초기화
        self.optimizer = optimizer_factory(self.train_param)

        scheduler, needs_loss = make_scheduler(
            scheduler_name, self.optimizer,
            total_steps=max_iter, params=scheduler_params
        )
        self.scheduler = scheduler
        self._sched_needs_loss = needs_loss

        self.max_iter = max_iter
        self.tolerance = tolerance

        # 외부에서 주입될 수 있는 EarlyStopper (옵션)
        self.early_stopper: Optional[EarlyStopper] = None
        self.last_early_stop_reason = None

        # loss 값중 최소값
        self.min_loss = float("inf")

    def _create_train_param(self) -> torch.nn.Parameter:
        """학습 가능한 파라미터 생성. 서브클래스에서 오버라이드 가능"""
        return torch.nn.Parameter(self.vertices[self.steiner_mask].clone())

    @property
    def dtype(self) -> torch.dtype:
        return self.train_param.dtype

    @property
    def device(self) -> Union[str, torch.device]:
        return self.train_param.device

    def _assemble_full(self) -> torch.Tensor:
        """고정·학습 정점을 (N,d) 하나로 결합하여 반환. grad가 train_param에 전달되도록 scatter 사용."""
        full = torch.empty(
            self.fixed_buffer.shape[0] + self.train_param.shape[0],
            self.train_param.shape[1],
            device=self.device,
            dtype=self.dtype,
        )

        #grad가 train_param에 전달되도록 index_copy_ 사용
        full = full.index_copy_(0, self.fixed_idx, self.fixed_buffer)
        full = full.index_copy_(0, self.train_idx, self.train_param)
        return full

    def run(self) -> tuple[torch.Tensor, list[float]]:
        """최적화를 수행하고 최종 loss(detach) 를 반환한다."""
        loss_history = []

        # early stopper 시작
        if self.early_stopper is not None:
            self.early_stopper.begin()

        for step in range(self.max_iter):
            self.optimizer.zero_grad()
            loss = self.objective(self._assemble_full(), self.edge_index)
            loss_val = loss.detach().item()
            loss_history.append(loss_val)
            loss.backward()

            grad = self.train_param.grad
            # min_iter 이전에는 grad-기반 종료를 막아 워밍업 보장
            if grad is not None and grad.norm() < self.tolerance:
                if (self.early_stopper is None) or (step >= self.early_stopper.cfg.min_iter):
                    # 종료 전 최선 파라미터 복원
                    if self.early_stopper is not None:
                        self.early_stopper.restore_if_needed(self.train_param)
                    self.last_early_stop_reason = "grad_tol"
                    break

            # 업데이트 크기 측정용 스냅샷
            if self.early_stopper is not None:
                self.early_stopper.on_pre_step(self.train_param)

            self.optimizer.step()
            self.post_step()

            # 스케줄러 스텝
            if self._sched_needs_loss:
                self.scheduler.step(loss_val)
            else:
                self.scheduler.step()

            # 조기 종료 판단
            if self.early_stopper is not None:
                cur_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                stop, reason = self.early_stopper.on_post_step(step, loss_val, self.train_param, cur_lrs)
                if stop:
                    self.early_stopper.restore_if_needed(self.train_param)
                    self.last_early_stop_reason = reason
                    break

        if self.last_early_stop_reason is None:
            self.last_early_stop_reason = "max_iter"

        self.min_loss = min(loss_history)
        return loss.detach(), loss_history

    def post_step(self) -> None:
        """각 최적화 스텝 후 호출되는 후처리 메소드. 기본 구현은 없음."""
        pass


    def updated_full(self) -> torch.Tensor:
        """최적화된 전체 정점을 detach 상태로 반환."""
        return self._assemble_full().detach()
