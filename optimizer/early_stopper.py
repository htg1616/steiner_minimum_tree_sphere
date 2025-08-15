from __future__ import annotations
import math, time
from dataclasses import dataclass, fields
from collections import deque
from typing import Optional, Tuple, List
import torch


@dataclass
class EarlyStopConfig:
    # 공통
    min_iter: int = 0
    time_budget_sec: Optional[float] = None

    # 개선 정체(plateau)
    early_patience: int = 0          # 0이면 비활성
    early_rel_tol: float = 1e-4
    early_abs_tol: float = 0.0
    restore_best: bool = True        # 종료 시 최선 파라미터 복원

    # 업데이트 크기 미소
    update_tol: Optional[float] = None  # None이면 비활성 (예: 1e-6)
    update_patience: int = 5

    # LR 바닥 지속
    lr_floor: Optional[float] = None    # None이면 비활성 (예: max(lr0/5000, 1e-7))
    lr_patience: int = 100

    # 안정 창(flat window)
    stable_window: int = 0              # 0이면 비활성 (예: 20)
    stable_eps: float = 1e-6

    # 발산/이상치 가드
    div_factor: float = 10.0            # best 대비 div_factor배 급등 시 중단

    @classmethod
    def from_dict(cls, d: dict) -> "EarlyStopConfig":
        allow = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in allow})


class EarlyStopper:
    def __init__(self, cfg: EarlyStopConfig):
        self.cfg = cfg
        self.start_time: Optional[float] = None
        self.best: float = float("inf")
        self.since_best: int = 0
        self.best_state: Optional[torch.Tensor] = None
        self.update_since: int = 0
        self.lr_since: int = 0
        self.win = deque(maxlen=cfg.stable_window) if cfg.stable_window > 0 else None
        self._pre_snapshot: Optional[torch.Tensor] = None

    # 루프 시작 시 1회
    def begin(self) -> None:
        self.start_time = time.time()

    # optimizer.step() 직전 호출(업데이트 크기 측정용)
    def on_pre_step(self, params: torch.Tensor) -> None:
        if self.cfg.update_tol is not None and params.numel() > 0:
            self._pre_snapshot = params.detach().clone()
        else:
            self._pre_snapshot = None

    # 한 스텝 종료 시 호출: stop 여부/사유 반환
    def on_post_step(
        self, step: int, loss_val: float, params: torch.Tensor, lrs: List[float]
    ) -> Tuple[bool, Optional[str]]:
        c = self.cfg

        # 시간 예산
        if c.time_budget_sec is not None and self.start_time is not None:
            if (time.time() - self.start_time) >= c.time_budget_sec and step >= c.min_iter:
                return True, "time_budget"

        # NaN/Inf/발산 가드
        if not math.isfinite(loss_val) or (math.isfinite(self.best) and loss_val > self.best * c.div_factor):
            return True, "divergence"

        # plateau 추적(최선 갱신/정체 카운트)
        if c.early_patience > 0:
            need = 0.0 if not math.isfinite(self.best) else max(c.early_abs_tol, abs(self.best) * c.early_rel_tol)
            if loss_val < self.best - need:
                self.best, self.since_best = loss_val, 0
                if c.restore_best:
                    self.best_state = params.detach().clone()
            else:
                self.since_best += 1

        # 안정 창(flat window)
        if self.win is not None:
            self.win.append(loss_val)
            if len(self.win) == self.win.maxlen and (max(self.win) - min(self.win)) <= c.stable_eps and step >= c.min_iter:
                return True, "flat_window"

        # 작은 업데이트(Δθ)
        if self._pre_snapshot is not None:
            with torch.no_grad():
                delta = (params - self._pre_snapshot).norm().item()
                base = max(1.0, self._pre_snapshot.norm().item())
            if delta <= c.update_tol * base:
                self.update_since += 1
                if self.update_since >= c.update_patience and step >= c.min_iter:
                    return True, "small_updates"
            else:
                self.update_since = 0

        # LR 바닥 지속
        if c.lr_floor is not None and c.lr_patience > 0:
            cur_lr = min(lrs) if isinstance(lrs, (list, tuple)) else float(lrs)
            if cur_lr <= c.lr_floor:
                self.lr_since += 1
                if self.lr_since >= c.lr_patience and step >= c.min_iter:
                    return True, "lr_floor_stall"
            else:
                self.lr_since = 0

        # plateau 종료
        if c.early_patience > 0 and self.since_best >= c.early_patience and step >= c.min_iter:
            return True, "plateau"

        return False, None

    # 종료 시 최선 파라미터 복원(옵션)
    def restore_if_needed(self, params: torch.Tensor) -> None:
        if self.cfg.restore_best and (self.best_state is not None):
            with torch.no_grad():
                params.copy_(self.best_state)
