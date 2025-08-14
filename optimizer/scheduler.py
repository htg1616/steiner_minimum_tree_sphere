import torch
import math
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

class NoOpScheduler:
    """아무 동작하지 않는 기본 스케줄러"""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, loss=None):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def constant_eta(optimizer, eta):
    base_lrs = [pg.get('initial_lr', pg['lr']) for pg in optimizer.param_groups]
    lambdas = [lambda e, c=float(eta/base): c for base in base_lrs]
    return LambdaLR(optimizer, lr_lambda=lambdas)


def make_scheduler(
    name: str | None,
    optimizer, *,
    total_steps: int | None = None,
    params: dict | None = None,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | NoOpScheduler, bool]:
    """
    returns: (scheduler, needs_loss)
    needs_loss=True → ReduceLROnPlateau 같이 loss 인자를 요구
    """
    if name is None:
        return NoOpScheduler(optimizer), False

    name = name.lower()
    params = params or {}

    if name == "cosine":
        if total_steps is None:
            raise ValueError("CosineAnnealingLR requires total_steps")
        scheduler_params = {
            "T_max": total_steps,
            "eta_min": params.get("eta_min", 0),
            **{k: v for k, v in params.items() if k != "eta_min"}
        }
        return CosineAnnealingLR(optimizer, **scheduler_params), False


    elif name == "cosine_hold":
        decay_steps = params.get("decay_steps")

        if decay_steps is None:
            if total_steps is not None:
                # total_steps가 있으면 100 또는 total_steps 중 작은 값으로 기본 설정
                decay_steps = min(100, total_steps)
            else:
                raise ValueError("cosine_hold requires params['decay_steps'] or total_steps")
        decay_steps = int(decay_steps)
        if decay_steps <= 0:
            raise ValueError("decay_steps must be > 0")

        #eta_min 기본값: 가장 작은 base_lr의 1/100 (하한 1e-6)
        base_lrs = [pg.get("initial_lr", pg["lr"]) for pg in optimizer.param_groups]
        default_eta = max(min(base_lrs) / 100.0, 1e-6)
        eta_min = float(params.get("eta_min", default_eta))

        # 스케줄러 구성: CosineAnnealingLR → (milestone에서) LambdaLR(상수 유지)
        cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
        hold = constant_eta(optimizer, eta_min)
        # milestones 길이는 schedulers 길이 - 1 이어야 함
        scheduler = SequentialLR(optimizer, schedulers=[cosine, hold], milestones=[decay_steps])

        return scheduler, False

    elif name == "onecycle":
        if total_steps is None:
            raise ValueError("OneCycleLR requires total_steps")

        # max_lr 기본값 설정
        if "max_lr" not in params:
            # optimizer의 현재 lr을 max_lr로 사용
            current_lr = optimizer.param_groups[0]['lr']
            params = {"max_lr": current_lr, **params}

        scheduler_params = {
            "total_steps": total_steps,
            "pct_start": params.get("pct_start", 0.3),
            "final_div_factor": params.get("final_div_factor", 1e4),
            **{k: v for k, v in params.items() if k not in ["pct_start", "final_div_factor"]}
        }
        return OneCycleLR(optimizer, **scheduler_params), False

    elif name == "plateau":
        scheduler_params = {
            "mode": params.get("mode", "min"),
            "patience": params.get("patience", 10),
            "threshold": params.get("threshold", 1e-4),
            "factor": params.get("factor", 0.5),
            "min_lr": params.get("min_lr", 0),
            **{k: v for k, v in params.items()
               if k not in ["mode", "patience", "threshold", "factor", "min_lr"]}
        }
        return ReduceLROnPlateau(optimizer, **scheduler_params), True

    else:
        raise ValueError(f"지원하지 않는 scheduler: {name}. 지원되는 scheduler는 'cosine', 'onecycle', 'plateau'입니다.")
