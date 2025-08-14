import math
import pytest
import torch

from optimizer.scheduler import make_scheduler, NoOpScheduler

def make_sgd(lr=0.1):
    p = torch.nn.Parameter(torch.randn(2, 2))
    return torch.optim.SGD([p], lr=lr)

# ---------- Factory & Error handling ----------
def test_noop_scheduler_basic():
    opt = make_sgd(0.1)
    sch, needs_loss = make_scheduler(None, opt, total_steps=10)
    assert isinstance(sch, NoOpScheduler)
    assert needs_loss is False
    before = sch.get_last_lr()[0]
    sch.step()  # no crash
    after = sch.get_last_lr()[0]
    assert before == after

@pytest.mark.parametrize("name", ["cosine", "onecycle"])
def test_missing_total_steps_raises(name):
    opt = make_sgd(0.1)
    with pytest.raises(ValueError):
        make_scheduler(name, opt)  # total_steps 없음

def test_invalid_name_raises():
    opt = make_sgd(0.1)
    with pytest.raises(ValueError):
        make_scheduler("invalid", opt, total_steps=5)

# ---------- CosineAnnealingLR ----------
def test_cosine_basic_decreases_to_eta_min():
    opt = make_sgd(0.2)
    eta_min = 0.0
    T = 5
    sch, needs_loss = make_scheduler("cosine", opt, total_steps=T, params={"eta_min": eta_min})
    assert needs_loss is False

    lrs = []
    for _ in range(T):
        sch.step()
        lrs.append(sch.get_last_lr()[0])

    assert lrs[-1] == pytest.approx(eta_min, abs=1e-6)
    assert lrs[0] >= lrs[-1]

# ---------- OneCycleLR ----------
def test_onecycle_default_max_lr_and_shape():
    base_lr = 0.1
    opt = make_sgd(base_lr)
    T = 8
    sch, needs_loss = make_scheduler("onecycle", opt, total_steps=T)  # max_lr 기본: 현재 lr
    assert needs_loss is False

    lrs = []
    for _ in range(T):
        sch.step()
        lrs.append(sch.get_last_lr()[0])

    # 모양만 확인: (some increase) then (decrease)
    assert max(lrs) > min(lrs)
    peak_idx = max(range(len(lrs)), key=lambda i: lrs[i])
    assert peak_idx > 0 and peak_idx < T-1
    assert lrs[-1] < lrs[peak_idx]

# ---------- ReduceLROnPlateau ----------
def test_plateau_reduces_when_no_improvement():
    opt = make_sgd(0.1)
    sch, needs_loss = make_scheduler(
        "plateau",
        opt,
        params={"patience": 1, "factor": 0.5, "threshold": 0.0, "min_lr": 0.0}
    )
    assert needs_loss is True

    def lr(): return sch.optimizer.param_groups[0]["lr"]

    # 첫 call: best=1.0
    sch.step(1.0)
    lr1 = lr()

    # 개선 없음 1회: 아직 감소 X (보통 patience를 채우는 중)
    sch.step(1.0)
    lr2 = lr()

    # 또 개선 없음: 이제 감소 기대
    sch.step(1.0)
    lr3 = lr()

    assert lr2 == pytest.approx(lr1)
    assert lr3 < lr2  # factor 적용되어 감소

# ---------- Cosine-Hold (CosineAnnealing -> Hold) ----------
def test_cosine_hold_simple():
    base_lr = 0.1
    opt = make_sgd(base_lr)
    decay_steps = 5
    eta_min = 1e-3

    sch, needs_loss = make_scheduler(
        "cosine_hold", opt, total_steps=20,
        params={"decay_steps": decay_steps, "eta_min": eta_min}
    )
    assert needs_loss is False

    lrs = []
    for _ in range(decay_steps + 2):  # 경계 바로 뒤까지 관찰
        sch.step()
        lrs.append(sch.get_last_lr()[0])

    # decay 구간 끝에서 eta_min 도달하고, 그 이후에도 그대로 유지
    assert lrs[decay_steps - 1] == pytest.approx(eta_min, abs=1e-9)
    assert lrs[decay_steps]     == pytest.approx(eta_min, abs=1e-9)

