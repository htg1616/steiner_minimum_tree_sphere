import pytest
from math import pi as PI

from geometry.dot import Dot
from graph.enums import InsertionMode
from graph.mst import MinimalSpanningTree
from graph.steiner import SteinerTree
from graph.local_opt import make_local_optimizer


@pytest.fixture
def simple_dots():
    dots = [Dot(PI/6, 0), Dot(PI/6, 2*PI/3), Dot(0, 0)]
    return dots

@pytest.mark.parametrize("backend, optim_name",
                         [("torch", "adam"),
                          ("geo", "radam")])
def test_make_local_optimizer_factory(simple_dots, backend, optim_name):
    """make_local_optimizer 팩토리 함수 테스트"""
    # 간단한 테스트 케이스 생성
    mst = MinimalSpanningTree(simple_dots)
    smt = SteinerTree(mst, InsertionMode.DECREASE_ONLY)

    if smt.steiner_count == 0:
        pytest.skip("No Steiner points to optimize")

    # optimizer 생성 테스트
    optimizer = make_local_optimizer(
        backend=backend,
        steiner_tree=smt,
        optim_name=optim_name,
        hyper_param={"lr": 0.01},
        max_iter=100,
        tolerance=1e-6,
        device="cpu"
    )

    assert optimizer is not None
    assert hasattr(optimizer, 'run')
    assert hasattr(optimizer, 'updated_full')


def test_make_local_optimizer_invalid_backend(simple_dots):
    """잘못된 백엔드에 대한 오류 처리 테스트"""
    mst = MinimalSpanningTree(simple_dots)
    smt = SteinerTree(mst, InsertionMode.DECREASE_ONLY)

    with pytest.raises(ValueError, match="Unknown backend"):
        make_local_optimizer(
            backend="invalid_backend",
            steiner_tree=smt,
            optim_name="adam",
            max_iter=100
        )

@pytest.mark.parametrize("backend, optim_name, hyper_param",
                        [("torch", "adam", {"lr": 0.01, "betas": (0.9, 0.999)}),
                        ("torch", "sgd", {"lr": 0.01, "momentum": 0.9}),
                        ("geo", "radam", {"lr": 0.01, "eps": 1e-4}),
                        ("geo", "rsgd", {"lr": 0.01, "momentum": 0.9})])
def test_local_optimizer(simple_dots, backend, optim_name, hyper_param):
    """다양한 optimizer 종류 생성 및 하이퍼 파라미터 전달 테스트"""
    mst = MinimalSpanningTree(simple_dots)
    smt = SteinerTree(mst, InsertionMode.DECREASE_ONLY)

    if smt.steiner_count == 0:
        pytest.skip("No Steiner points to optimize")

    # optimizer 테스트
    optimizer = make_local_optimizer(
        backend="torch",
        steiner_tree=smt,
        optim_name="adam",
        hyper_param={"lr": 0.01},
        max_iter=10,
        tolerance=1e-6,
        device="cpu"
    )

    assert optimizer is not None

    for key, expected_value in hyper_param.items():
        assert optimizer.param_groups[0][key] == expected_value, f"Expected {key} to be {expected_value}, got {optimizer.param_groups[0][key]}"

