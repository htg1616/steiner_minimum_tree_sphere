import pytest
import random
from math import pi as PI

from geometry.dot import Dot
from graph.enums import InsertionMode
from graph.mst import MinimalSpanningTree
from graph.steiner import SteinerTree
from graph.local_opt import make_local_optimizer

# demo.py에서 가져온 테스트 케이스
testcases = [
    ([Dot(PI / 6, 2 * PI), Dot(PI / 6, 2 * PI / 3), Dot(PI / 6, 4 * PI / 3)], "testcase_1"),
    ([Dot(PI / 12, PI), Dot(PI / 4, PI / 2), Dot(PI / 4, 2 * PI)], "testcase_2"),
    ([Dot(PI / 12, 2 * PI), Dot(PI / 2, PI / 4), Dot(PI / 2, PI / 12), Dot(PI / 3, PI / 6)], "testcase_3"),
    ([Dot(0.00000001, 2 * PI), Dot(PI / 4, PI / 2), Dot(PI / 4, 2 * PI), Dot(PI / 3, PI)], "testcase_4"),
    ([Dot(0.00000001, 2 * PI), Dot(PI / 6, PI / 6), Dot(PI / 6, 2 * PI), Dot(PI / 3, 2 * PI), Dot(PI / 6, -PI / 6)], "testcase_5")
]

# 특수 각도 테스트 케이스
special_angle_cases = [
    # 정확히 120° 각도를 이루는 점들 (Steiner 점 삽입 경계 조건)
    ([Dot(PI/6, 0), Dot(PI/6, 2*PI/3), Dot(PI/6, 4*PI/3)], "exactly_120_degrees"),

    # 119° 각도 (Steiner 점 삽입 대상)
    ([Dot(PI/6, 0), Dot(PI/6, 119*PI/180), Dot(PI/6, 239*PI/180)], "almost_120_degrees_under"),

    # 121° 각도 (Steiner 점 삽입 대상 아님)
    ([Dot(PI/6, 0), Dot(PI/6, 121*PI/180), Dot(PI/6, 241*PI/180)], "almost_120_degrees_over"),

    # 일직선상 배치 (180°)
    ([Dot(PI/6, 0), Dot(PI/6, PI/2), Dot(PI/6, PI)], "straight_line"),

    # 정삼각형 (각 60°)
    ([Dot(PI/3, 0), Dot(PI/3, 2*PI/3), Dot(PI/3, 4*PI/3)], "equilateral_triangle"),

    # 매우 작은 각도 (1°)
    ([Dot(PI/6, 0), Dot(PI/6, PI/180), Dot(PI/6, PI)], "very_small_angle"),

    # 매우 가까운 두 점과 하나의 원거리 점
    ([Dot(PI/6, 0), Dot(PI/6, 0.001), Dot(PI/6, PI)], "very_close_points"),
]

# 특수 각도 케이스를 기존 테스트케이스에 추가
testcases.extend(special_angle_cases)

# 랜덤 테스트 케이스 생성
for i in range(50):
    dots = [Dot() for _ in range(random.randint(3, 5))]
    testcases.append((dots, f"random_case_{i+1}"))

@pytest.mark.parametrize("dots, name", testcases)
def test_algorithm_length_reduction(dots, name):
    """
    MST -> SMT -> Optimized SMT 각 단계에서 그래프의 총 길이가 감소하거나 같은지 테스트합니다.
    """
    # 1. MST 생성
    mst = MinimalSpanningTree(dots)
    mst_length = mst.length()

    # 2. Steiner Tree 생성
    smt = SteinerTree(mst, InsertionMode.DECREASE_ONLY)
    smt_length = smt.length()

    # 스타이너 점이 없으면 최적화 스킵
    if smt.steiner_count == 0:
        print(f"Testing {name}: No Steiner points, skipping optimization")
        return

    # 스타이너 점들의 정보 수집
    steiner_points_info = []
    for i, sp in enumerate(smt.steiner_vertices):
        steiner_idx = smt.fixed_count + i
        neighbors = smt.adj[steiner_idx]
        steiner_points_info.append({
            "index": steiner_idx,
            "position": {"theta": sp.theta, "phi": sp.phi},
            "neighbors": list(neighbors)
        })

    # 연결 정보 수집
    connections = []
    total_vertices = smt.fixed_count + smt.steiner_count
    for u in range(total_vertices):
        for v in smt.adj[u]:
            if u < v:  # 중복 방지
                connections.append({
                    "from": u,
                    "to": v,
                    "distance": smt.get_distance(u, v)
                })

    # 3. Optimizer 적용
    optimizer = make_local_optimizer(
        backend="geo",
        steiner_tree=smt,
        optim_name="radam",
        hyper_param={"lr": 0.001},
        max_iter=50,
        tolerance=1e-6,
        device="cpu"
    )

    # 최적화 실행
    final_loss, loss_history = optimizer.run()

    opt_smt_length = final_loss.item()

    # SMT 길이 >= Optimized SMT 길이 여부 확인
    assert smt_length >= opt_smt_length - 1e-5, f"Optimized SMT should be shorter than or equal to SMT for {name}\n" \
                                        f"MST: {mst_length:.6f}, SMT: {smt_length:.6f}, Opt-SMT: {opt_smt_length:.6f}\n" \
                                        f"Steiner points: {steiner_points_info}\n" \
                                        f"Connections: {connections}\n"\
                                        f"loss_history chunks (10 gap): {[loss_history[i] for i in range(0, len(loss_history), 10)]}\n"\
                                        f"first loss: {loss_history[0]}\n"\

