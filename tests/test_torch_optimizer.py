import torch
import pytest
import math

from optimizer.torch_optimizer import TorchOptimizer, length_angles, make_torch_factory


class TestLengthAngles:
    """length_angles 함수 테스트"""

    def test_triangle_angles(self):
        """세 각이 모두 직각인 삼각형의 각도 계산 테스트"""
        # 북극과 적도상 2개 점으로 이루어진 삼각형
        theta_phi = torch.tensor([
            [math.pi/2, 0.0],
            [math.pi/2, math.pi/2],
            [0.0, 0.0]
        ], dtype=torch.float64)

        edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 0]
        ], dtype=torch.long)

        total_length = length_angles(theta_phi, edge_index)

        # 각 변의 길이는 pi/2이므로 총 길이는 3*pi/2
        expected = 3 * math.pi / 2
        assert torch.isclose(total_length, torch.tensor(expected, dtype=torch.float64), rtol=1e-5)

    def test_gradient_computation(self):
        """그래디언트 계산 테스트"""
        theta_phi = torch.tensor([
            [math.pi/2, 0.0],
            [math.pi/3, math.pi/4]
        ], dtype=torch.float64, requires_grad=True)

        edge_index = torch.tensor([[0, 1]], dtype=torch.long)

        loss = length_angles(theta_phi, edge_index)
        loss.backward()

        assert theta_phi.grad is not None
        assert torch.isfinite(theta_phi.grad).all()
        assert theta_phi.grad.shape == theta_phi.shape

    def test_single_edge(self):
        """단일 간선 테스트"""
        theta_phi = torch.tensor([
            [0.0, 0.0],      # 북극
            [math.pi, 0.0]   # 남극
        ], dtype=torch.float64)

        edge_index = torch.tensor([[0, 1]], dtype=torch.long)

        distance = length_angles(theta_phi, edge_index)
        # 북극-남극 거리는 π
        assert torch.isclose(distance, torch.tensor(math.pi, dtype=torch.float64), rtol=1e-5)


class TestMakeTorchFactory:
    """make_torch_factory 함수 테스트"""

    @pytest.mark.parametrize("name, hyper_param, expected_class", [
        ("adam", {"lr": 0.01, "betas": (0.9, 0.999)}, torch.optim.Adam),
        ("sgd", {"lr": 0.1, "momentum": 0.9}, torch.optim.SGD)
    ])
    def test_optimizers(self, name, hyper_param, expected_class):
        """지원되는 옵티마이저 팩토리 테스트"""
        factory = make_torch_factory(name, hyper_param)

        param = torch.nn.Parameter(torch.randn(3, 2))
        optimizer = factory(param)

        assert isinstance(optimizer, expected_class)
        for key, value in hyper_param.items():
            assert optimizer.param_groups[0][key] == value

    def test_invalid_optimizer_name(self):
        with pytest.raises(ValueError):
            make_torch_factory("invalid_optimizer", {"lr": 0.01})

class TestTorchOptimizer:
    """TorchOptimizer 클래스 테스트"""

    @pytest.fixture
    def simple_graph_data(self):
        """간단한 그래프 데이터 fixture"""
        # 4개 점: 3개 터미널 + 1개 스타이너
        vertices = torch.tensor([
            [0.0, 0.0],      # 터미널 1
            [math.pi/2, 0],  # 터미널 2
            [math.pi/2, math.pi/2], # 터미널 3
            [math.pi/3, math.pi/3] #스타이너 점
        ], dtype=torch.float64)

        edge_index = torch.tensor([
            [0, 3], [1, 3], [2, 3]  # 터미널들이 스타이너 점에 연결
        ], dtype=torch.long)

        steiner_mask = torch.tensor([False, False, False, True])

        return vertices, edge_index, steiner_mask

    @pytest.mark.parametrize("theta, phi, description", [
        (-0.1, 0.0, "negative_theta"),
        (math.pi + 0.1, 0.0, "theta_over_pi"),
        (math.pi/2, -math.pi/2, "negative_phi"),
        (math.pi/2, 3 * math.pi, "phi_over_2pi"),
        (0.0, 0.0, "theta_phi_zero"),
        (math.pi, 0.0, "thtea_pi_phi_zero"),
        (math.pi/2, 2*math.pi - 1e-6, "boundary_phi_max"),
        (math.pi + 0.1, -0.1, "both_violations"),
    ])
    def test_post_step(self, simple_graph_data, theta, phi, description):
        """post_step 제약 조건 테스트 - 다양한 경계값과 위반값 처리"""
        vertices, edge_index, steiner_mask = simple_graph_data

        optimizer = TorchOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name="adam"
        )

        # 테스트할 값 설정
        with torch.no_grad():
            optimizer.train_param[0, 0] = theta
            optimizer.train_param[0, 1] = phi

        optimizer.post_step()

        # 제약 조건 확인
        final_theta = optimizer.train_param[0, 0].item()
        final_phi = optimizer.train_param[0, 1].item()

        assert 0 <= final_theta <= math.pi, \
            f"[{description}] theta 제약 조건 위반: {final_theta}"
        assert 0 <= final_phi < 2 * math.pi, \
            f"[{description}] phi 제약 조건 위반: {final_phi}"

        # 특정 케이스에 대한 추가 검증
        if description == "negative_phi":
            # 음수 phi는 2π를 더해서 양수로 변환되어야 함
            expected_phi = (phi % (2 * math.pi))
            assert abs(final_phi - expected_phi) < 1e-6, \
                f"음수 phi 처리 오류: 예상 {expected_phi}, 실제 {final_phi}"

    @staticmethod
    def build_case_north_pole():
        """북극 근처에서의 경계 테스트"""
        epsilon = 1e-3

        vertices = torch.tensor([
            [epsilon, 0],  # 북극 + 노이즈 1
            [epsilon, 2 * math.pi/3],  # 남극 + 노이즈
            [epsilon, 4 * math.pi/3],
            [0, 0] #스타이너 점
        ], dtype=torch.float64)
        edge_index = torch.tensor([[0, 3], [1, 3], [2, 3]], dtype=torch.long)
        steiner_mask = torch.tensor([False, False, False, True])

        return vertices, edge_index, steiner_mask

    @staticmethod
    def build_case_south_pole():
        """남극 근처에서의 경계 테스트"""
        epsilon = 1e-3

        vertices = torch.tensor([
            [math.pi - epsilon, 0],  # 거의 남극 + 노이즈
            [math.pi - epsilon, 2 * math.pi/3],
            [math.pi - epsilon, 4 * math.pi/3],
            [math.pi, 0] #스타이너 점
        ], dtype=torch.float64)
        edge_index = torch.tensor([[0, 3], [1, 3], [2, 3]], dtype=torch.long)
        steiner_mask = torch.tensor([False, False, False, True])

        return vertices, edge_index, steiner_mask

    @pytest.mark.parametrize("builder",
        [ build_case_north_pole, build_case_south_pole ],
        ids=["north_pole", "south_pole"]
    )
    def test_boundary_stress(self, builder):
        """양 극점 근처 스트레스 테스트"""
        vertices, edge_index, steiner_mask = builder()

        # 원본 터미널 좌표 저장
        original_terminals = vertices[~steiner_mask].clone()

        optimizer = TorchOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name="adam",
            hyper_param={"lr": 0.01},
            max_iter=10
        )

        # 최적화 실행
        final_loss, loss_history = optimizer.run()

        # updated_full()로 전체 좌표 획득
        updated_vertices = optimizer.updated_full()
        updated_terminals = updated_vertices[~steiner_mask]

        # 터미널 좌표가 변하지 않았는지 확인
        assert torch.allclose(original_terminals, updated_terminals, rtol=1e-10, atol=1e-10), \
            "터미널 좌표가 최적화 과정에서 변경되었습니다"

        # 손실값과 그래디언트가 유한한지 확인
        assert torch.isfinite(final_loss), "최종 손실값이 무한대이거나 NaN입니다"
        assert all(math.isfinite(loss) for loss in loss_history), \
            "손실 기록에 무한대 또는 NaN 값이 있습니다"

        # 스타이너 점이 유한한 값인지 확인
        updated_vertices = optimizer.updated_full()
        assert torch.isfinite(updated_vertices).all(), \
            "업데이트된 좌표에 무한대 또는 NaN 값이 있습니다"
