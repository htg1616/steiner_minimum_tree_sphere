import torch
import geoopt
import pytest
import math

from optimizer.geo_optimizer import length_xyzs, make_geoopt_factory, GeoOptimizer


class TestLengthXYZs:
    """length_xyzs 함수 테스트"""

    @pytest.mark.parametrize("xyz, edge_index, expected, description", [
        (torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0, 1]], dtype=torch.long),
         0.0,
        "same_point"),
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
         torch.tensor([[0, 1]], dtype=torch.long),
         math.pi / 2,
         "orthogonal_vectors"),
        (torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64),
         torch.tensor([[0, 1]], dtype=torch.long),
         math.pi,
         "antipodal_points"),
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0, 1], [1, 2], [2, 0]]),
        3 * math.pi / 2,
        "multiple_edges")
        ])
    def test_distance(self, xyz, edge_index, expected, description):
        distance = length_xyzs(xyz, edge_index)
        assert torch.isclose(distance, torch.tensor(expected, dtype=torch.float64))


    def test_gradient_computation(self):
        """미분 가능성 테스트"""
        xyz = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [math.sqrt(2) / 2, math.sqrt(2) / 2, 0.0]
        ], dtype=torch.float64, requires_grad=True)

        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        total_distance = length_xyzs(xyz, edge_index)
        total_distance.backward()

        assert xyz.grad is not None
        assert torch.isfinite(xyz.grad).all()
        assert xyz.grad.shape == xyz.shape


class TestMakeGeoFactory:
    """make_geoopt_factory 함수 테스트"""

    @pytest.mark.parametrize("name, hyper_param, expected_class", [
        ("radam", {"lr": 0.02, "eps": 1e-4}, geoopt.optim.RiemannianAdam),
        ("rsgd", {"lr": 0.1, "momentum": 0.9}, geoopt.optim.RiemannianSGD)
    ])
    def test_optimizers(self, name, hyper_param, expected_class):
        """옵티마이저 매핑 및 하이퍼파라미터 전파 테스트"""
        factory = make_geoopt_factory(name, hyper_param)

        # 더미 매니폴드 파라미터 생성
        sphere = geoopt.Sphere()
        param = geoopt.ManifoldParameter(torch.randn(3, 3), manifold=sphere)

        optimizer = factory(param)
        assert isinstance(optimizer, expected_class)

        # 파라미터 그룹에서 하이퍼파라미터 확인
        param_group = optimizer.param_groups[0]
        for key, expected_value in hyper_param.items():
            assert param_group[key] == expected_value

    def test_invalid_optimizer_name(self):
        """잘못된 옵티마이저 이름 테스트"""
        with pytest.raises(ValueError, match="지원하지 않는 geoopt optimizer"):
            make_geoopt_factory("invalid_optimizer", {"lr": 0.01})


class TestGeoOptimizer:
    """GeoOptimizer 클래스 테스트"""

    @pytest.fixture
    def simple_graph_data(self):
        """3D 단위벡터 그래프 데이터 fixture"""
        # 4개 점: 3개 터미널 + 1개 스타이너 (별 모양)
        vertices = torch.tensor([
            [1.0, 0.0, 0.0],        # x축 터미널
            [0.0, 1.0, 0.0],        # y축 터미널
            [0.0, 0.0, 1.0],        # z축 터미널
            [1.0, 1.0, 0.0]         # 스타이너 점 (정규화 필요)
        ], dtype=torch.float64)

        # 스타이너 점을 단위벡터로 정규화
        vertices[3] = vertices[3] / vertices[3].norm()

        edge_index = torch.tensor([
            [0, 3], [1, 3], [2, 3]  # 터미널들이 스타이너 점에 연결
        ], dtype=torch.long)

        steiner_mask = torch.tensor([False, False, False, True])

        return vertices, edge_index, steiner_mask

    def test_input_shape_validation(self):
        """입력 형태 검증 테스트"""
        # 잘못된 차원 (2D 대신 3D)
        wrong_vertices = torch.randn(4, 2)  # (N, 2) 형태
        edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        steiner_mask = torch.tensor([False, True])

        with pytest.raises(ValueError):
            GeoOptimizer(
                vertices=wrong_vertices,
                edge_index=edge_index,
                steiner_mask=steiner_mask,
                optim_name="radam"
            )

    def test_create_train_param_manifold(self, simple_graph_data):
        """매니폴드 파라미터 생성 테스트"""
        vertices, edge_index, steiner_mask = simple_graph_data

        optimizer = GeoOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name="radam"
        )

        manifold_param = optimizer._create_train_param()

        # ManifoldParameter인지 확인
        assert isinstance(manifold_param, geoopt.ManifoldParameter)

        # Sphere 매니폴드인지 확인
        assert isinstance(manifold_param.manifold, geoopt.Sphere)

        # requires_grad=True인지 확인
        assert manifold_param.requires_grad

        # 단위 벡터인지 확인 (스타이너 점만)
        steiner_norms = manifold_param.norm(dim=-1)
        assert torch.allclose(steiner_norms, torch.ones_like(steiner_norms), atol=1e-6)

    def test_normalization_in_manifold_creation(self):
        """매니폴드 파라미터 생성 시 정규화 테스트"""
        # 의도적으로 노름이 1이 아닌 벡터 생성
        vertices = torch.tensor([
            [1.0, 0.0, 0.0],        # 단위벡터
            [2.0, 1.0, 1.0]         # 노름이 1이 아님
        ], dtype=torch.float64)

        edge_index = torch.tensor([[0, 1]], dtype=torch.long)
        steiner_mask = torch.tensor([False, True])

        optimizer = GeoOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name="radam"
        )

        manifold_param = optimizer._create_train_param()

        # 정규화되어 단위벡터가 되었는지 확인
        norms = manifold_param.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("scheduler_name, scheduler_param", [
        ("cosine", {"eta_min": 0.0}),
        ("plateau", {"patience": 1, "factor": 0.5}),
        (None, None),
    ])
    def test_scheduler(self, simple_graph_data, scheduler_name, scheduler_param):
        vertices, edge_index, steiner_mask = simple_graph_data
        optz = GeoOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name="radam",
            hyper_param={"lr": 0.01},
            max_iter=8,
            scheduler_name=scheduler_name,
            scheduler_param={"eta_min": 0.0} if scheduler_name == "cosine" else None,
        )
        final_loss, hist = optz.run()
        assert torch.isfinite(final_loss)
        assert all(math.isfinite(x) for x in hist)

    @pytest.mark.parametrize("optim_name", ["radam", "rsgd"])
    def test_integration_run(self, simple_graph_data, optim_name):
        """통합 실행 테스트"""
        vertices, edge_index, steiner_mask = simple_graph_data

        # 초기 손실 계산
        initial_loss = length_xyzs(vertices, edge_index)

        # 원본 터미널 좌표 저장
        original_terminals = vertices[~steiner_mask].clone()

        optimizer = GeoOptimizer(
            vertices=vertices,
            edge_index=edge_index,
            steiner_mask=steiner_mask,
            optim_name=optim_name,
            hyper_param={"lr": 0.01},
            max_iter=20
        )

        # 최적화 실행
        final_loss, loss_history = optimizer.run()

        # 모든 손실값 검증
        assert torch.isfinite(final_loss)
        assert all(math.isfinite(loss) for loss in loss_history), "손실 기록에 무한대 또는 NaN 값이 있습니다"
        assert len(loss_history) <= 20, "손실 기록이 max_iter를 초과했습니다"

        # updated_full() 검증
        updated_vertices = optimizer.updated_full()

        # Shape 확인
        assert updated_vertices.shape == vertices.shape, f"Shape 불일치: 예상 {vertices.shape}, 실제 {updated_vertices.shape}"

        # 모든 점이 단위벡터인지 확인
        norms = updated_vertices.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "업데이트된 좌표가 단위벡터가 아닙니다"

        # 터미널 좌표가 불변인지 확인
        updated_terminals = updated_vertices[~steiner_mask]
        assert torch.allclose(original_terminals, updated_terminals, rtol=1e-10, atol=1e-10), \
            "터미널 좌표가 최적화 과정에서 변경되었습니다"

        # 개선 여부 확인 (손실이 줄어들거나 거의 같아야 함)
        tolerance = 1e-5
        assert final_loss <= initial_loss + tolerance, \
            f"최적화 후 손실이 증가했습니다: {initial_loss.item():.6f} -> {final_loss.item():.6f}"
