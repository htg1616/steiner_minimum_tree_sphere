import argparse
import json
import logging
import os
import pickle
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from geometry.dot import Dot
from graph.enums import InsertionMode
from graph.mst import MinimalSpanningTree
from graph.local_opt import make_local_optimizer
from graph.steiner import SteinerTree
from optimizer.early_stopper import EarlyStopConfig, EarlyStopper

INPUT_BASE = os.path.join(PROJECT_ROOT, "data", "inputs")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data", "outputs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def test_case(dots: list[Dot], exp_config: dict) -> dict:
    """
    하나의 테스트 케이스를 수행하고 결과를 딕셔너리 형태로 반환.
    """
    # MST 생성
    mst = MinimalSpanningTree(dots)
    mst_len = mst.length()

    # SMT 생성 (삽입 모드에 따라)
    insertion_mode = InsertionMode(exp_config["insertion_mode"])
    smt = SteinerTree(mst, insertion_mode)
    smt_len = smt.length()

    # 스타이너 점이 0개인 경우 지역최적화를 건너뛰고 SMT 길이를 그대로 반환
    if smt.steiner_count == 0:
        return {
            "mst_length": mst_len,
            "smt_length": smt_len,
            "opt_smt_length": smt_len,  # 최적화 없이 SMT 길이 그대로
            "opt_smt_curve": [smt_len],  # 단일 값 리스트
            "optimization_iterations": 0  # 최적화 실행 안함
        }

    # EarlyStopper 인스턴스 생성 (없으면 비활성)
    early_cfg = EarlyStopConfig.from_dict(exp_config.get("early_stop", {}))
    early_stopper = EarlyStopper(early_cfg)

    # nested 구조로 변경된 설정 사용
    optimizer = make_local_optimizer(
        backend=exp_config["backend"],
        steiner_tree=smt,
        optim_name=exp_config["optimizer"]["name"],
        optim_param=exp_config["optimizer"].get("params", {}),
        max_iter=exp_config["max_iterations"],
        scheduler_name=exp_config["scheduler"]["name"],
        scheduler_param=exp_config["scheduler"].get("params", {}),
        tolerance=exp_config["tolerance"],
        device=exp_config["device"],
        early_stopper=early_stopper,  # ← 주입
    )

    # 최적화 실행
    final_loss, loss_history = optimizer.run()

    # 결과 사전
    return {
        "mst_length": mst_len,
        "smt_length": smt_len,
        "opt_smt_length": final_loss.item() if hasattr(final_loss, 'item') else float(final_loss),
        "opt_smt_curve": [float(loss) for loss in loss_history],
        "optimization_iterations": len(loss_history)
    }


def run_experiments(num_dots: list[int], num_tests: int, exp_config: dict):
    """
    data/inputs/{num_dot} 폴더에서 .pkl 파일 num_tests개를 읽어 실험 실행 후,
    data/outputs/{num_dot} 폴더에 JSON 결과 저장
    """
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 전체 Dot 개수 리스트에 tqdm 적용
    for num_dot in num_dots:
        subdir = f"{num_dot} dots"
        in_dir = os.path.join(INPUT_BASE, subdir)
        out_dir = os.path.join(OUTPUT_BASE, subdir)

        if not os.path.isdir(in_dir):
            logging.warning(f"'{subdir}' 폴더가 없습니다. 건너뜁니다.")
            continue

        os.makedirs(out_dir, exist_ok=True)
        logging.info(
            f"[실험 시작] {subdir} - backend={exp_config['backend']} - "
            f"optimizer={exp_config['optimizer']['name']} - "
            f"scheduler={exp_config['scheduler']['name']} - "
            f"max_iter={exp_config['max_iterations']}"
        )

        # 각 테스트 케이스 파일에 tqdm 적용
        for i in tqdm(range(1, num_tests + 1), desc=f"{num_dot} dots 처리중"):
            fname = f"{i:03d}_test.pkl"
            # 테스트 케이스 로드
            path = os.path.join(in_dir, fname)
            with open(path, "rb") as rf:
                dots = pickle.load(rf)

            try:
                # 실험 수행
                result = test_case(dots, exp_config)

                # JSON 저장 (.json 확장자)
                json_path = os.path.join(out_dir, fname.replace(".pkl", ".json"))
                with open(json_path, "w", encoding="utf-8") as wf:
                    json.dump(result, wf, ensure_ascii=False, indent=2)
            
            except Exception as e:
                logging.error(f"테스트 케이스 {fname} 실행 중 오류: {e}")
                continue

        logging.info(f"[완료] {subdir} 결과 저장 → {out_dir}")


def main():
    # 디버깅: 현재 작업 디렉토리와 경로 출력
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"스크립트 위치: {os.path.dirname(os.path.abspath(__file__))}")

    """
    설정 파일(config/)에서 실험 파라미터를 읽고 run_experiments를 호출합니다.
    """
    # 설정 파일 경로
    gen_cfg_path = os.path.join(CONFIG_DIR, "generate_test_config.json")
    exp_cfg_path = os.path.join(CONFIG_DIR, "experiment_config.json")

    # 설정 로드
    with open(gen_cfg_path, "r", encoding="utf-8") as gf:
        gen_cfg = json.load(gf)
    with open(exp_cfg_path, "r", encoding="utf-8") as ef:
        exp_cfg = json.load(ef)

    num_dots = gen_cfg["num_dots"]  # Dot 개수 리스트
    num_tests = gen_cfg["num_tests"]  # 폴더당 테스트 파일 수

    # CLI 옵션으로 experiment config 조정 가능
    parser = argparse.ArgumentParser(description="Geodesic Steiner Tree 실험 스크립트")
    parser.add_argument("-b", "--backend", type=str, default=exp_cfg.get("backend", "geo"),
                  help="최적화 백엔드 (예: torch, numpy)")
    parser.add_argument("-o", "--optimizer", type=str, default=exp_cfg["optimizer"]["name"],
                        help="옵티마이저 이름")
    parser.add_argument("-s", "--scheduler", type=str, default=exp_cfg["scheduler"]["name"],
                        help="스케줄러 이름")
    parser.add_argument("-i", "--insertion_mode", type=str, default=exp_cfg["insertion_mode"],
                        help="삽입 모드 (NORMAL, GREEDY, RANDOM)")
    parser.add_argument("-m", "--max_iterations", type=int, default=exp_cfg["max_iterations"],
                        help="최대 반복 횟수")
    args = parser.parse_args()

    # CLI 인자로 실험 설정 업데이트
    exp_cfg["backend"] = args.backend
    exp_cfg["optimizer"]["name"] = args.optimizer
    exp_cfg["scheduler"]["name"] = args.scheduler
    exp_cfg["insertion_mode"] = args.insertion_mode
    exp_cfg["max_iterations"] = args.max_iterations

    run_experiments(num_dots, num_tests, exp_cfg)


if __name__ == "__main__":
    main()
