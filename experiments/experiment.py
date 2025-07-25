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
from graph.local_opt import LocalOptimizedGraph
from graph.steiner import SteinerTree

INPUT_BASE = os.path.join(PROJECT_ROOT, "data", "inputs")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data", "outputs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def test_case(dots: list[Dot], generations: int, insertion_mode: InsertionMode) -> dict:
    """
    하나의 테스트 케이스를 수행하고 결과를 딕셔너리 형태로 반환.
    """
    # MST 생성
    mst = MinimalSpanningTree(dots)
    mst_len = mst.length()

    # SMT 생성 (두 가지 삽입 방식)
    smt = SteinerTree(mst, insertion_mode)
    smt_len = smt.length()

    # 지역 최적화
    opt_smt = LocalOptimizedGraph(smt)
    opt_smt_curve = opt_smt.optimize(generations)

    # 결과 사전
    return {
        "mst_length": mst_len,
        "smt_length": smt_len,
        "opt_smt_length": opt_smt_curve[-1],
        "opt_smt_curve": opt_smt_curve
    }


def run_experiments(num_dots: list[int], num_tests: int, generations: int, insertion_mode: InsertionMode):
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
        logging.info(f"[실험 시작] {subdir} - generations={generations} - insertion_mode={insertion_mode}")

        # 각 테스트 케이스 파일에 tqdm 적용
        for i in tqdm(range(1, num_tests + 1), desc=f"{num_dot} dots 처리중"):
            fname = f"{i:03d}_test.pkl"
            # 테스트 케이스 로드
            path = os.path.join(in_dir, fname)
            with open(path, "rb") as rf:
                dots = pickle.load(rf)

            # 실험 수행
            result = test_case(dots, generations, insertion_mode)

            # JSON 저장 (.json 확장자)
            json_path = os.path.join(out_dir, fname.replace(".pkl", ".json"))
            with open(json_path, "w", encoding="utf-8") as wf:
                json.dump(result, wf, ensure_ascii=False, indent=2)

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
    generations = exp_cfg["generations"]  # 기본 세대 수
    insertion_mode = exp_cfg["insertion_mode"]  # 삽입 모드

    # CLI 옵션으로 experiment config 조정 가능
    parser = argparse.ArgumentParser(description="Geodesic Steiner Tree 실험 스크립트")
    parser.add_argument("-g", "--generations", type=int, default=generations,
                        help="로컬 최적화 세대 수")
    parser.add_argument("-i", "--insertion_mode", type=InsertionMode, choices=list(InsertionMode) ,default=insertion_mode, help="thompson's method 세부 모드")
    args = parser.parse_args()

    run_experiments(num_dots, num_tests, args.generations, args.insertion_mode)


if __name__ == "__main__":
    main()
