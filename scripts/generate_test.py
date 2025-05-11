import json
import os
import pickle
import random
import sys

from geometry.dot import Dot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

INPUT_BASE = os.path.join(PROJECT_ROOT, "data", "inputs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


def generate_test(num_dots: int, num_test: int, base_seed: int):
    """
    num_dot 개의 Dot 객체를 생성한 후
    data/inputs/{num_dot} dots 폴더에 num_test개만큼 저장하고,
    각 파일에 사용된 시드를 seeds.json에 기록합니다.
    """
    output_dir = os.path.join(INPUT_BASE, f"{num_dots} dots")
    os.makedirs(output_dir, exist_ok=True)

    seed_map = {}
    for i in range(1, num_test + 1):
        seed = base_seed + i  # 재현 가능한 시드 설정
        random.seed(seed)
        dots = [Dot() for _ in range(num_dots)]

        filename = f"{i:03d}_test.pkl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(dots, f)

        seed_map[filename] = seed  # 파일명과 시드 매핑

    # 시드 매핑 정보 저장
    seeds_path = os.path.join(output_dir, "seeds.json")
    with open(seeds_path, "w", encoding="utf-8") as f:
        json.dump(seed_map, f, ensure_ascii=False, indent=2)

    print(f"'{output_dir}'에 {num_test}개의 테스트 파일 생성 (base_seed={base_seed})")


def main():
    # 설정 파일 로드: config/generate_test_config.json
    generate_test_cfg_path = os.path.join(CONFIG_DIR, "generate_test_config.json")
    with open(generate_test_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_seed = cfg["base_seed"]  # 기본 시드
    num_dots_list = cfg["num_dots"]  # 테스트할 Dot 개수 리스트
    num_tests = cfg["num_tests"]  # 폴더당 생성할 테스트 수

    # 입력 디렉토리 기본 폴더 생성
    os.makedirs(INPUT_BASE, exist_ok=True)

    # 설정에 따라 각 num_dot별 테스트 생성
    for num_dots in num_dots_list:
        generate_test(num_dots, num_tests, base_seed)


if __name__ == "__main__":
    main()
