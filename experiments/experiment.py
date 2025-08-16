import argparse
import json
import logging
import os
import pickle
import sys
import hashlib
import datetime
import pandas as pd
import time

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


def _short_tag(exp_cfg: dict) -> str:
    opt = exp_cfg["optimizer"]; sch = exp_cfg["scheduler"]
    lr = opt.get("params", {}).get("lr", "")
    ds = sch.get("params", {}).get("decay_steps", "")
    em = sch.get("params", {}).get("eta_min", "")
    name = f'{opt["name"]}-{sch["name"]}'
    return f"{name}-lr{lr}-ds{ds}-em{em}"

def _run_id(exp_cfg: dict) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    return f"{now}_{_short_tag(exp_cfg)}"

def _flatten_cfg(cfg: dict) -> str:
    # 설정 스냅샷 해시용(안정적 직렬화)
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False)

def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def test_case(dots: list[Dot], exp_config: dict) -> dict:
    """
    하나의 테스트 케이스를 수행하고 결과를 딕셔너리 형태로 반환.
    """
    # MST 생성 시간 측정
    mst_start = time.time()
    mst = MinimalSpanningTree(dots)
    mst_len = mst.length()
    mst_time = time.time() - mst_start

    # SMT 생성 시간 측정
    smt_start = time.time()
    insertion_mode = InsertionMode(exp_config["insertion_mode"])
    smt = SteinerTree(mst, insertion_mode)
    smt_len = smt.length()
    smt_time = time.time() - smt_start

    # 스타이너 점이 0개인 경우 지역최적화를 건너뛰고 SMT 길이를 그대로 반환
    if smt.steiner_count == 0:
        return {
            "mst_length": mst_len,
            "smt_length": smt_len,
            "opt_smt_length": smt_len,  # 최적화 없이 SMT 길이 그대로
            "opt_smt_curve": [smt_len],  # 단일 값 리스트
            "steiner_ratio": 1 - float(smt_len) / mst_len,
            "opt_steiner_ratio": 1 - float(smt_len) / mst_len,  # 최적화 안했으므로 동일
            "optimization_iterations": 0,  # 최적화 실행 안함
            "early_stop_reason": "No Steiner points, optimization skipped",
            "mst_time": mst_time,
            "smt_time": smt_time,
            "opt_time": 0.0,  # 최적화 실행 안함
            "total_time": mst_time + smt_time,
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

    # 최적화 실행 시간 측정
    opt_start = time.time()
    final_loss, loss_history = optimizer.run()
    opt_time = time.time() - opt_start

    min_loss = optimizer.min_loss
    early_reason = getattr(optimizer, "last_early_stop_reason", None)

    # 결과 사전
    return {
        "mst_length": mst_len,
        "smt_length": smt_len,
        "opt_smt_curve": [float(loss) for loss in loss_history],
        "opt_smt_length": float(min_loss),
        "steiner_ratio": 1 - smt_len / mst_len,
        "opt_steiner_ratio": 1 - float(final_loss) / mst_len,
        "optimization_iterations": len(loss_history),
        "early_stop_reason": early_reason,
        "mst_time": mst_time,
        "smt_time": smt_time,
        "opt_time": opt_time,
        "total_time": mst_time + smt_time + opt_time,
    }


def run_experiments(num_dots: list[int], num_tests: int, exp_config: dict):
    """
    data/inputs/{num_dot} 폴더에서 .pkl 파일 num_tests개를 읽어 실험 실행 후,
    data/outputs/run_id/{num_dot}_dots/ 폴더에 결과 저장
    """
    try:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        logging.info(f"출력 기본 디렉토리 생성/확인: {OUTPUT_BASE}")
    except Exception as e:
        logging.error(f"출력 기본 디렉토리 생성 실패: {e}")
        return

    # run 전용 최상위 디렉토리 생성
    run_id = _run_id(exp_config)
    run_base_dir = os.path.join(OUTPUT_BASE, run_id)

    try:
        os.makedirs(run_base_dir, exist_ok=True)
        logging.info(f"Run 디렉토리 생성: {run_base_dir}")
    except Exception as e:
        logging.error(f"Run 디렉토리 생성 실패: {run_base_dir}, 오류: {e}")
        return

    # manifest.json을 run 최상위에 저장
    manifest = {
        "run_id": run_id,
        "num_dots": num_dots,
        "config": exp_config,
        "config_sha1": hashlib.sha1(_flatten_cfg(exp_config).encode("utf-8")).hexdigest(),
        "created_at": datetime.datetime.now().isoformat(),
    }

    try:
        _write_json(os.path.join(run_base_dir, "manifest.json"), manifest)
        logging.info(f"manifest.json 저장 완료: {run_base_dir}")
    except Exception as e:
        logging.error(f"manifest.json 저장 실패: {e}")
        return

    # 전체 Run의 집계용 메트릭스
    all_metrics_rows = []

    # 전체 Dot 개수 리스트에 tqdm 적용
    for num_dot in num_dots:
        subdir = f"{num_dot} dots"
        in_dir = os.path.join(INPUT_BASE, subdir)
        
        if not os.path.isdir(in_dir):
            logging.warning(f"입력 폴더가 없습니다: {in_dir}")
            continue

        # 각 점 개수별 디렉토리 생성
        dot_dir = os.path.join(run_base_dir, f"{num_dot}_dots")
        cases_dir = os.path.join(dot_dir, "cases")
        curves_dir = os.path.join(dot_dir, "curves")

        try:
            os.makedirs(cases_dir, exist_ok=True)
            os.makedirs(curves_dir, exist_ok=True)
            logging.info(f"점별 디렉토리 생성: {dot_dir}")
        except Exception as e:
            logging.error(f"점별 디렉토리 생성 실패: {dot_dir}, 오류: {e}")
            continue
            
        metrics_rows = []

        logging.info(
            f"[실험 시작] {num_dot} dots - backend={exp_config['backend']} - "
            f"optimizer={exp_config['optimizer']['name']} - "
            f"scheduler={exp_config['scheduler']['name']} - "
            f"max_iter={exp_config['max_iterations']}"
        )

        # 실제 존재하는 파일들만 처리
        existing_files = []
        for i in range(1, num_tests + 1):
            fname = f"{i:03d}_test.pkl"
            path = os.path.join(in_dir, fname)
            if os.path.exists(path):
                existing_files.append((i, fname, path))
            else:
                logging.warning(f"파일이 존재하지 않음: {path}")
        
        if not existing_files:
            logging.error(f"처리할 파일이 없습니다: {in_dir}")
            continue
            
        logging.info(f"처리할 파일 수: {len(existing_files)}/{num_tests}")

        # 각 테스트 케이스 파일에 tqdm 적용
        for i, fname, path in tqdm(existing_files, desc=f"{num_dot} dots 처리중"):
            try:
                # 테스트 케이스 로드
                with open(path, "rb") as rf:
                    dots = pickle.load(rf)
                logging.debug(f"파일 로드 성공: {fname}")

                # 실험 수행
                result = test_case(dots, exp_config)
                logging.debug(f"실험 수행 완료: {fname}")
                
                # per-case 저장
                case_json = os.path.join(cases_dir, fname.replace(".pkl", ".json"))
                case_data = {
                    "mst_length": result["mst_length"],
                    "smt_length": result["smt_length"],
                    "opt_smt_length": result["opt_smt_length"],
                    "steiner_ratio": result["steiner_ratio"],
                    "optimization_iterations": result["optimization_iterations"],
                    "early_stop_reason": result["early_stop_reason"],
                    "mst_time": result["mst_time"],
                    "smt_time": result["smt_time"],
                    "opt_time": result["opt_time"],
                    "total_time": result["total_time"],
                }
                _write_json(case_json, case_data)
                logging.debug(f"case JSON 저장 완료: {case_json}")
                
                # 곡선은 별도
                curve_json = os.path.join(curves_dir, fname.replace(".pkl", "_curve.json"))
                curve_data = {"opt_smt_curve": result["opt_smt_curve"]}
                _write_json(curve_json, curve_data)
                logging.debug(f"curve JSON 저장 완료: {curve_json}")

                # 집계 행
                metric_row = {
                    "num_dot": num_dot,
                    "case": fname.replace(".pkl", ""),
                    "steiner_ratio": result["steiner_ratio"],
                    "opt_steiner_ratio": result["opt_steiner_ratio"],
                    "mst_length": result["mst_length"],
                    "smt_length": result["smt_length"],
                    "opt_smt_length": result["opt_smt_length"],
                    "iterations": result["optimization_iterations"],
                    "early_stop_reason": result["early_stop_reason"],
                    "mst_time": result["mst_time"],
                    "smt_time": result["smt_time"],
                    "opt_time": result["opt_time"],
                    "total_time": result["total_time"],
                }
                metrics_rows.append(metric_row)
                all_metrics_rows.append(metric_row)

            except FileNotFoundError as e:
                logging.error(f"파일을 찾을 수 없음: {fname}, 오류: {e}")
                continue
            except pickle.UnpicklingError as e:
                logging.error(f"pickle 파일 로드 실패: {fname}, 오류: {e}")
                continue
            except json.JSONEncodeError as e:
                logging.error(f"JSON 인코딩 실패: {fname}, 오류: {e}")
                continue
            except Exception as e:
                logging.error(f"테스트 케이스 {fname} 실행 중 예상치 못한 오류: {e}")
                import traceback
                logging.error(f"스택 트레이스: {traceback.format_exc()}")
                continue

        # 점별 metrics.csv 저장
        if metrics_rows:
            try:
                df = pd.DataFrame(metrics_rows)
                csv_path = os.path.join(dot_dir, "metrics.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8')
                logging.info(f"점별 metrics.csv 저장 완료: {csv_path} ({len(metrics_rows)}개 행)")
            except Exception as e:
                logging.error(f"점별 metrics.csv 저장 실패: {e}")
        else:
            logging.warning(f"저장할 metrics 데이터가 없습니다: {num_dot} dots")

        logging.info(f"[완료] {num_dot} dots 결과 저장 → {dot_dir}")

    # 전체 Run 통합 metrics.csv 저장
    if all_metrics_rows:
        try:
            df_all = pd.DataFrame(all_metrics_rows)
            all_csv_path = os.path.join(run_base_dir, "all_metrics.csv")
            df_all.to_csv(all_csv_path, index=False, encoding='utf-8')
            logging.info(f"통합 metrics.csv 저장 완료: {all_csv_path} ({len(all_metrics_rows)}개 행)")
        except Exception as e:
            logging.error(f"통합 metrics.csv 저장 실패: {e}")

    # 전체 실험 결과 요약 파일 제작
    if all_metrics_rows:
        try:
            summary_stats = _generate_experiment_summary(all_metrics_rows, exp_config)

            # 요약 CSV 저장 (num_dot별 통계)
            summary_csv_path = os.path.join(run_base_dir, "experiment_summary.csv")
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')

            logging.info(f"실험 요약 CSV 저장 완료: {summary_csv_path}")
        except Exception as e:
            logging.error(f"실험 요약 파일 생성 실패: {e}")

    logging.info(f"[전체 완료] Run 결과 저장 → {run_base_dir}")


def _generate_experiment_summary(all_metrics_rows: list[dict], exp_config: dict) -> list[dict]:
    """
    전체 실험 결과에 대한 요약 통계를 CSV용 리스트로 생성합니다.
    """
    df = pd.DataFrame(all_metrics_rows)

    # num_dot별 상세 통계를 CSV 행으로 변환
    summary_rows = []
    for num_dot in sorted(df["num_dot"].unique()):
        subset = df[df["num_dot"] == num_dot]

        # early_stop_reason별 개수 계산
        early_counts = subset["early_stop_reason"].value_counts().to_dict()

        # 기본 통계 행
        row = {
            "num_dot": int(num_dot),
            "case_count": len(subset),
            "avg_steiner_ratio": round(float(subset["steiner_ratio"].mean()), 6),
            "std_steiner_ratio": round(float(subset["steiner_ratio"].std()), 6),
            "avg_opt_steiner_ratio": round(float(subset["opt_steiner_ratio"].mean()), 6),
            "std_opt_steiner_ratio": round(float(subset["opt_steiner_ratio"].std()), 6),
            "avg_iterations": round(float(subset["iterations"].mean()), 2),
            "std_iterations": round(float(subset["iterations"].std()), 2),
            "avg_mst_time": round(float(subset["mst_time"].mean()), 6),
            "avg_smt_time": round(float(subset["smt_time"].mean()), 6),
            "avg_opt_time": round(float(subset["opt_time"].mean()), 6),
            "avg_total_time": round(float(subset["total_time"].mean()), 6),
        }

        # early_stop_reason별 개수를 컬럼으로 추가
        for reason, count in early_counts.items():
            col_name = f"early_stop_{reason}_count" if reason else "early_stop_None_count"
            row[col_name] = int(count)

        summary_rows.append(row)

    return summary_rows


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
