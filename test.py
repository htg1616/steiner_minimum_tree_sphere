from graph import *


def test(dots: list[Dot], generation: int):
    mst = MinimalSpanningTree(dots)
    mst_length = mst.length()

    smt_false = SteinerTree(dots, mst.adj_list, False)
    smt_true = SteinerTree(dots, mst.adj_list, True)
    smt_true_length = smt_true.length()

    opt_smt_false = LocalOptimizedGraph(smt_false.vertices, smt_false.si_vertices, smt_false.adj_list)
    opt_smt_false_lengths = opt_smt_false.optimze(generation)
    opt_smt_true = LocalOptimizedGraph(smt_true.vertices, smt_true.si_vertices, smt_true.adj_list)
    opt_smt_true_lengths = opt_smt_true.optimze(generation)

    results = {
        "mst_length": mst_length,
        "smt_true_length": smt_true_length,
        "opt_smt_false_lengths": opt_smt_false_lengths,
        "opt_smt_true_lengths": opt_smt_true_lengths,
    }

    print()
    print(f"mst_length: {mst_length}")
    print(f"smt_true_length: {smt_true_length}")
    print(f"opt_smt_false_length: {opt_smt_false_lengths[-1]}")
    print(f"opt_smt_true_length: {opt_smt_true_lengths[-1]}")

    return results


def run_all_tests(generation=1000):
    input_base = 'inputs'
    output_base = 'outputs'

    # inputs 디렉토리 내의 각 서브디렉토리를 순회한다.
    for subdir in os.listdir(input_base):
        sub_input_path = os.path.join(input_base, subdir)
        sub_output_path = os.path.join(output_base, subdir)
        if os.path.isdir(sub_input_path):
            # 해당 서브디렉토리명으로 outputs 내에 서브디렉토리를 생성한다.
            if not os.path.exists(sub_output_path):
                os.mkdir(sub_output_path)
            test_files = sorted(os.listdir(sub_input_path))
            for test_file in test_files:
                test_path = os.path.join(sub_input_path, test_file)
                # pickle 형식으로 저장된 테스트 케이스를 불러온다.
                with open(test_path, 'rb') as f:
                    dots = pickle.load(f)
                result = test(dots, generation)
                # 동일한 파일 이름으로 outputs 내에 저장한다.
                output_file_path = os.path.join(sub_output_path, test_file)
                with open(output_file_path, 'wb') as f:
                    pickle.dump(result, f)

    print("모든 테스트 결과가 저장되었다.")


import os
import pickle


def compute_averages(outputs_dir='outputs', output_txt='averages.txt'):
    averages = {}

    # outputs 디렉토리 내의 서브디렉토리를 순회
    for subdir in os.listdir(outputs_dir):
        subdir_path = os.path.join(outputs_dir, subdir)
        if os.path.isdir(subdir_path):
            v1_list, v2_list, v3_list = [], [], []
            test_files = sorted(os.listdir(subdir_path))
            for test_file in test_files:
                file_path = os.path.join(subdir_path, test_file)
                # pickle 형식으로 저장된 결과 불러오기
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                mst_length = data.get("mst_length", None)
                smt_true_length = data.get("smt_true_length", None)
                opt_smt_false_lengths = data.get("opt_smt_false_lengths", [])
                opt_smt_true_lengths = data.get("opt_smt_true_lengths", [])

                # mst_length이 0 또는 None이면 계산할 수 없으므로 건너뛴다.
                if not mst_length or mst_length == 0:
                    continue

                # 최적화 결과는 리스트의 마지막 값을 사용한다고 가정
                if opt_smt_false_lengths:
                    opt_false = opt_smt_false_lengths[-1]
                else:
                    opt_false = None
                if opt_smt_true_lengths:
                    opt_true = opt_smt_true_lengths[-1]
                else:
                    opt_true = None

                # 값이 정상적으로 존재하는 경우 계산
                if smt_true_length is not None and opt_false is not None and opt_true is not None:
                    v1 = 1 - (smt_true_length / mst_length)
                    v2 = 1 - (opt_false / mst_length)
                    v3 = 1 - (opt_true / mst_length)

                    v1_list.append(v1)
                    v2_list.append(v2)
                    v3_list.append(v3)

            # 서브디렉토리 별 평균 계산
            count = len(v1_list)
            if count > 0:
                avg_v1 = sum(v1_list) / count
                avg_v2 = sum(v2_list) / count
                avg_v3 = sum(v3_list) / count
                averages[subdir] = (avg_v1, avg_v2, avg_v3)

    # averages.txt 파일에 결과 저장
    with open(output_txt, 'w') as f:
        for subdir, (avg_v1, avg_v2, avg_v3) in averages.items():
            f.write(f'{subdir}\n')
            f.write(f'  1 - (smt_true_length / mst_length): {avg_v1:.6f}\n')
            f.write(f'  1 - (opt_smt_false_length / mst_length): {avg_v2:.6f}\n')
            f.write(f'  1 - (opt_smt_true_length / mst_length): {avg_v3:.6f}\n\n')

    print(f"평균 계산 결과가 '{output_txt}' 파일에 저장되었다.")


if __name__ == "__main__":
    run_all_tests()
    compute_averages()

