import time

from graph import *

PI = math.pi  # 상수는 대문자를 사용해야함

random.seed(time.time())


def test(dots):
    mst_test = MinimalSpanningTree(dots)
    print("MST 인접리스트", mst_test.adj_list)
    print("MST 길이", mst_test.length())
    print()

    smt_test = SteinerTree(dots, mst_test.adj_list)
    print("SMT 인접리스트", smt_test.adj_list)
    print("SMT 길이", smt_test.length())
    print()

    opt_smt_test = LocalOptimizedGraph(smt_test.vertices, smt_test.si_vertices, smt_test.adj_list)
    result = opt_smt_test.optimze()
    print("OPT_SMT 길이", opt_smt_test.length())


testcases = [[Dot(PI / 6, 2 * PI), Dot(PI / 6, 2 * PI / 3), Dot(PI / 6, 4 * PI / 3)],
             [Dot(PI / 12, PI), Dot(PI / 4, PI / 2), Dot(PI / 4, 2 * PI)],
             [Dot(PI / 12, 2 * PI), Dot(PI / 2, PI / 4), Dot(PI / 2, PI / 12), Dot(PI / 3, PI / 6)],
             [Dot(0.00000001, 2 * PI), Dot(PI / 4, PI / 2), Dot(PI / 4, 2 * PI), Dot(PI / 3, PI)],
             [Dot(0.00000001, 2 * PI), Dot(PI / 6, PI / 6), Dot(PI / 6, 2 * PI), Dot(PI / 3, 2 * PI),
              Dot(PI / 6, -PI / 6)],
             [Dot(PI / 2, 0), Dot(0, 0), Dot(0, PI / 2)]
             ]


for i in range(6):
    print()
    print(f'=====<테스트케이스{i + 1}>=====')
    dots = testcases[i]
    test(dots)

for i in range(1):
    print(f'=====<랜덤케이스{i + 1}>=====')
    dots = [Dot() for i in range(random.randint(1000, 1050))]
    test(dots)
