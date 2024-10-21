from graph import *

PI = math.pi  # 상수는 대문자를 사용해야함

testcases = [[Dot(PI / 6, 2 * PI), Dot(PI / 6, 2 * PI / 3), Dot(PI / 6, 4 * PI / 3)],
             [Dot(PI/12, PI), Dot(PI / 2, PI / 2), Dot(PI / 2, 2 * PI)],
             [Dot(PI/12, 2 * PI), Dot(PI / 2, PI / 4), Dot(PI / 2, PI/12), Dot(PI/3, PI/6)],
             [Dot(0.00000001, 2 * PI), Dot(PI / 2, PI / 2), Dot(PI / 2, 2 * PI), Dot(PI/3,  PI)],
             [Dot(0.00000001, 2 * PI), Dot(PI / 6, PI / 6), Dot(PI / 6, 2 * PI), Dot(PI / 3, 2 * PI),
              Dot(PI / 6, -PI / 6)]
             ]
for i in range(5):
    print()
    print(f'=====<테스트케이스{i + 1}>=====')
    dots = testcases[i]

    msttest = MinimalSpanningTree(dots)
    print("MST 인접리스트", msttest.adj_list)
    print()

    smttest=SteinerTree(dots, msttest.adj_list)
    print("SMT 인접리스트", smttest.adj_list)
    print("SMT 간선", smttest.vertices)
    print("SMT 새로운 간선", smttest.si_vertices)
    print()

# 테스트 케이스6
print()
print('=====<테스트케이스6>=====')

a = Dot(PI / 3, PI / 2)
b = Dot(PI / 3, 2 * PI / 3)
c = Dot(PI / 3, 2 * PI)
d = Dot(0.7855726963, 1.046673952)

print(f'a,d간 거리: {a - d}')  # 약 0.486
print(f'b,d간 거리: {b - d}')  # 약 0.85
print(f'c,d간 거리: {c - d}')  # 약 0.85

c1 = Dot(PI / 3, 2 * PI)

print(c1 == c)
print(c1 == b)
