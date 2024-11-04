import pickle as pkl
from graph import *

def test(num_dot: int, generation: int, generation_list: list[int]):
    dots = [Dot() for i in range(num_dot)]
    mst_test = MinimalSpanningTree(dots)
    mst_test_length = mst_test.length()
    smt_test = SteinerTree(dots, mst_test.adj_list)
    opt_smt_test = LocalOptimizedGraph(smt_test.vertices, smt_test.si_vertices, smt_test.adj_list)
    result = [None] + opt_smt_test.optimze(generation)
    return [(1-result[i]/mst_test_length)*100 for i in generation_list] #mst에 비하여 감소한 퍼센트를 저장

import time
start_time = time.time()

num_dot_list = [100, 500, 1000]
iterations = 10000
generation = 500
generation_list = [50, 100, 500]
with open("results.pkl", "wb") as f:
    for num_dot in num_dot_list:
        for _ in range(iterations):
            result = test(num_dot, generation, generation_list)
            pkl.dump(result, f)

end_time = time.time()
print(end_time - start_time)