import pickle as pkl
import os

from graph import *

def generate_test(num_dot: int, num_test: int):
    os.mkdir(f"inputs/{num_dot} dots")
    for i in range(1, num_test + 1):
        dots = [Dot() for _ in range(num_dot)]
        with open(f"inputs/{num_dot} dots/{i} test.pkl", "wb") as f:
            pkl.dump(dots, f)

if __name__ == "__main__":
    num_dot_list = [10, 50, 100, 500]
    num_test = 100
    for num_dot in num_dot_list:
        generate_test(num_dot, num_test)