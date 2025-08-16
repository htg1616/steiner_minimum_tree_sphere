# Steiner Minimum Tree on a Sphere
  **Steiner Minimum Tree on a Sphere** is a Python implementation of a heuristic algorithm for the **Geodesic Steiner Tree Problem (GSTP) on the sphere**. Given a set of terminal points on a sphere, the project finds a short network (graph) connecting all points with minimal total great-circle distance by introducing auxiliary **Steiner points**. This repository provides a complete pipeline to generate random spherical instances, construct an initial Steiner Tree (via MST + iterative Steiner point insertion), and refine it with gradient-based optimization.

## Background and Motivation ##
The **Geodesic Steiner Tree Problem (GSTP)** asks for the shortest network of paths on a surface (here, a sphere) that connects a given set of terminal points. On a plane, the analogous **Euclidean Steiner Tree Problem (ESTP)** is well-known to be NP-hard and has the property that internal Steiner points (extra junctions added to shorten the network) meet at 120° angles. On a sphere, the problem is equally non-trivial: distances are along great-circle arcs and the geometry requires careful adaptation of planar approaches. Notably, the Smith algorithm, widely used for ESTP, is not applicable to GSTP. Consequently, there has been limited practical research on constructing Steiner trees on curved surfaces.

## Results Summary ##
*(Results and analysis are in progress.)*

We are in the process of running comprehensive benchmarks to evaluate the performance of this Steiner tree heuristic on the sphere. The plan is to compare the total network lengths using **Steiner Ratio** — defined as the ratio of the Steiner minimal tree length to the minimum spanning tree (MST) length — and analyze the efficiency of various gradient-based optimizers in terms of final length, convergence behavior, and stability. 

## Installation and Setup ##

To use this project, ensure you have Python 3.x (recommend 3.9+ for compatibility with PyTorch and Geoopt) and `pip` installed. The repository is not yet available as a pip package, so you should clone the GitHub project and install dependencies manually:

```bash
# 1. Clone the repository
git clone https://github.com/htg1616/steiner_minimum_tree_sphere.git
cd steiner_minimum_tree_sphere

# 2. (Optional) Create a virtual environment for isolation
python -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate    # on Windows

# 3. Install required dependencies
pip install -r requirements.txt
```

This will install the necessary packages listed in `requirements.txt`. Key dependencies include:

- PyTorch (for tensor computations and autograd)
- Geoopt (for Riemannian optimization on the sphere)
- NumPy (for basic numeric operations)
- tqdm (for progress bars during experiments)
- pandas (for experiments data)
- Matplotlib (optional, for visualization scripts)
- pytest (optional, if you want to run the test suite)

## Usage Instructions ##

The project is organized to allow easy generation of random test instances and running the full experiment pipeline. All configuration is managed via JSON files in the `config/` directory, and you can modify or override via CLI options.

### 1. Generating Random Instances ###

Before running the Steiner tree algorithm, you may want to generate some random test cases (sets of points on the sphere). Use the `experiments/generate_instance.py`.
```bash
python experiments/generate_instance.py
```
This reads the configuration from `config/generate_test_config.json`. In that config file, you can specify:
- `base_seed`: an integer seed for random generation (ensuring reproducibility).
- `num_dots`: a list of integers, each representing a number of points to generate in a test set (e.g., [10, 50, 100] to create instances of 10 points, 50 points, etc.).
- `num_tests`: how many instances (files) to generate for each number of points.

When you run `generate_instance.py`, it will create random points on the unit sphere (uniformly distributed) for each specified size and save them as pickle files in the `data/inputs/` directory. For example, if `num_dots` includes 50 and `num_tests` is 5, you will get a folder `data/inputs/50 dots/` containing files `001_test.pkl`, `002_test.pkl`, ..., `005_test.pkl` each with 50 random points, along with a `seeds.json` recording the seeds used. These files serve as input instances for the experiments.

### 2. Running the Experiments ###

Once you have some input instances (or you prepared your own set of points in data/inputs), use the main experiment script to run the Steiner Tree algorithm on all instances and record results
```bash
python experiments/experiment.py
```
this will load the experiment parameters from `config/experiment_config.json` and execute the full pipeline for each test file. The steps include constructing the MST, inserting Steiner points, and performing local optimization as described above. For each input `.pkl` file, an output `.json` result will be saved in the corresponding `data/outputs/{n} dots/directory` (mirroring the input structure). The result JSON for each test case includes:
- `mst_length`: the total length of the initial MST (no Steiner points).
- `smt_length`: the total length after Steiner point insertion (before continuous optimization).
- `opt_smt_length`: the total length after local optimization of Steiner points (final result).
- `opt_smt_curve`: the history of the loss (tree length) during the optimization iterations, which can be used to plot convergence.
- `optimization_iterations`: how many iterations the optimizer ran for that case.

You can monitor the progress in the console: the script uses `tqdm` to show a progress bar for each set of instances, and logging outputs summarizing which configuration is being run.

**Experiment Configuration**: The behavior of the Steiner tree algorithm is controlled by `config/experiment_config.json`. Key fields in this configuration include:

- `backend`: `"torch"` or `"geo"` – which optimization backend to use for local refinement.

 - `optimizer_name`: `"adam"`, `"sgd"`, `"radam"`, or `"rsgd"` – the optimizer for the local phase.

- `optimizer_params`: a dictionary of hyperparameters for the optimizer (learning rate, etc.). For example, for Adam you might specify `{ "lr": 1e-3 }`, or for RAdam `{ "lr": 5e-3, "betas": [0.9, 0.9995], ... }`.

- `scheduler_name` and `scheduler_params`: (optional) specify a learning rate scheduler and its parameters. Supported schedulers include `"cosine"` (cosine annealing), `"cosine_hold"` (cosine decay then hold), `"onecycle"` (one-cycle learning rate policy), and `"plateau"` (reduce on plateau). 

- `max_iterations`: maximum iterations for the optimizer (the local optimization loop).

- `tolerance`: a small threshold for gradient norm to decide early convergence.

- `device`: `"cpu"` or `"cuda"` – where to run the computations. You can set this to `"cuda"` if you have a GPU and PyTorch installed with CUDA support, which can speed up the optimization for larger instances. **Note**: GPU support not tested yet

- `insertion_mode`: the strategy for Steiner point insertion. Currently only `"decrease_only"` is supported, which accepts a new Steiner point only if it reduces the total network length.

## License and Citation ##

**License**: To be added. 

**Citation and Acknowledgment**: This project is developed as part of the high school graduation thesis of **Taegyun Hwang** at Daegu Science High School (South Korea). If you use or build upon this code, please acknowledge the author. For any questions or inquiries, you can reach out via email: htg1616@gmail.com.

The algorithms and implementation are built upon well-established concepts in mathematical optimization and computational geometry. In particular, we acknowledge:

- The classical Steiner Tree heuristics (e.g.,“Thompson’s method”)
- The developers of optimization techniques such as Adam (Kingma & Ba, 2014) and the Geoopt library’s Riemannian optimizers (which enable efficient manifold optimization on the sphere).

*Detailed academic references and literature citations will be added in future updates*
