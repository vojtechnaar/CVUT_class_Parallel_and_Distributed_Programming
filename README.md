# Parallel Tetromino Solver

A comparison of sequential and OpenMP-based parallel strategies for solving the tetromino covering optimization problem.

This repository contains three implementations of the same branch-and-bound solver:

- a sequential baseline
- an OpenMP task-parallel version
- an OpenMP data-parallel version

The project was developed as part of the Parallel and Distributed Programming course at Czech Technical University in Prague.

## Problem definition

The objective is to cover a rectangular board with T- and Z-shaped tetrominoes while minimizing the total value of uncovered cells.

Constraints:

- tetrominoes cannot overlap
- cells may remain uncovered, but their values contribute to the objective
- the difference between the number of T and Z pieces is at most 1
- the solver searches for the minimum uncovered cost

## Why this project is interesting

This problem is computationally expensive because the search space grows very quickly. The project focuses on:

- branch-and-bound pruning
- feasibility checks for partial states
- parallel exploration of independent subproblems
- comparison between task-parallel and data-parallel approaches

## Repository structure

```text
.
├── SeqSolution/
│   └── naarvojt-PDP-26-1.cpp
├── TaskParallelSolution/
│   └── TaskParallelism.cpp
├── DataParallelSolution/
│   └── DataParallelism.cpp
├── maps/
│   ├── mapa3_5.txt
│   ├── mapa5_11.txt
│   └── ...
├── README.md
├── readme.txt
├── .gitignore
└── .DS_Store (ignored locally)
```

## Algorithms

### Sequential solver
The baseline implementation explores the search tree using depth-first search with pruning. It keeps track of the best known solution and avoids exploring branches that cannot improve the current optimum.

### Task-parallel solver
The task-parallel version uses OpenMP tasks to explore independent DFS branches concurrently. This approach is useful when the search tree naturally splits into many independent subproblems.

### Data-parallel solver
The data-parallel version builds a frontier of partial states and then distributes those states across threads using OpenMP work-sharing loops. This pattern is useful for comparing a more structured parallel decomposition against task-driven parallelism.

## Build

This project uses OpenMP, so it requires an OpenMP-capable compiler.

### Requirements

- C++17 compiler
- OpenMP support

### macOS

If you are using Apple Clang, OpenMP support may require `libomp`:

```bash
clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp -lomp SeqSolution/naarvojt-PDP-26-1.cpp -o seq_solver
clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp -lomp TaskParallelSolution/TaskParallelism.cpp -o task_solver
clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp -lomp DataParallelSolution/DataParallelism.cpp -o data_solver
```

### Linux / GCC

```bash
g++ -std=c++17 -O2 -fopenmp SeqSolution/naarvojt-PDP-26-1.cpp -o seq_solver
g++ -std=c++17 -O2 -fopenmp TaskParallelSolution/TaskParallelism.cpp -o task_solver
g++ -std=c++17 -O2 -fopenmp DataParallelSolution/DataParallelism.cpp -o data_solver
```

### Run

```bash
./seq_solver maps/mapa5_11.txt
./task_solver maps/mapa5_11.txt
./data_solver maps/mapa5_11.txt
```

## Output

Each solver prints:

- the final board state
- minimum uncovered cost
- number of DFS calls
- execution time in milliseconds

## Benchmark results

The following benchmark results were measured on the included sample maps using the project benchmark runner.

| Map | Sequential (ms) | Task-parallel (ms) | Data-parallel (ms) | Best speedup |
|---|---:|---:|---:|---:|
| mapa3_5.txt | 0 | 2 | 1 | 0.0x |
| mapa5_11.txt | 1479 | 757 | 861 | 1.95x |
| mapa7_7.txt | 12 | 33 | 9 | 1.33x |
| mapa7_10.txt | 3460 | 1809 | 1565 | 2.21x |

These results show that the parallel strategies can reduce runtime significantly on larger instances, while preserving the same optimal cost.

> Raw benchmark output is also available in [MapsComparison/Results.txt](MapsComparison/Results.txt).

## Project status

This repository is primarily an academic and comparative parallel programming project. It is intended to demonstrate algorithmic design, pruning strategies, and parallel execution patterns rather than a production-ready software package.

## Future work

- add a proper MPI-based distributed implementation
- unify shared solver logic into a common core
- improve benchmarking and result reporting
- add a cleaner CLI interface and configuration options

## Author

Vojtech Naar

## Course

Parallel and Distributed Programming (NI-PDP), Czech Technical University in Prague
