Quatromino Covering Solver – ČVUT NI-PDP

This project implements algorithms to solve the SQM problem: covering a rectangular board with T and Z quatrominoes while minimizing the sum of uncovered cell values. It was developed as a semester project for the Parallel and Distributed Programming (NI-PDP) course at ČVUT.

Problem:

- Input: board A × B with integer values in each cell.
- Pieces: T and Z tetrominoes (4 cells), can be rotated and mirrored.
- Constraints: 
  - Pieces do not overlap.  
  - Additional pieces cannot be placed.  
  - The number of T and Z pieces differs by at most 1.
- Objective: minimize the sum of values of uncovered cells.

Implemented Algorithms:

1. Sequential DFS (Branch & Bound)
   - Depth-First Search with backtracking.
   - Branch on all valid placements of T/Z or leaving a cell uncovered.
   - Pruning: stop branches exceeding current best cost or impossible parity.

2. OpenMP Task Parallelism
   - DFS branches are executed as parallel tasks on multiple CPU cores.
   - Tasks are scheduled dynamically to balance workload.

3. OpenMP Data Parallelism
   - Computational loops (candidate generation, heuristic scoring) are parallelized using OpenMP parallel for.

4. MPI Distributed Solver
   - Master process generates initial DFS branches.
   - Worker processes explore assigned branches and return best solutions.
   - Uses MPI communication (MPI_Send, MPI_Recv, MPI_Reduce).

Performance:

- Measures sequential and parallel runtimes.
- Computes minimum uncovered cost and DFS call count.
- Evaluates speedup and scalability across multiple cores or processes.


Author:

Vojtech Naar – CVUT NI-PDP course project
