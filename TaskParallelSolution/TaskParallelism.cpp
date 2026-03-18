#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <set>
#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>
#include <omp.h> 
using namespace std;

// ===================== DATA STRUCTURES =====================
// One placed piece on the board
struct Placement {
    char type;              // 'T' or 'Z'
    int id;                 // number used in output, for example T1
    vector<int> cells;      // the 4 cells covered by this piece
};

// A shape is stored as 4 relative coordinates
using Offset = pair<int, int>;
using Shape = array<Offset, 4>;

// ===================== GLOBAL VARIABLES =====================
// Board size
int rows, cols, totalCells;

// Shared best solution found so far
long bestCost = INT_MAX;
vector<string> bestOutput;

// Total DFS calls across all tasks
long long dfsCalls = 0;

// Input values
vector<int> values;         // value of each cell

// All unique rotations/reflections of T and Z
vector<Shape> tShapes;
vector<Shape> zShapes;

// Limit for task creation depth
const int TASK_DEPTH_LIMIT = 2;

// State of one DFS branch - each task gets its own copy
struct SearchState {
    long currentCost = 0;
    int countT = 0;
    int countZ = 0;
    int nextTId = 1;
    int nextZId = 1;
    vector<int> state;          // 0 = undecided, -1 = uncovered, 1 = covered
    vector<string> labels;      // current labels written in covered cells
};

// ===================== BASIC BOARD HELPERS =====================

// Convert (row,col) to linear index
int cellIndex(int r, int c) {
    return r * cols + c;
}

// Check if coordinate lies inside board
bool insideBoard(int r, int c) {
    return 0 <= r && r < rows && 0 <= c && c < cols;
}

// ===================== SHAPE MANIPULATION =====================

// Move shape so its top-left used cell becomes (0,0)
Shape normalizeShape(Shape shape) {
    int minRow = INT_MAX, minCol = INT_MAX;

    for (auto cell : shape) {
        minRow = min(minRow, cell.first);
        minCol = min(minCol, cell.second);
    }

    for (auto& cell : shape) {
        cell.first -= minRow;
        cell.second -= minCol;
    }

    sort(shape.begin(), shape.end());
    return shape;
}

// Rotate shape 90 degrees
Shape rotate90(Shape shape) {
    for (auto& cell : shape) {
        cell = {cell.second, -cell.first};
    }
    return shape;
}

// Mirror shape horizontally
Shape reflectShape(Shape shape) {
    for (auto& cell : shape) {
        cell = {cell.first, -cell.second};
    }
    return shape;
}

// Build all unique rotations and mirror versions of one base shape
vector<Shape> buildAllOrientations(Shape baseShape) {
    set<vector<Offset>> seen;
    vector<Shape> result;

    for (int mirror = 0; mirror < 2; mirror++) {
        Shape currentShape;
        if (mirror == 0) currentShape = baseShape;
        else currentShape = reflectShape(baseShape);

        for (int rotation = 0; rotation < 4; rotation++) {
            Shape normalized = normalizeShape(currentShape);
            vector<Offset> key(normalized.begin(), normalized.end());

            if (seen.insert(key).second) {
                result.push_back(normalized);
            }

            currentShape = rotate90(currentShape);
        }
    }

    return result;
}

// ===================== SEARCH UTILITIES =====================

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Find the first cell that has not been decided yet
int findFirstUndecidedCell(const SearchState& s) {
    for (int i = 0; i < totalCells; i++) {
        if (s.state[i] == 0) return i;
    }
    return -1;
}

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Check if it is still possible to finish with |T - Z| <= 1
bool canStillFixTZDifference(const SearchState& s, int remainingUndecided) {
    int difference = s.countT - s.countZ;
    int maxMorePieces = remainingUndecided / 4;

    int minDiff = -maxMorePieces - 1;
    int maxDiff =  maxMorePieces + 1;

    return difference >= minDiff && difference <= maxDiff;
}

// ===================== PLACEMENT GENERATION =====================

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Generate all ways to place a piece so that it covers the chosen cell
void generatePlacementsAtCell(const SearchState& s, int pivot, char pieceType, const vector<Shape>& shapes, vector<Placement>& out) {

    int row = pivot / cols;
    int col = pivot % cols;

    // Try every orientation of the piece
    for (const auto& shape : shapes) {

        // Try every block of the shape as the pivot anchor
        for (int anchor = 0; anchor < 4; anchor++) {

            int startRow = row - shape[anchor].first;
            int startCol = col - shape[anchor].second;

            vector<int> coveredCells;
            bool ok = true;

            // Check all 4 blocks of the piece
            for (int i = 0; i < 4; i++) {

                int r = startRow + shape[i].first;
                int c = startCol + shape[i].second;

                // Placement must stay inside the board and on undecided cells
                if (!insideBoard(r, c) || s.state[cellIndex(r, c)] != 0) {
                    ok = false;
                    break;
                }

                coveredCells.push_back(cellIndex(r, c));
            }

            // If placement was valid, store it
            if (ok) {
                out.push_back({pieceType, -1, coveredCells});
            }
        }
    }
}

// ===================== BOARD STATE OPERATIONS =====================

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Save the current board as the best answer
void saveBestBoard(const SearchState& s) {
    vector<string> candidate(totalCells);

    for (int i = 0; i < totalCells; i++) {
        if (s.state[i] == -1) candidate[i] = to_string(values[i]);
        else candidate[i] = s.labels[i];
    }

    #pragma omp critical(best_solution_update) // CHANGED FOR TASK PARALLELISM
    {
        if (s.currentCost < bestCost) {
            bestCost = s.currentCost;
            bestOutput = move(candidate);
        }
    }
}

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Place a tetromino on the board
void placePiece(SearchState& s, Placement& placement) {
    placement.id = (placement.type == 'T' ? s.nextTId++ : s.nextZId++);
    string pieceLabel(1, placement.type);
    pieceLabel += to_string(placement.id);

    for (int cell : placement.cells) {
        s.state[cell] = 1;
        s.labels[cell] = pieceLabel;
    }

    if (placement.type == 'T') s.countT++;
    else s.countZ++;
}

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Undo placing one piece
void removePiece(SearchState& s, const Placement& placement) {
    for (int cell : placement.cells) {
        s.state[cell] = 0;
        s.labels[cell].clear();
    }

    if (placement.type == 'T') {
        s.countT--;
        s.nextTId--;
    } else {
        s.countZ--;
        s.nextZId--;
    }
}

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Mark a cell uncovered
void markUncovered(SearchState& s, int cell) {
    s.state[cell] = -1;
    s.currentCost += values[cell];
}

// CHANGED FOR TASK PARALLELISM (added variable into the function)
// Undo uncovered decision
void undoMarkUncovered(SearchState& s, int cell) {
    s.state[cell] = 0;
    s.currentCost -= values[cell];
}

// ===================== MAIN DFS SEARCH =====================

// CHANGED FOR TASK PARALLELISM
// Main recursive search
void dfs(SearchState& s, int depth = 0) {
    #pragma omp atomic
    dfsCalls++;

    // Stop if this branch is already worse than the best answer
    long snapshotBest;
    #pragma omp atomic read
    snapshotBest = bestCost;

    if (s.currentCost >= snapshotBest) return;

    int firstCell = findFirstUndecidedCell(s);

    // If there is no undecided cell left, we have a complete solution
    if (firstCell == -1) {
        if (abs(s.countT - s.countZ) <= 1) {
            long currentBest;
            #pragma omp atomic read
            currentBest = bestCost;

            if (s.currentCost < currentBest) {
                saveBestBoard(s);
            }
        }
        return;
    }

    // Count how many cells are still undecided
    int remainingUndecided = 0;
    for (int x : s.state) {
        if (x == 0) remainingUndecided++;
    }

    // If the T/Z balance can no longer be fixed, stop
    if (!canStillFixTZDifference(s, remainingUndecided)) return;

    // Build all candidate placements that cover the chosen cell
    vector<Placement> placements;
    placements.reserve(64);
    generatePlacementsAtCell(s, firstCell, 'T', tShapes, placements);
    generatePlacementsAtCell(s, firstCell, 'Z', zShapes, placements);

    // CHANGED FOR TASK PARALLELISM
    // In upper levels create OpenMP tasks, deeper in the tree continue sequentially
    if (depth < TASK_DEPTH_LIMIT) {
        #pragma omp taskgroup // wait for child tasks
        {
            // Try placing a tetromino
            for (auto placement : placements) {
                SearchState child = s;
                placePiece(child, placement);

                if (canStillFixTZDifference(child, remainingUndecided - 4)) {
                    #pragma omp task firstprivate(child, depth) // spawn child task
                    {
                        dfs(child, depth + 1);
                    }
                }
            }

            // Try leaving the chosen cell uncovered
            SearchState child = s;
            markUncovered(child, firstCell);

            #pragma omp task firstprivate(child, depth) // spawn child task
            {
                dfs(child, depth + 1);
            }
        }
    } else {
        // Original sequential recursion
        for (auto& placement : placements) {
            placePiece(s, placement);

            if (canStillFixTZDifference(s, remainingUndecided - 4)) {
                dfs(s, depth + 1);
            }

            removePiece(s, placement);
        }

        markUncovered(s, firstCell);
        dfs(s, depth + 1);
        undoMarkUncovered(s, firstCell);
    }
}

// ===================== INPUT =====================

bool readInput(istream& in) {
    if (!(in >> rows >> cols)) return false;

    totalCells = rows * cols;
    values.assign(totalCells, 0);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            in >> values[cellIndex(r, c)];
        }
    }

    return true;
}

// ===================== MAIN =====================

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    istream* input = &cin;
    ifstream file;

    if (argc > 1) {
        file.open(argv[1]);
        if (!file) {
            cerr << "Cannot open file\n";
            return 1;
        }
        input = &file;
    }

    if (!readInput(*input)) {
        cerr << "Invalid input\n";
        return 1;
    }

    // Build all rotations and mirror versions of T and Z
    tShapes = buildAllOrientations({Offset{0,0}, Offset{0,1}, Offset{0,2}, Offset{1,1}});
    zShapes = buildAllOrientations({Offset{0,0}, Offset{0,1}, Offset{1,1}, Offset{1,2}});

    // CHANGED FOR TASK PARALLELISM
    // Create initial search state
    SearchState root;
    root.state.assign(totalCells, 0);
    root.labels.assign(totalCells, "");

    // Timing
    auto startTime = chrono::high_resolution_clock::now();

    // CHANGED FOR TASK PARALLELISM
    // Run DFS in OpenMP task parallel region
    #pragma omp parallel
    {
        #pragma omp single
        {
            dfs(root, 0);
        }
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

    // Print the best board found
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (c) cout << ' ';
            cout << bestOutput[cellIndex(r, c)];
        }
        cout << "\n";
    }

    cout << "\nMIN_COST " << bestCost << "\n";
    cout << "DFS_CALLS " << dfsCalls << "\n";
    cout << "TIME_MS " << elapsed.count() << "\n";
    return 0;
}