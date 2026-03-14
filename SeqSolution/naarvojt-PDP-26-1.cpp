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

// Cost of the current partial solution and the best one found so far
long currentCost = 0;
long bestCost = INT_MAX;

// How many T and Z pieces are currently used
int countT = 0;
int countZ = 0;

// Counters used to create labels like T1, T2, Z1...
int nextTId = 1;
int nextZId = 1;

// Number of DFS calls
long long dfsCalls = 0;

// Input values and current board state
vector<int> values;         // value of each cell
vector<int> state;          // 0 = undecided, -1 = uncovered, 1 = covered
vector<string> labels;      // current labels written in covered cells
vector<string> bestOutput;  // best final board found

// All unique rotations/reflections of T and Z
vector<Shape> tShapes;
vector<Shape> zShapes;

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

// Find the first cell that has not been decided yet
int findFirstUndecidedCell() {
    for (int i = 0; i < totalCells; i++) {
        if (state[i] == 0) return i;
    }
    return -1;
}

// Check if it is still possible to finish with |T - Z| <= 1
bool canStillFixTZDifference(int remainingUndecided) {
    int difference = countT - countZ;
    int maxMorePieces = remainingUndecided / 4;

    int minDiff = -maxMorePieces - 1;
    int maxDiff =  maxMorePieces + 1;

    return difference >= minDiff && difference <= maxDiff;
}

// ===================== PLACEMENT GENERATION =====================

// Generate all ways to place a piece so that it covers the chosen cell
void generatePlacementsAtCell(int pivot, char pieceType, const vector<Shape>& shapes, vector<Placement>& out) {

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
                if (!insideBoard(r, c) || state[cellIndex(r, c)] != 0) {
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

// Save the current board as the best answer
void saveBestBoard() {
    bestOutput.resize(totalCells);

    for (int i = 0; i < totalCells; i++) {
        if (state[i] == -1) bestOutput[i] = to_string(values[i]);
        else bestOutput[i] = labels[i];
    }
}

// Place a tetromino on the board
void placePiece(Placement& placement) {
    placement.id = (placement.type == 'T' ? nextTId++ : nextZId++);
    string pieceLabel(1, placement.type);
    pieceLabel += to_string(placement.id);

    for (int cell : placement.cells) {
        state[cell] = 1;
        labels[cell] = pieceLabel;
    }

    if (placement.type == 'T') countT++;
    else countZ++;
}

// Undo placing one piece
void removePiece(const Placement& placement) {
    for (int cell : placement.cells) {
        state[cell] = 0;
        labels[cell].clear();
    }

    if (placement.type == 'T') {
        countT--;
        nextTId--;
    } else {
        countZ--;
        nextZId--;
    }
}

// Mark a cell uncovered
void markUncovered(int cell) {
    state[cell] = -1;
    currentCost += values[cell];
}

// Undo uncovered decision
void undoMarkUncovered(int cell) {
    state[cell] = 0;
    currentCost -= values[cell];
}

// ===================== MAIN DFS SEARCH =====================

// Main recursive search
void dfs() {
    dfsCalls++;

    // Stop if this branch is already worse than the best answer
    if (currentCost >= bestCost) return;

    int firstCell = findFirstUndecidedCell();

    // If there is no undecided cell left, we have a complete solution
    if (firstCell == -1) {
        if (abs(countT - countZ) <= 1 && currentCost < bestCost) {
            bestCost = currentCost;
            saveBestBoard();
        }
        return;
    }

    // Count how many cells are still undecided
    int remainingUndecided = 0;
    for (int x : state) {
        if (x == 0) remainingUndecided++;
    }

    // If the T/Z balance can no longer be fixed, stop
    if (!canStillFixTZDifference(remainingUndecided)) return;

    // Build all candidate placements that cover the chosen cell
    vector<Placement> placements;
    placements.reserve(64);
    generatePlacementsAtCell(firstCell, 'T', tShapes, placements);
    generatePlacementsAtCell(firstCell, 'Z', zShapes, placements);

    // Try placing a tetromino
    for (auto& placement : placements) {
        placePiece(placement);

        if (canStillFixTZDifference(remainingUndecided - 4)) {
        dfs();
        }
        
        removePiece(placement);
    }

    // Try leaving the chosen cell uncovered
    markUncovered(firstCell);
    dfs();
    undoMarkUncovered(firstCell);
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

    // Start with an empty undecided board
    state.assign(totalCells, 0);
    labels.assign(totalCells, "");

    // Timing
    auto startTime = chrono::high_resolution_clock::now();

    // Run the recursive search
    dfs();

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