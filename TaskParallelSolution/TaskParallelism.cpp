#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

// ===================== DATA STRUCTURES =====================

struct Placement {
    char type;         // 'T' or 'Z'
    int id;            // piece id used for label (T1, Z2, ...)
    vector<int> cells; // covered cells
};

using Offset = pair<int, int>;
using Shape = array<Offset, 4>;

struct SearchNode {
    long currentCost = 0;
    int countT = 0;
    int countZ = 0;
    int nextTId = 1;
    int nextZId = 1;
    int remainingUndecided = 0;
    vector<int> state;     // 0 = undecided, -1 = uncovered, 1 = covered
    vector<string> labels; // piece labels for covered cells
};

// ===================== GLOBAL READ-ONLY PROBLEM DATA =====================

int rows = 0;
int cols = 0;
int totalCells = 0;
vector<int> values;
vector<Shape> tShapes;
vector<Shape> zShapes;

// ===================== BASIC BOARD HELPERS =====================

int cellIndex(int r, int c) {
    return r * cols + c;
}

bool insideBoard(int r, int c) {
    return 0 <= r && r < rows && 0 <= c && c < cols;
}

// ===================== SHAPE MANIPULATION =====================

Shape normalizeShape(Shape shape) {
    int minRow = INT_MAX;
    int minCol = INT_MAX;
    for (const auto& cell : shape) {
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

Shape rotate90(Shape shape) {
    for (auto& cell : shape) {
        cell = {cell.second, -cell.first};
    }
    return shape;
}

Shape reflectShape(Shape shape) {
    for (auto& cell : shape) {
        cell = {cell.first, -cell.second};
    }
    return shape;
}

vector<Shape> buildAllOrientations(Shape baseShape) {
    set<vector<Offset>> seen;
    vector<Shape> result;
    for (int mirror = 0; mirror < 2; mirror++) {
        Shape currentShape = (mirror == 0 ? baseShape : reflectShape(baseShape));
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

int findFirstUndecidedCell(const SearchNode& node) {
    for (int i = 0; i < totalCells; i++) {
        if (node.state[i] == 0) return i;
    }
    return -1;
}

bool canStillFixTZDifference(int countT, int countZ, int remainingUndecided) {
    int difference = countT - countZ;
    int maxMorePieces = remainingUndecided / 4;
    int minDiff = -maxMorePieces - 1;
    int maxDiff = maxMorePieces + 1;
    return difference >= minDiff && difference <= maxDiff;
}

void saveBoard(const SearchNode& node, vector<string>& output) {
    output.resize(totalCells);
    for (int i = 0; i < totalCells; i++) {
        output[i] = (node.state[i] == -1 ? to_string(values[i]) : node.labels[i]);
    }
}

// ===================== PLACEMENT GENERATION =====================

void generatePlacementsAtCell(const SearchNode& node,
                              int pivot,
                              char pieceType,
                              const vector<Shape>& shapes,
                              vector<Placement>& out) {
    int row = pivot / cols;
    int col = pivot % cols;

    for (const auto& shape : shapes) {
        for (int anchor = 0; anchor < 4; anchor++) {
            int startRow = row - shape[anchor].first;
            int startCol = col - shape[anchor].second;
            vector<int> coveredCells;
            coveredCells.reserve(4);
            bool ok = true;

            for (int i = 0; i < 4; i++) {
                int r = startRow + shape[i].first;
                int c = startCol + shape[i].second;
                int idx = cellIndex(r, c);
                if (!insideBoard(r, c) || node.state[idx] != 0) {
                    ok = false;
                    break;
                }
                coveredCells.push_back(idx);
            }

            if (ok) out.push_back({pieceType, -1, coveredCells});
        }
    }
}

// ===================== STATE OPERATIONS =====================

void placePiece(SearchNode& node, Placement& placement) {
    placement.id = (placement.type == 'T' ? node.nextTId++ : node.nextZId++);
    string pieceLabel(1, placement.type);
    pieceLabel += to_string(placement.id);

    for (int cell : placement.cells) {
        node.state[cell] = 1;
        node.labels[cell] = pieceLabel;
    }
    if (placement.type == 'T') node.countT++;
    else node.countZ++;
    node.remainingUndecided -= 4;
}

void removePiece(SearchNode& node, const Placement& placement) {
    for (int cell : placement.cells) {
        node.state[cell] = 0;
        node.labels[cell].clear();
    }
    if (placement.type == 'T') {
        node.countT--;
        node.nextTId--;
    } else {
        node.countZ--;
        node.nextZId--;
    }
    node.remainingUndecided += 4;
}

void markUncovered(SearchNode& node, int cell) {
    node.state[cell] = -1;
    node.currentCost += values[cell];
    node.remainingUndecided--;
}

void undoMarkUncovered(SearchNode& node, int cell) {
    node.state[cell] = 0;
    node.currentCost -= values[cell];
    node.remainingUndecided++;
}

// ===================== DFS USED INSIDE TASKS =====================

void dfsFromFrontier(SearchNode& node,
                     long& localBestCost,
                     vector<string>& localBestOutput,
                     const atomic<long>& sharedBestCost,
                     long long& localDfsCalls) {
    localDfsCalls++;

    long sharedBound = sharedBestCost.load(memory_order_relaxed);
    long pruneBound = min(localBestCost, sharedBound);
    if (node.currentCost >= pruneBound) return;

    int firstCell = findFirstUndecidedCell(node);
    if (firstCell == -1) {
        if (abs(node.countT - node.countZ) <= 1 && node.currentCost < localBestCost) {
            localBestCost = node.currentCost;
            saveBoard(node, localBestOutput);
        }
        return;
    }

    if (!canStillFixTZDifference(node.countT, node.countZ, node.remainingUndecided)) return;

    vector<Placement> placements;
    placements.reserve(64);
    generatePlacementsAtCell(node, firstCell, 'T', tShapes, placements);
    generatePlacementsAtCell(node, firstCell, 'Z', zShapes, placements);

    for (auto& placement : placements) {
        placePiece(node, placement);
        if (canStillFixTZDifference(node.countT, node.countZ, node.remainingUndecided)) {
            dfsFromFrontier(node, localBestCost, localBestOutput, sharedBestCost, localDfsCalls);
        }
        removePiece(node, placement);
    }

    markUncovered(node, firstCell);
    dfsFromFrontier(node, localBestCost, localBestOutput, sharedBestCost, localDfsCalls);
    undoMarkUncovered(node, firstCell);
}

// ===================== SHORT BFS SPLIT PHASE =====================

void buildFrontierWithShortBfs(vector<SearchNode>& frontier,
                               int maxBfsDepth,
                               int targetTaskCount,
                               atomic<long>& bestCost,
                               vector<string>& bestOutput,
                               long long& bfsCalls) {
    for (int depth = 0; depth < maxBfsDepth; depth++) {
        vector<SearchNode> next;
        next.reserve(frontier.size() * 8);

        for (const SearchNode& node : frontier) {
            bfsCalls++;

            if (node.currentCost >= bestCost.load(memory_order_relaxed)) continue;

            int firstCell = findFirstUndecidedCell(node);
            if (firstCell == -1) {
                if (abs(node.countT - node.countZ) <= 1 &&
                    node.currentCost < bestCost.load(memory_order_relaxed)) {
                    bestCost.store(node.currentCost, memory_order_relaxed);
                    saveBoard(node, bestOutput);
                }
                continue;
            }

            if (!canStillFixTZDifference(node.countT, node.countZ, node.remainingUndecided)) continue;

            vector<Placement> placements;
            placements.reserve(64);
            generatePlacementsAtCell(node, firstCell, 'T', tShapes, placements);
            generatePlacementsAtCell(node, firstCell, 'Z', zShapes, placements);

            for (const Placement& basePlacement : placements) {
                SearchNode child = node;
                Placement placement = basePlacement;
                placePiece(child, placement);
                if (canStillFixTZDifference(child.countT, child.countZ, child.remainingUndecided)) {
                    next.push_back(std::move(child));
                }
            }

            SearchNode uncovered = node;
            markUncovered(uncovered, firstCell);
            next.push_back(std::move(uncovered));
        }

        if (next.empty()) break;
        frontier.swap(next);
        if (static_cast<int>(frontier.size()) >= targetTaskCount) break;
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

    tShapes = buildAllOrientations({Offset{0, 0}, Offset{0, 1}, Offset{0, 2}, Offset{1, 1}});
    zShapes = buildAllOrientations({Offset{0, 0}, Offset{0, 1}, Offset{1, 1}, Offset{1, 2}});

    SearchNode root;
    root.remainingUndecided = totalCells;
    root.state.assign(totalCells, 0);
    root.labels.assign(totalCells, "");

    atomic<long> bestCost(LONG_MAX);
    vector<string> bestOutput;
    long long dfsCalls = 0;

    auto startTime = chrono::high_resolution_clock::now();

    vector<SearchNode> frontier = {root};
    const int maxThreads = omp_get_max_threads();
    const int bfsDepth = 3;                  // short BFS
    const int targetTaskCount = maxThreads * 8;
    long long bfsCalls = 0;

    buildFrontierWithShortBfs(frontier, bfsDepth, targetTaskCount, bestCost, bestOutput, bfsCalls);
    dfsCalls += bfsCalls;

    if (!frontier.empty()) {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for (size_t i = 0; i < frontier.size(); i++) {
                    SearchNode startNode = frontier[i];
                    #pragma omp task firstprivate(startNode)
                    {
                        long localBestCost = bestCost.load(memory_order_relaxed);
                        vector<string> localBestOutput;
                        long long localDfsCalls = 0;

                        dfsFromFrontier(startNode,
                                        localBestCost,
                                        localBestOutput,
                                        bestCost,
                                        localDfsCalls);

                        #pragma omp atomic
                        dfsCalls += localDfsCalls;

                        if (!localBestOutput.empty()) {
                            #pragma omp critical(best_update)
                            {
                                if (localBestCost < bestCost.load(memory_order_relaxed)) {
                                    bestCost.store(localBestCost, memory_order_relaxed);
                                    bestOutput = localBestOutput;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto endTime = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (c) cout << ' ';
            cout << bestOutput[cellIndex(r, c)];
        }
        cout << "\n";
    }

    cout << "\nMIN_COST " << bestCost.load(memory_order_relaxed) << "\n";
    cout << "DFS_CALLS " << dfsCalls << "\n";
    cout << "TIME_MS " << elapsed.count() << "\n";
    return 0;
}
