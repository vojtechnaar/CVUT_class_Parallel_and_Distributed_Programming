#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Result {
    std::string label;
    std::string map;
    long long timeMs = 0;
    long long dfsCalls = 0;
    long long minCost = 0;
};

static std::string runCommand(const std::string& cmd) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);
    return output;
}

static long long parseLong(const std::string& text, const std::string& key) {
    std::regex pattern(key + R"(\s+(-?\d+))");
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
        return std::stoll(match[1].str());
    }
    return -1;
}

static bool compileIfNeeded(const std::string& source, const std::string& exe, const std::string& compiler) {
    fs::path out = fs::path(exe);
    fs::path src = fs::path(source);

    if (fs::exists(out)) {
        return true;
    }

    std::string includeFlags;
    std::string libFlags;

    for (const std::string& includeDir : {"/opt/homebrew/opt/libomp/include", "/usr/local/opt/libomp/include"}) {
        if (fs::exists(includeDir)) {
            includeFlags += " -I\"" + includeDir + "\"";
        }
    }

    for (const std::string& libDir : {"/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"}) {
        if (fs::exists(libDir)) {
            libFlags += " -L\"" + libDir + "\"";
        }
    }

    std::string cmd = compiler + " -std=c++17 -O2 -Xpreprocessor -fopenmp" + includeFlags + libFlags + " -lomp " +
                      "\"" + src.string() + "\" -o \"" + out.string() + "\"";

    std::string result = runCommand(cmd);
    (void)result;
    return fs::exists(out);
}

static Result benchmarkSolver(const std::string& solverExe, const std::string& solverName, const std::string& mapFile) {
    std::string cmd = "\"" + solverExe + "\" \"" + mapFile + "\" 2>&1";
    std::string output = runCommand(cmd);

    Result r;
    r.label = solverName;
    r.map = mapFile;
    r.timeMs = parseLong(output, "TIME_MS");
    r.dfsCalls = parseLong(output, "DFS_CALLS");
    r.minCost = parseLong(output, "MIN_COST");
    return r;
}

int main() {
    const std::vector<std::string> maps = {
        "maps/mapa3_5.txt",
        "maps/mapa5_11.txt",
        "maps/mapa7_7.txt",
        "maps/mapa7_10.txt"
    };

    const std::string repoRoot = fs::current_path().string();
    const std::string buildDir = repoRoot + "/.bench_build";
    fs::create_directories(buildDir);

    const std::string compiler = "clang++";
    const std::vector<std::pair<std::string, std::string>> solvers = {
        {"Sequential", repoRoot + "/SeqSolution/naarvojt-PDP-26-1.cpp"},
        {"Task", repoRoot + "/TaskParallelSolution/TaskParallelism.cpp"},
        {"Data", repoRoot + "/DataParallelSolution/DataParallelism.cpp"}
    };

    std::vector<std::pair<std::string, std::string>> built;
    for (const auto& [name, source] : solvers) {
        std::string exe = buildDir + "/" + name + "_solver";
        if (compileIfNeeded(source, exe, compiler)) {
            built.push_back({name, exe});
        }
    }

    std::cout << "Benchmarking solvers across a few maps\n\n";

    for (const auto& map : maps) {
        std::cout << "Map: " << map << "\n";
        std::vector<Result> results;
        for (const auto& [name, exe] : built) {
            Result r = benchmarkSolver(exe, name, map);
            results.push_back(r);
        }

        if (results.empty()) {
            std::cout << "  No compiled solvers available.\n\n";
            continue;
        }

        const auto seqIt = std::find_if(results.begin(), results.end(),
            [](const Result& r) { return r.label == "Sequential"; });

        if (seqIt != results.end() && seqIt->timeMs > 0) {
            for (const auto& r : results) {
                double speedup = r.timeMs > 0 ? (double)seqIt->timeMs / (double)r.timeMs : 0.0;
                std::cout << "  " << std::setw(8) << std::left << r.label
                          << "  time=" << std::setw(6) << r.timeMs << " ms"
                          << "  minCost=" << r.minCost
                          << "  dfsCalls=" << r.dfsCalls
                          << "  speedup=" << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        } else {
            for (const auto& r : results) {
                std::cout << "  " << std::setw(8) << std::left << r.label
                          << "  time=" << std::setw(6) << r.timeMs << " ms"
                          << "  minCost=" << r.minCost
                          << "  dfsCalls=" << r.dfsCalls << "\n";
            }
        }

        std::cout << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
