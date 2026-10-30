#include "inference/io.hpp"
#include "build_info.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
std::size_t number(const char* argument) {
    const std::string value(argument);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected a nonnegative integer");
    return std::stoul(value);
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 8) {
            std::cerr << "Usage: inference_bench MODEL INPUTS MODE THREADS WARMUP REPEATS TARGET_MS\n";
            return 2;
        }
        auto model = inference::load_model(argv[1]);
        auto inputs = inference::load_matrix(argv[2]);
        const std::string mode(argv[3]);
        const auto threads = number(argv[4]), warmup = number(argv[5]), repeats = number(argv[6]), target_ms = number(argv[7]);
        if ((mode != "baseline" && mode != "optimized") || (mode == "baseline" && threads != 1) ||
            inputs.rows == 0 || inputs.columns != model.first().inputs || warmup < 1 || repeats < 3 ||
            repeats > 1000 || target_ms < 1 || target_ms > 10000)
            throw std::invalid_argument("invalid benchmark configuration");
        inference::Engine engine(model, threads);
        std::vector<float> last_baseline;
        const std::vector<float>* scores = nullptr;
        auto run = [&] {
            if (mode == "baseline") { last_baseline = inference::baseline(model, inputs.values); scores = &last_baseline; }
            else scores = &engine.run(inputs.values);
        };
        for (std::size_t i = 0; i < warmup; ++i) run();
        auto measure = [&](std::size_t iterations) {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < iterations; ++i) run();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        std::size_t iterations = 1;
        while (measure(iterations) * 1000 < target_ms && iterations < 1048576) iterations *= 2;
        std::vector<double> times;
        double checksum = 0;
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            times.push_back(measure(iterations) / iterations);
            checksum += std::accumulate(scores->begin(), scores->end(), 0.0);
        }
        std::cout << std::setprecision(17)
                  << "{\"mode\":" << std::quoted(mode) << ",\"threads\":" << threads
                  << ",\"batch\":" << inputs.rows << ",\"iterations\":" << iterations
                  << ",\"warmup\":" << warmup << ",\"compiler\":" << std::quoted(INFERENCE_COMPILER)
                  << ",\"build_type\":" << std::quoted(INFERENCE_BUILD_TYPE)
                  << ",\"flags\":" << std::quoted(INFERENCE_FLAGS)
                  << ",\"native\":" << std::quoted(INFERENCE_NATIVE)
                  << ",\"checksum\":" << checksum << ",\"seconds_per_call\":[";
        for (std::size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
