#include "inference/io.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if ((argc < 5 || argc > 7) || std::string(argv[1]) != "infer") {
            std::cerr << "Usage: inference_cli infer MODEL INPUTS OUTPUT [baseline|optimized] [threads]\n";
            return 2;
        }
        const auto model = inference::load_model(argv[2]);
        const auto inputs = inference::load_matrix(argv[3]);
        if (inputs.columns != model.first().inputs) throw std::invalid_argument("input columns do not match model");
        const std::string mode = argc >= 6 ? argv[5] : "baseline";
        if (mode != "baseline" && mode != "optimized") throw std::invalid_argument("unknown inference mode");
        std::size_t threads = 1;
        if (argc == 7) {
            const std::string value = argv[6];
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("threads must be an integer");
            threads = std::stoul(value);
        }
        if (mode == "baseline" && threads != 1) throw std::invalid_argument("baseline is single-threaded");
        inference::Engine engine(model, threads);
        const auto scores = mode == "baseline" ? inference::baseline(model, inputs.values) : engine.run(inputs.values);
        inference::save_matrix(argv[4], {inputs.rows, model.second().outputs, scores});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
