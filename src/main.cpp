#include "inference/io.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc != 5 || std::string(argv[1]) != "infer") {
            std::cerr << "Usage: inference_cli infer MODEL INPUTS OUTPUT\n";
            return 2;
        }
        const auto model = inference::load_model(argv[2]);
        const auto inputs = inference::load_matrix(argv[3]);
        if (inputs.columns != model.first().inputs) throw std::invalid_argument("input columns do not match model");
        const auto scores = inference::baseline(model, inputs.values);
        inference::save_matrix(argv[4], {inputs.rows, model.second().outputs, scores});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
