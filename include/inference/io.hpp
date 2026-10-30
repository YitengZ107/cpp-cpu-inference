#pragma once
#include "inference/engine.hpp"
#include <string>
namespace inference {
struct Matrix {
    std::size_t rows, columns;
    std::vector<float> values;
};
Model load_model(const std::string& path);
Matrix load_matrix(const std::string& path);
void save_matrix(const std::string& path, const Matrix& matrix);
}
