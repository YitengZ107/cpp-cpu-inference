#include "inference/engine.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inference {
std::size_t checked_size(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::overflow_error("tensor dimensions overflow size_t");
    return rows * columns;
}
namespace {
void validate(const Dense& layer) {
    if (layer.inputs == 0 || layer.outputs == 0)
        throw std::invalid_argument("dense dimensions must be positive");
    if (layer.weights.size() != checked_size(layer.inputs, layer.outputs) ||
        layer.bias.size() != layer.outputs)
        throw std::invalid_argument("dense weight or bias dimensions do not match");
    for (const auto* buffer : {&layer.weights, &layer.bias})
        for (float value : *buffer)
            if (!std::isfinite(value)) throw std::invalid_argument("non-finite model parameter");
}
void dense_baseline(const Dense& layer, const float* input, float* output, bool relu) {
    for (std::size_t o = 0; o < layer.outputs; ++o) {
        float value = layer.bias[o];
        for (std::size_t i = 0; i < layer.inputs; ++i)
            value += input[i] * layer.weights[i * layer.outputs + o];
        output[o] = relu ? std::max(0.0f, value) : value;
    }
}
} // namespace
Model::Model(Dense first, Dense second) : first_(std::move(first)), second_(std::move(second)) {
    validate(first_);
    validate(second_);
    if (first_.outputs != second_.inputs)
        throw std::invalid_argument("hidden dimensions do not match");
}
std::vector<float> baseline(const Model& model, const std::vector<float>& input) {
    const auto& first = model.first();
    const auto& second = model.second();
    if (input.size() % first.inputs != 0)
        throw std::invalid_argument("input must contain whole samples");
    for (float value : input)
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite input");
    const auto batch = input.size() / first.inputs;
    std::vector<float> output(checked_size(batch, second.outputs));
    for (std::size_t row = 0; row < batch; ++row) {
        std::vector<float> hidden(first.outputs);
        dense_baseline(first, input.data() + row * first.inputs, hidden.data(), true);
        dense_baseline(second, hidden.data(), output.data() + row * second.outputs, false);
    }
    return output;
}
} // namespace inference
