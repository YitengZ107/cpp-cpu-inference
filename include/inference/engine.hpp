#pragma once
#include <cstddef>
#include <vector>

namespace inference {
// Owning contiguous float buffers. Weight (input i, output o) is weights[i*out+o].
struct Dense {
    std::size_t inputs, outputs;
    std::vector<float> weights, bias;
};

class Model {
public:
    Model(Dense first, Dense second);
    const Dense& first() const noexcept { return first_; }
    const Dense& second() const noexcept { return second_; }
private:
    Dense first_, second_;
};

// Deliberately simple baseline: strided dot products, fresh storage per sample.
std::vector<float> baseline(const Model& model, const std::vector<float>& input);
// Checked multiplication shared by allocations and binary IO.
std::size_t checked_size(std::size_t rows, std::size_t columns);
} // namespace inference
