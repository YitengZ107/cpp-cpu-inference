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
void dense_contiguous(const Dense& layer, const float* input, float* output, bool relu) {
    std::copy(layer.bias.begin(), layer.bias.end(), output);
    for (std::size_t i = 0; i < layer.inputs; ++i) {
        const float value = input[i];
        const float* weights = layer.weights.data() + i * layer.outputs;
        // Adjacent output lanes are independent; the compiler can vectorize this loop.
        for (std::size_t o = 0; o < layer.outputs; ++o) output[o] += value * weights[o];
    }
    if (relu)
        for (std::size_t o = 0; o < layer.outputs; ++o) output[o] = std::max(0.0f, output[o]);
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
Engine::Engine(Model model, std::size_t threads) : model_(std::move(model)), thread_count_(threads) {
    if (threads == 0 || threads > 256) throw std::invalid_argument("threads must be in [1,256]");
    hidden_.resize(checked_size(model_.first().outputs, threads));
    try {
        for (std::size_t i = 1; i < threads; ++i) workers_.emplace_back(&Engine::worker_loop, this, i);
    } catch (...) { stop(); throw; }
}
Engine::~Engine() { stop(); }
void Engine::stop() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    start_.notify_all();
    for (auto& worker : workers_) if (worker.joinable()) worker.join();
}
void Engine::process(std::size_t worker) {
    const auto& first = model_.first();
    const auto& second = model_.second();
    // Quotient/remainder partition avoids batch*worker overflow.
    const auto quotient = batch_ / thread_count_, remainder = batch_ % thread_count_;
    const auto begin = worker * quotient + std::min(worker, remainder);
    const auto end = begin + quotient + (worker < remainder ? 1 : 0);
    float* hidden = hidden_.data() + worker * first.outputs;
    for (auto row = begin; row < end; ++row) {
        dense_contiguous(first, input_ + row * first.inputs, hidden, true);
        dense_contiguous(second, hidden, output_.data() + row * second.outputs, false);
    }
}
void Engine::worker_loop(std::size_t worker) {
    std::size_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
        lock.unlock();
        process(worker);
        lock.lock();
        if (--remaining_ == 0) done_.notify_one();
    }
}
const std::vector<float>& Engine::run(const std::vector<float>& input) {
    const auto& first = model_.first();
    const auto& second = model_.second();
    if (&input == &output_) throw std::invalid_argument("input cannot alias this engine's output");
    if (input.size() % first.inputs != 0) throw std::invalid_argument("input must contain whole samples");
    for (float value : input)
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite input");
    const auto batch = input.size() / first.inputs;
    output_.resize(checked_size(batch, second.outputs));
    if (batch == 0) return output_;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        input_ = input.data(); batch_ = batch;
        remaining_ = workers_.size();
        ++generation_;
    }
    start_.notify_all();
    process(0); // The calling thread participates.
    if (!workers_.empty()) {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }
    return output_;
}
} // namespace inference
