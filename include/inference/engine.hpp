#pragma once
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <thread>
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
// Owns a model and reusable buffers. Returned scores live until the next run.
// One caller at a time per Engine. Separate Engine instances may run concurrently.
class Engine {
public:
    explicit Engine(Model model, std::size_t threads = 1);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    const std::vector<float>& run(const std::vector<float>& input);
private:
    void process(std::size_t worker);
    void worker_loop(std::size_t worker);
    void stop();
    Model model_;
    std::vector<float> hidden_, output_;
    std::size_t thread_count_, batch_ = 0, generation_ = 0, remaining_ = 0;
    const float* input_ = nullptr;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    bool stopping_ = false;
};
// Checked multiplication shared by allocations and binary IO.
std::size_t checked_size(std::size_t rows, std::size_t columns);
} // namespace inference
