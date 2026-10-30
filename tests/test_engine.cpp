#include "inference/engine.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected invalid argument to be rejected");
}
inference::Model example() {
    // h = relu([x0+2*x1-1, -x0+x1]); y = 3*h0-2*h1+0.5
    return inference::Model({2, 2, {1, -1, 2, 1}, {-1, 0}}, {2, 1, {3, -2}, {0.5f}});
}
}
int main() {
    try {
        const auto model = example();
        const auto scores = inference::baseline(model, {1, 2, -1, -2, 0, 0});
        require(scores == std::vector<float>({10.5f, 0.5f, 0.5f}), "hand-computed ReLU/batch result");
        require(inference::baseline(model, {}).empty(), "empty batch");
        rejects([&] { inference::baseline(model, {1}); });
        rejects([&] { inference::baseline(model, {std::numeric_limits<float>::quiet_NaN(), 1}); });
        rejects([&] { inference::baseline(model, {std::numeric_limits<float>::infinity(), 1}); });
        rejects([] { inference::Model({0, 2, {}, {0, 0}}, {2, 1, {0, 0}, {0}}); });
        rejects([] { inference::Model({2, 2, {0}, {0, 0}}, {2, 1, {0, 0}, {0}}); });
        rejects([] { inference::Model({2, 2, {0, 0, 0, 0}, {0}}, {2, 1, {0, 0}, {0}}); });
        rejects([] { inference::Model({1, 1, {0}, {0}}, {2, 1, {0, 0}, {0}}); });
        rejects([] { inference::Model({1, 1, {std::numeric_limits<float>::infinity()}, {0}}, {1, 1, {0}, {0}}); });
        rejects([] { inference::checked_size(std::numeric_limits<std::size_t>::max(), 2); });
        inference::Model zero({1, 1, {0}, {-1}}, {1, 1, {100}, {2}});
        require(inference::baseline(zero, {123})[0] == 2, "negative hidden activation must be zero");
        inference::Engine engine(model);
        const std::vector<float> input{1, 2, -1, -2, 0, 0};
        require(engine.run(input) == scores, "optimized hand-computed result");
        const auto* storage = engine.run(input).data();
        require(engine.run(input).data() == storage, "same-size runs must reuse output storage");
        require(engine.run({}).empty(), "optimized empty batch");
        require(engine.run(input) == scores, "grow after empty batch");
        rejects([&] { engine.run({1}); });
        rejects([&] { engine.run({1, std::numeric_limits<float>::quiet_NaN()}); });
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (const std::size_t outputs : {1u, 3u, 10u, 64u}) {
            inference::Dense a{7, outputs, std::vector<float>(7 * outputs), std::vector<float>(outputs)};
            inference::Dense b{outputs, 3, std::vector<float>(outputs * 3), std::vector<float>(3)};
            for (auto* buffer : {&a.weights, &a.bias, &b.weights, &b.bias})
                for (float& value : *buffer) value = distribution(rng);
            inference::Model random_model(a, b);
            inference::Engine random_engine(random_model);
            std::vector<float> random_inputs(7 * 19);
            for (float& value : random_inputs) value = distribution(rng);
            const auto expected = inference::baseline(random_model, random_inputs);
            const auto actual = random_engine.run(random_inputs);
            for (std::size_t i = 0; i < expected.size(); ++i)
                require(std::abs(expected[i] - actual[i]) < 1e-5f, "optimized randomized parity including vector tails");
        }
        std::cout << "All engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
