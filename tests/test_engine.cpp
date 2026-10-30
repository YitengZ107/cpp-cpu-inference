#include "inference/engine.hpp"
#include <cmath>
#include <iostream>
#include <limits>
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
        std::cout << "All engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
