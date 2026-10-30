#include "inference/io.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace inference {
namespace {
constexpr std::size_t max_bytes = 256 * 1024 * 1024;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
std::vector<unsigned char> read(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path);
    auto size = file.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > max_bytes)
        throw std::runtime_error("file exceeds 256 MiB limit");
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!data.empty() && !file.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("cannot read " + path);
    return data;
}
std::uint32_t u32(const unsigned char* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
float f32(const unsigned char* p) {
    auto bits = u32(p);
    float value;
    std::memcpy(&value, &bits, 4);
    if (!std::isfinite(value)) throw std::runtime_error("non-finite file value");
    return value;
}
void write_u32(std::ostream& file, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) file.put(static_cast<char>((value >> shift) & 255));
}
void dimension(std::size_t value) {
    if (value == 0 || value > 1000000) throw std::runtime_error("invalid dimension");
}
void exact_payload(const std::vector<unsigned char>& data, std::size_t header, std::size_t floats) {
    if (floats > (max_bytes - header) / 4 || data.size() != header + floats * 4)
        throw std::runtime_error("file size does not match dimensions");
}
}
Model load_model(const std::string& path) {
    const auto data = read(path);
    if (data.size() < 20 || std::memcmp(data.data(), "DMLP0001", 8) != 0)
        throw std::runtime_error("bad model header");
    std::size_t in = u32(data.data() + 8), hidden = u32(data.data() + 12), out = u32(data.data() + 16);
    dimension(in); dimension(hidden); dimension(out);
    auto first_count = checked_size(in, hidden), second_count = checked_size(hidden, out);
    // Dimensions are capped, so addition is safe on 64-bit; bound each term before sum for 32-bit.
    if (first_count > max_bytes / 4 || second_count > max_bytes / 4)
        throw std::runtime_error("model too large");
    exact_payload(data, 20, first_count + hidden + second_count + out);
    std::size_t offset = 20;
    auto layer = [&](std::size_t inputs, std::size_t outputs) {
        Dense result{inputs, outputs, std::vector<float>(checked_size(inputs, outputs)), std::vector<float>(outputs)};
        for (std::size_t o = 0; o < outputs; ++o)
            for (std::size_t i = 0; i < inputs; ++i, offset += 4)
                result.weights[i * outputs + o] = f32(data.data() + offset);
        for (float& value : result.bias) { value = f32(data.data() + offset); offset += 4; }
        return result;
    };
    auto first = layer(in, hidden);
    auto second = layer(hidden, out);
    return Model(std::move(first), std::move(second));
}
Matrix load_matrix(const std::string& path) {
    const auto data = read(path);
    if (data.size() < 16 || std::memcmp(data.data(), "DMLPMAT1", 8) != 0)
        throw std::runtime_error("bad matrix header");
    std::size_t rows = u32(data.data() + 8), columns = u32(data.data() + 12);
    dimension(columns);
    exact_payload(data, 16, checked_size(rows, columns));
    Matrix matrix{rows, columns, std::vector<float>(rows * columns)};
    for (std::size_t i = 0; i < matrix.values.size(); ++i) matrix.values[i] = f32(data.data() + 16 + i * 4);
    return matrix;
}
void save_matrix(const std::string& path, const Matrix& matrix) {
    dimension(matrix.columns);
    if (matrix.rows > std::numeric_limits<std::uint32_t>::max() ||
        matrix.values.size() != checked_size(matrix.rows, matrix.columns) ||
        matrix.values.size() > (max_bytes - 16) / 4)
        throw std::invalid_argument("invalid output matrix");
    for (float value : matrix.values)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite inference output");
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot write " + path);
    file.write("DMLPMAT1", 8);
    write_u32(file, static_cast<std::uint32_t>(matrix.rows));
    write_u32(file, static_cast<std::uint32_t>(matrix.columns));
    for (float value : matrix.values) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, 4);
        write_u32(file, bits);
    }
    if (!file) throw std::runtime_error("output write failed");
}
}
