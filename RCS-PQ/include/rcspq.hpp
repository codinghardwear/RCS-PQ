#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rcspq {

enum class Metric { L2, InnerProduct, Cosine };

struct Config {
    std::size_t subdim = 16;
    std::size_t num_centroids = 256;
    double train_ratio = 0.0256;
    std::uint32_t sample_seed = 20261111U;
    std::uint64_t kmeans_seed = 20261111ULL;
    std::size_t kmeans_iterations = 15;
    std::size_t threads = 0;       // 0: OpenMP runtime default
    std::size_t batch_size = 262144;
    Metric metric = Metric::L2;
};

struct Result {
    std::size_t num_points = 0;
    std::size_t dimension = 0;
    std::size_t num_chunks = 0;
    std::size_t train_points = 0;
    double sampling_seconds = 0.0;
    double training_seconds = 0.0;
    double encoding_seconds = 0.0;
    double total_seconds = 0.0;
    std::string simd_backend;
};

// Reads a DiskANN-style float32 .bin matrix (uint32 rows, uint32 cols, row-major floats),
// trains the RCS-PQ codebooks, and writes pivots, centering vector, chunk offsets, and codes.
Result build(const std::string &input_path, const std::string &output_dir, const Config &config);

const char *simd_backend();
const char *metric_name(Metric metric);

} // namespace rcspq
