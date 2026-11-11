#include "rcspq.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "rcspq_standalone_smoke";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path input = root / "synthetic.bin";
    const std::uint32_t n = 1000, dim = 12;
    {
        std::ofstream out(input, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&n), sizeof(n));
        out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        std::mt19937 gen(7);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(static_cast<std::size_t>(n) * dim);
        for (float &x : data) x = dist(gen);
        out.write(reinterpret_cast<const char *>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }

    rcspq::Config cfg;
    cfg.subdim = 4;
    cfg.num_centroids = 8;
    cfg.train_ratio = 0.7;
    cfg.sample_seed = 1234;
    cfg.kmeans_seed = 5678;
    cfg.kmeans_iterations = 3;
    cfg.threads = 2;
    cfg.batch_size = 113;

    const fs::path output = root / "out";
    const rcspq::Result result = rcspq::build(input.string(), output.string(), cfg);
    check(result.num_points == n, "wrong point count");
    check(result.dimension == dim, "wrong dimension");
    check(result.num_chunks == 3, "wrong chunk count");
    check(result.train_points >= cfg.num_centroids, "too few training points");

    auto check_header = [](const fs::path &path, std::uint32_t rows, std::uint32_t cols) {
        std::ifstream in(path, std::ios::binary);
        std::uint32_t r = 0, c = 0;
        in.read(reinterpret_cast<char *>(&r), sizeof(r));
        in.read(reinterpret_cast<char *>(&c), sizeof(c));
        check(in.good() && r == rows && c == cols, "incorrect output matrix header");
    };
    check_header(output / "pivots.bin", 8, 12);
    check_header(output / "centroid.bin", 12, 1);
    check_header(output / "chunk_offsets.bin", 4, 1);
    check_header(output / "codes.bin", n, 3);

    const std::uintmax_t code_bytes = fs::file_size(output / "codes.bin");
    check(code_bytes == 8 + static_cast<std::uintmax_t>(n) * 3, "incorrect code file size");
    std::ifstream codes(output / "codes.bin", std::ios::binary);
    codes.seekg(8);
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(n) * 3);
    codes.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
    check(codes.good(), "could not read code payload");
    for (std::uint8_t code : payload) check(code < 8, "code index exceeds centroid count");

    // Exercise the metric-aware final-assignment paths as well. As documented,
    // codebook training remains the same L2 Lloyd objective for these modes.
    cfg.metric = rcspq::Metric::InnerProduct;
    const fs::path ip_output = root / "out_ip";
    const rcspq::Result ip_result = rcspq::build(input.string(), ip_output.string(), cfg);
    check(ip_result.num_points == n, "IP path failed");
    cfg.metric = rcspq::Metric::Cosine;
    const fs::path cosine_output = root / "out_cosine";
    const rcspq::Result cosine_result = rcspq::build(input.string(), cosine_output.string(), cfg);
    check(cosine_result.num_points == n, "cosine path failed");

    fs::remove_all(root);
    std::cout << "RCS-PQ standalone smoke test passed\n";
    return 0;
}
