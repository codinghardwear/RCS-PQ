#include "rcspq.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <limits>

namespace {
void usage(const char *program) {
    std::cout
        << "RCS-PQ standalone PQ construction\n\n"
        << "Usage:\n  " << program << " --input DATA.bin --output OUTPUT_DIR [options]\n\n"
        << "Required:\n"
        << "  --input PATH          DiskANN-style float32 .bin matrix\n"
        << "  --output DIR          Output directory\n\n"
        << "Options:\n"
        << "  --subdim N            Target subvector dimensionality (default: 16)\n"
        << "  --centroids K         Number of centroids per chunk, 1..256 (default: 256)\n"
        << "  --train-ratio R       Bernoulli training-sample ratio in (0,1] (default: 0.0256)\n"
        << "  --sample-seed N       Training-sample RNG seed (default: 20261111)\n"
        << "  --kmeans-seed N       K-means++ RNG seed (default: 20261111)\n"
        << "  --kmeans-iters N      Maximum Lloyd iterations (default: 15)\n"
        << "  --threads N           OpenMP threads; 0 uses runtime default (default: 0)\n"
        << "  --batch-size N        Encoding vectors per batch (default: 262144)\n"
        << "  --metric NAME         l2, ip, or cosine (default: l2)\n"
        << "  --help                Show this help\n\n"
        << "Output files: pivots.bin, centroid.bin, chunk_offsets.bin, codes.bin, config.json, summary.txt\n";
}

template <typename T>
T parse_number(const std::string &value, const std::string &name) {
    try {
        if constexpr (std::is_floating_point<T>::value) {
            std::size_t pos = 0;
            const double x = std::stod(value, &pos);
            if (pos != value.size()) throw std::invalid_argument("trailing characters");
            return static_cast<T>(x);
        } else {
            std::size_t pos = 0;
            const unsigned long long x = std::stoull(value, &pos, 10);
            if (pos != value.size()) throw std::invalid_argument("trailing characters");
            return static_cast<T>(x);
        }
    } catch (...) {
        throw std::invalid_argument("Invalid value for " + name + ": " + value);
    }
}
}

int main(int argc, char **argv) {
    try {
        std::string input, output;
        rcspq::Config cfg;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
            if (i + 1 >= argc) throw std::invalid_argument("Missing value after " + arg);
            const std::string value = argv[++i];
            if (arg == "--input") input = value;
            else if (arg == "--output") output = value;
            else if (arg == "--subdim") cfg.subdim = parse_number<std::size_t>(value, arg);
            else if (arg == "--centroids") cfg.num_centroids = parse_number<std::size_t>(value, arg);
            else if (arg == "--train-ratio") cfg.train_ratio = parse_number<double>(value, arg);
            else if (arg == "--sample-seed") cfg.sample_seed = parse_number<std::uint32_t>(value, arg);
            else if (arg == "--kmeans-seed") cfg.kmeans_seed = parse_number<std::uint64_t>(value, arg);
            else if (arg == "--kmeans-iters") cfg.kmeans_iterations = parse_number<std::size_t>(value, arg);
            else if (arg == "--threads") cfg.threads = parse_number<std::size_t>(value, arg);
            else if (arg == "--batch-size") cfg.batch_size = parse_number<std::size_t>(value, arg);
            else if (arg == "--metric") {
                if (value == "l2") cfg.metric = rcspq::Metric::L2;
                else if (value == "ip") cfg.metric = rcspq::Metric::InnerProduct;
                else if (value == "cosine") cfg.metric = rcspq::Metric::Cosine;
                else throw std::invalid_argument("--metric must be l2, ip, or cosine");
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }
        if (input.empty() || output.empty()) {
            usage(argv[0]);
            throw std::invalid_argument("Both --input and --output are required");
        }
        const rcspq::Result r = rcspq::build(input, output, cfg);
        std::cout << "\nCompleted successfully.\n"
                  << "Training sample: " << r.train_points << "\n"
                  << "PQ Training: " << r.training_seconds << " s\n"
                  << "PQ Encoding: " << r.encoding_seconds << " s\n"
                  << "Total: " << r.total_seconds << " s\n"
                  << "Outputs: " << output << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
