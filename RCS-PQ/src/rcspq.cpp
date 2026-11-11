// RCS-PQ standalone implementation.
// Portions of the PQ construction path are adapted from DiskANN 0.7.0
// (Copyright (c) Microsoft Corporation; MIT-licensed). See THIRD_PARTY_NOTICES.md.

#include "rcspq.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <immintrin.h>

namespace fs = std::filesystem;

namespace rcspq {
namespace {
using Clock = std::chrono::steady_clock;

inline double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

struct MatrixHeader {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

MatrixHeader read_header(std::ifstream &in, const std::string &path) {
    MatrixHeader h;
    in.read(reinterpret_cast<char *>(&h.rows), sizeof(h.rows));
    in.read(reinterpret_cast<char *>(&h.cols), sizeof(h.cols));
    if (!in || h.rows == 0 || h.cols == 0)
        throw std::runtime_error("Invalid or empty float32 .bin matrix: " + path);
    return h;
}

void write_bin_header(std::ofstream &out, std::uint32_t rows, std::uint32_t cols) {
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    if (!out) throw std::runtime_error("Failed to write output matrix header");
}

template <typename T>
void write_matrix_bin(const fs::path &path, const T *data, std::uint32_t rows, std::uint32_t cols) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create output file: " + path.string());
    write_bin_header(out, rows, cols);
    const std::uint64_t count = static_cast<std::uint64_t>(rows) * cols;
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) throw std::runtime_error("Failed writing output file: " + path.string());
}

inline float squared_l2(const float *a, const float *b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

std::vector<std::uint32_t> make_chunk_offsets(std::size_t dim, std::size_t subdim) {
    const std::size_t chunks = (dim + subdim - 1) / subdim;
    const std::size_t low = dim / chunks;
    const std::size_t high_count = dim % chunks;
    std::vector<std::uint32_t> offsets(chunks + 1, 0);
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t width = low + (c < high_count ? 1 : 0);
        offsets[c + 1] = offsets[c] + static_cast<std::uint32_t>(width);
    }
    return offsets;
}

void sample_training_data(const std::string &input_path, double ratio, std::uint32_t seed,
                          std::size_t expected_points, std::size_t dim,
                          std::vector<float> &sample, std::size_t &sampled_points) {
    std::ifstream in(input_path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input file: " + input_path);
    const MatrixHeader h = read_header(in, input_path);
    if (h.rows != expected_points || h.cols != dim)
        throw std::runtime_error("Input matrix header changed while sampling");

    sample.clear();
    const std::size_t reserve_points = std::min<std::size_t>(expected_points,
        static_cast<std::size_t>(std::ceil(static_cast<double>(expected_points) * ratio * 1.08)) + 1024);
    if (reserve_points <= std::numeric_limits<std::size_t>::max() / dim)
        sample.reserve(reserve_points * dim);

    std::vector<float> row(dim);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (std::size_t i = 0; i < expected_points; ++i) {
        in.read(reinterpret_cast<char *>(row.data()), static_cast<std::streamsize>(dim * sizeof(float)));
        if (!in) throw std::runtime_error("Unexpected EOF while sampling input vectors");
        if (distribution(generator) < ratio)
            sample.insert(sample.end(), row.begin(), row.end());
    }
    sampled_points = sample.size() / dim;
    if (sampled_points == 0) throw std::runtime_error("Training sample is empty; increase --train-ratio");
}

void compute_mean_and_center(std::vector<float> &data, std::size_t n, std::size_t dim,
                             std::vector<float> &mean, bool center) {
    mean.assign(dim, 0.0f);
    if (!center || n == 0) return;

    const int threads = std::max(1, omp_get_max_threads());
    std::vector<double> partial(static_cast<std::size_t>(threads) * dim, 0.0);
#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        double *local = partial.data() + static_cast<std::size_t>(tid) * dim;
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            const float *row = data.data() + static_cast<std::size_t>(i) * dim;
            for (std::size_t d = 0; d < dim; ++d) local[d] += static_cast<double>(row[d]);
        }
    }
    for (std::size_t d = 0; d < dim; ++d) {
        double sum = 0.0;
        for (int t = 0; t < threads; ++t) sum += partial[static_cast<std::size_t>(t) * dim + d];
        mean[d] = static_cast<float>(sum / static_cast<double>(n));
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        float *row = data.data() + static_cast<std::size_t>(i) * dim;
        for (std::size_t d = 0; d < dim; ++d) row[d] -= mean[d];
    }
}

struct CodebookView {
    std::size_t dim = 0;
    std::size_t num_centers = 0;
    std::size_t padded_centers = 0;
    Metric metric = Metric::L2;
    std::vector<float> centroids_soa;
    std::vector<float> score_bias;
};

#if defined(__AVX512F__)
static constexpr std::size_t SIMD_WIDTH = 16;
#elif defined(__AVX2__)
static constexpr std::size_t SIMD_WIDTH = 8;
#else
static constexpr std::size_t SIMD_WIDTH = 1;
#endif

inline std::size_t round_up(std::size_t x, std::size_t a) { return ((x + a - 1) / a) * a; }

void prepare_codebook(CodebookView &cb, const float *pivot_data, std::size_t pivot_stride,
                      std::size_t pivot_offset, std::size_t subdim, std::size_t num_centers, Metric metric) {
    cb.dim = subdim;
    cb.num_centers = num_centers;
    cb.padded_centers = round_up(num_centers, SIMD_WIDTH);
    cb.metric = metric;
    cb.centroids_soa.assign(subdim * cb.padded_centers, 0.0f);
    cb.score_bias.assign(cb.padded_centers, std::numeric_limits<float>::infinity());

    for (std::size_t c = 0; c < num_centers; ++c) {
        const float *src = pivot_data + c * pivot_stride + pivot_offset;
        double norm_sq = 0.0;
        for (std::size_t d = 0; d < subdim; ++d) norm_sq += static_cast<double>(src[d]) * src[d];
        const float inv_norm = (metric == Metric::Cosine && norm_sq > 0.0)
            ? static_cast<float>(1.0 / std::sqrt(norm_sq)) : 1.0f;
        for (std::size_t d = 0; d < subdim; ++d)
            cb.centroids_soa[d * cb.padded_centers + c] = src[d] * inv_norm;
        cb.score_bias[c] = (metric == Metric::L2) ? static_cast<float>(0.5 * norm_sq) : 0.0f;
    }
}

std::uint32_t assign_centroid(const float *vec, const CodebookView &cb, float *best_score_out = nullptr) {
    float best_score = std::numeric_limits<float>::infinity();
    std::uint32_t best_id = 0;
#if defined(__AVX512F__)
    for (std::size_t c = 0; c < cb.padded_centers; c += 16) {
        __m512 acc = _mm512_setzero_ps();
#if defined(__GNUC__)
        if (cb.dim <= 8) {
#pragma GCC unroll 8
            for (std::size_t d = 0; d < cb.dim; ++d) {
                const __m512 x = _mm512_set1_ps(vec[d]);
                const __m512 cent = _mm512_loadu_ps(cb.centroids_soa.data() + d * cb.padded_centers + c);
                acc = _mm512_fmadd_ps(x, cent, acc);
            }
        } else
#endif
        {
            for (std::size_t d = 0; d < cb.dim; ++d) {
                const __m512 x = _mm512_set1_ps(vec[d]);
                const __m512 cent = _mm512_loadu_ps(cb.centroids_soa.data() + d * cb.padded_centers + c);
                acc = _mm512_fmadd_ps(x, cent, acc);
            }
        }
        const __m512 bias = _mm512_loadu_ps(cb.score_bias.data() + c);
        const __m512 scores = _mm512_sub_ps(bias, acc);
        const float block_min = _mm512_reduce_min_ps(scores);
        const __mmask16 equal_mask = _mm512_cmp_ps_mask(scores, _mm512_set1_ps(block_min), _CMP_EQ_OQ);
        const std::uint32_t valid_lanes = static_cast<std::uint32_t>(std::min<std::size_t>(16, cb.num_centers - c));
        const __mmask16 valid_mask = static_cast<__mmask16>((1u << valid_lanes) - 1u);
        const unsigned candidates = static_cast<unsigned>(equal_mask & valid_mask);
        const std::uint32_t lane = candidates ? static_cast<std::uint32_t>(__builtin_ctz(candidates)) : 0u;
        if (block_min < best_score) { best_score = block_min; best_id = static_cast<std::uint32_t>(c + lane); }
    }
#elif defined(__AVX2__)
    alignas(32) float lanes[8];
    for (std::size_t c = 0; c < cb.padded_centers; c += 8) {
        __m256 acc = _mm256_setzero_ps();
#if defined(__GNUC__)
        if (cb.dim <= 8) {
#pragma GCC unroll 8
            for (std::size_t d = 0; d < cb.dim; ++d) {
                const __m256 x = _mm256_set1_ps(vec[d]);
                const __m256 cent = _mm256_loadu_ps(cb.centroids_soa.data() + d * cb.padded_centers + c);
#if defined(__FMA__)
                acc = _mm256_fmadd_ps(x, cent, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(x, cent));
#endif
            }
        } else
#endif
        {
            for (std::size_t d = 0; d < cb.dim; ++d) {
                const __m256 x = _mm256_set1_ps(vec[d]);
                const __m256 cent = _mm256_loadu_ps(cb.centroids_soa.data() + d * cb.padded_centers + c);
#if defined(__FMA__)
                acc = _mm256_fmadd_ps(x, cent, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(x, cent));
#endif
            }
        }
        const __m256 bias = _mm256_loadu_ps(cb.score_bias.data() + c);
        const __m256 scores = _mm256_sub_ps(bias, acc);
        _mm256_store_ps(lanes, scores);
        for (std::uint32_t lane = 0; lane < 8; ++lane) {
            const std::size_t id = c + lane;
            if (id >= cb.num_centers) break;
            if (lanes[lane] < best_score) { best_score = lanes[lane]; best_id = static_cast<std::uint32_t>(id); }
        }
    }
#else
    for (std::size_t c = 0; c < cb.num_centers; ++c) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < cb.dim; ++d)
            dot += vec[d] * cb.centroids_soa[d * cb.padded_centers + c];
        const float score = cb.score_bias[c] - dot;
        if (score < best_score) { best_score = score; best_id = static_cast<std::uint32_t>(c); }
    }
#endif
    if (best_score_out) *best_score_out = best_score;
    return best_id;
}

void assign_batch_strided(const float *data, std::size_t n, std::size_t row_stride, std::size_t data_offset,
                          const CodebookView &cb, std::uint8_t *labels, std::size_t label_stride,
                          std::size_t label_offset) {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const std::uint32_t label = assign_centroid(data + static_cast<std::size_t>(i) * row_stride + data_offset, cb);
        labels[static_cast<std::size_t>(i) * label_stride + label_offset] = static_cast<std::uint8_t>(label);
    }
}

void kmeanspp_select(const float *data, std::size_t n, std::size_t dim, float *centers,
                     std::size_t k, std::uint64_t seed) {
    if (n == 0 || dim == 0 || k == 0) return;
    if (k > n) throw std::runtime_error("Number of centroids cannot exceed training sample count");
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> first_dist(0, n - 1);
    std::vector<std::uint8_t> picked(n, 0);
    const std::size_t first = first_dist(generator);
    picked[first] = 1;
    std::memcpy(centers, data + first * dim, dim * sizeof(float));

    // Match the DiskANN-derived path: above 2^23 points, use unique random pivots.
    if (n > (static_cast<std::size_t>(1) << 23)) {
        std::uniform_int_distribution<std::size_t> any_dist(0, n - 1);
        std::size_t selected = 1;
        while (selected < k) {
            const std::size_t candidate = any_dist(generator);
            if (picked[candidate]) continue;
            picked[candidate] = 1;
            std::memcpy(centers + selected * dim, data + candidate * dim, dim * sizeof(float));
            ++selected;
        }
        return;
    }

    std::vector<float> min_dist(n, 0.0f);
    constexpr std::size_t SAMPLE_BLOCK = 4096;
    const std::size_t blocks = (n + SAMPLE_BLOCK - 1) / SAMPLE_BLOCK;
    std::vector<double> block_sums(blocks, 0.0);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    const int threads = std::max(1, omp_get_max_threads());
    std::size_t chosen = n;
#pragma omp parallel num_threads(threads) shared(min_dist, block_sums, chosen, picked, centers, data)
    {
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
            min_dist[static_cast<std::size_t>(i)] = squared_l2(data + static_cast<std::size_t>(i) * dim, data + first * dim, dim);
        for (std::size_t selected = 1; selected < k; ++selected) {
#pragma omp for schedule(static)
            for (std::int64_t b = 0; b < static_cast<std::int64_t>(blocks); ++b) {
                const std::size_t begin = static_cast<std::size_t>(b) * SAMPLE_BLOCK;
                const std::size_t end = std::min(n, begin + SAMPLE_BLOCK);
                double sum = 0.0;
                for (std::size_t i = begin; i < end; ++i) sum += static_cast<double>(min_dist[i]);
                block_sums[static_cast<std::size_t>(b)] = sum;
            }
#pragma omp single
            {
                double total = 0.0;
                for (double value : block_sums) total += value;
                chosen = n;
                if (total > 0.0 && std::isfinite(total)) {
                    const double target = unit_dist(generator) * total;
                    double prefix = 0.0;
                    std::size_t chosen_block = blocks - 1;
                    double local_target = 0.0;
                    for (std::size_t b = 0; b < blocks; ++b) {
                        if (target < prefix + block_sums[b] || b + 1 == blocks) {
                            chosen_block = b; local_target = target - prefix; break;
                        }
                        prefix += block_sums[b];
                    }
                    const std::size_t begin = chosen_block * SAMPLE_BLOCK;
                    const std::size_t end = std::min(n, begin + SAMPLE_BLOCK);
                    double local_prefix = 0.0;
                    for (std::size_t i = begin; i < end; ++i) {
                        local_prefix += static_cast<double>(min_dist[i]);
                        if (local_target < local_prefix && min_dist[i] > 0.0f) { chosen = i; break; }
                    }
                }
                if (chosen >= n || picked[chosen]) {
                    const std::size_t start = first_dist(generator);
                    for (std::size_t off = 0; off < n; ++off) {
                        const std::size_t candidate = (start + off) % n;
                        if (!picked[candidate] && (total == 0.0 || min_dist[candidate] > 0.0f)) { chosen = candidate; break; }
                    }
                }
                if (chosen >= n || picked[chosen]) {
                    for (std::size_t candidate = 0; candidate < n; ++candidate)
                        if (!picked[candidate]) { chosen = candidate; break; }
                }
                picked[chosen] = 1;
                std::memcpy(centers + selected * dim, data + chosen * dim, dim * sizeof(float));
            }
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
                const std::size_t point = static_cast<std::size_t>(i);
                const float dist = squared_l2(data + point * dim, data + chosen * dim, dim);
                min_dist[point] = std::min(min_dist[point], dist);
            }
        }
    }
}

float run_lloyds(float *data, std::size_t n, std::size_t dim, float *centers,
                 std::size_t k, std::size_t max_reps) {
    if (n == 0 || dim == 0 || k == 0) return 0.0f;
    const int threads = std::max(1, omp_get_max_threads());
    const std::size_t sums_per_thread = k * dim;
    std::vector<double> thread_sums(static_cast<std::size_t>(threads) * sums_per_thread, 0.0);
    std::vector<std::uint64_t> thread_counts(static_cast<std::size_t>(threads) * k, 0);
    std::vector<double> thread_losses(static_cast<std::size_t>(threads) * 8, 0.0);
    std::vector<float> data_norms(n, 0.0f);
    std::vector<std::uint32_t> labels(n, 0);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        double norm_sq = 0.0;
        const float *row = data + static_cast<std::size_t>(i) * dim;
        for (std::size_t d = 0; d < dim; ++d) norm_sq += static_cast<double>(row[d]) * row[d];
        data_norms[static_cast<std::size_t>(i)] = static_cast<float>(norm_sq);
    }

    float residual = std::numeric_limits<float>::max();
    for (std::size_t iter = 0; iter < max_reps; ++iter) {
        CodebookView cb;
        prepare_codebook(cb, centers, dim, 0, dim, k, Metric::L2);
        std::fill(thread_sums.begin(), thread_sums.end(), 0.0);
        std::fill(thread_counts.begin(), thread_counts.end(), 0);
        std::fill(thread_losses.begin(), thread_losses.end(), 0.0);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            double local_loss = 0.0;
            const std::size_t sum_base = static_cast<std::size_t>(tid) * sums_per_thread;
            const std::size_t count_base = static_cast<std::size_t>(tid) * k;
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
                const std::size_t point = static_cast<std::size_t>(i);
                float selected_score = 0.0f;
                const std::uint32_t label = assign_centroid(data + point * dim, cb, &selected_score);
                labels[point] = label;
                thread_counts[count_base + label]++;
                double *cluster_sum = thread_sums.data() + sum_base + static_cast<std::size_t>(label) * dim;
                const float *row = data + point * dim;
                for (std::size_t d = 0; d < dim; ++d) cluster_sum[d] += static_cast<double>(row[d]);
                local_loss += static_cast<double>(data_norms[point]) + 2.0 * static_cast<double>(selected_score);
            }
            thread_losses[static_cast<std::size_t>(tid) * 8] = local_loss;
        }
#pragma omp parallel for schedule(static) num_threads(threads)
        for (std::int64_t c = 0; c < static_cast<std::int64_t>(k); ++c) {
            const std::size_t center = static_cast<std::size_t>(c);
            std::uint64_t count = 0;
            for (int t = 0; t < threads; ++t) count += thread_counts[static_cast<std::size_t>(t) * k + center];
            float *dst = centers + center * dim;
            if (count == 0) { std::fill(dst, dst + dim, 0.0f); continue; }
            for (std::size_t d = 0; d < dim; ++d) {
                double sum = 0.0;
                for (int t = 0; t < threads; ++t)
                    sum += thread_sums[static_cast<std::size_t>(t) * sums_per_thread + center * dim + d];
                dst[d] = static_cast<float>(sum / static_cast<double>(count));
            }
        }
        double residual_sum = 0.0;
        for (int t = 0; t < threads; ++t) residual_sum += thread_losses[static_cast<std::size_t>(t) * 8];
        const float new_residual = static_cast<float>(residual_sum);
        const bool converged = (new_residual < std::numeric_limits<float>::epsilon()) ||
            (iter != 0 && new_residual > 0.0f && (residual - new_residual) / new_residual < 0.00001f);
        residual = new_residual;
        if (converged) break;
    }
    return residual;
}

void write_config(const fs::path &path, const Config &cfg, const Result &r) {
    std::ofstream out(path);
    out << "{\n"
        << "  \"metric\": \"" << metric_name(cfg.metric) << "\",\n"
        << "  \"num_points\": " << r.num_points << ",\n"
        << "  \"dimension\": " << r.dimension << ",\n"
        << "  \"num_chunks\": " << r.num_chunks << ",\n"
        << "  \"subdim_requested\": " << cfg.subdim << ",\n"
        << "  \"num_centroids\": " << cfg.num_centroids << ",\n"
        << "  \"train_ratio\": " << std::setprecision(10) << cfg.train_ratio << ",\n"
        << "  \"train_points\": " << r.train_points << ",\n"
        << "  \"sample_seed\": " << cfg.sample_seed << ",\n"
        << "  \"kmeans_seed\": " << cfg.kmeans_seed << ",\n"
        << "  \"kmeans_iterations\": " << cfg.kmeans_iterations << ",\n"
        << "  \"threads\": " << (cfg.threads ? cfg.threads : static_cast<std::size_t>(omp_get_max_threads())) << ",\n"
        << "  \"batch_size\": " << cfg.batch_size << ",\n"
        << "  \"simd_backend\": \"" << r.simd_backend << "\",\n"
        << "  \"sampling_seconds\": " << r.sampling_seconds << ",\n"
        << "  \"training_seconds\": " << r.training_seconds << ",\n"
        << "  \"encoding_seconds\": " << r.encoding_seconds << ",\n"
        << "  \"total_seconds\": " << r.total_seconds << "\n"
        << "}\n";
}

} // namespace

const char *simd_backend() {
#if defined(__AVX512F__)
    return "AVX-512F (16 centroid lanes)";
#elif defined(__AVX2__)
    return "AVX2 (8 centroid lanes)";
#else
    return "scalar fallback";
#endif
}

const char *metric_name(Metric metric) {
    switch (metric) {
        case Metric::L2: return "l2";
        case Metric::InnerProduct: return "ip";
        case Metric::Cosine: return "cosine";
    }
    return "unknown";
}

Result build(const std::string &input_path, const std::string &output_dir, const Config &cfg) {
    if (cfg.subdim == 0) throw std::invalid_argument("--subdim must be positive");
    if (cfg.num_centroids == 0 || cfg.num_centroids > 256)
        throw std::invalid_argument("--centroids must be in [1, 256] for one-byte PQ codes");
    if (!(cfg.train_ratio > 0.0 && cfg.train_ratio <= 1.0))
        throw std::invalid_argument("--train-ratio must be in (0, 1]");
    if (cfg.kmeans_iterations == 0) throw std::invalid_argument("--kmeans-iters must be positive");
    if (cfg.batch_size == 0) throw std::invalid_argument("--batch-size must be positive");
    if (cfg.threads > 0) omp_set_num_threads(static_cast<int>(cfg.threads));

    const auto total_start = Clock::now();
    std::ifstream header_in(input_path, std::ios::binary);
    if (!header_in) throw std::runtime_error("Cannot open input file: " + input_path);
    const MatrixHeader h = read_header(header_in, input_path);
    const std::size_t n = h.rows, dim = h.cols;
    header_in.close();
    const std::uintmax_t expected_bytes = 2 * sizeof(std::uint32_t) +
        static_cast<std::uintmax_t>(n) * dim * sizeof(float);
    const std::uintmax_t actual_bytes = fs::file_size(input_path);
    if (actual_bytes != expected_bytes)
        throw std::runtime_error("Input file size does not match its float32 matrix header: " + input_path);

    const std::vector<std::uint32_t> offsets = make_chunk_offsets(dim, cfg.subdim);
    const std::size_t chunks = offsets.size() - 1;
    if (cfg.num_centroids > n) throw std::runtime_error("Centroid count exceeds dataset size");

    fs::create_directories(output_dir);
    const fs::path outdir(output_dir);
    for (const char *name : {"pivots.bin", "centroid.bin", "chunk_offsets.bin", "codes.bin", "config.json", "summary.txt"}) {
        if (fs::exists(outdir / name))
            throw std::runtime_error("Output already exists; choose a new --output directory: " + (outdir / name).string());
    }
    Result result;
    result.num_points = n;
    result.dimension = dim;
    result.num_chunks = chunks;
    result.simd_backend = simd_backend();

    std::cout << "RCS-PQ standalone construction\n"
              << "Input: " << input_path << "\n"
              << "N=" << n << " D=" << dim << " subdim=" << cfg.subdim << " chunks=" << chunks
              << " K=" << cfg.num_centroids << " threads=" << omp_get_max_threads() << "\n"
              << "Metric=" << metric_name(cfg.metric) << " SIMD=" << result.simd_backend << "\n";

    // Stage 1: deterministic Bernoulli sampling, matching the source integration's sampling model.
    std::vector<float> train_data;
    const auto sampling_start = Clock::now();
    sample_training_data(input_path, cfg.train_ratio, cfg.sample_seed, n, dim, train_data, result.train_points);
    result.sampling_seconds = elapsed_seconds(sampling_start, Clock::now());
    if (result.train_points < cfg.num_centroids)
        throw std::runtime_error("Training sample has fewer points than --centroids");
    std::cout << "Training sample: " << result.train_points << " vectors (" << result.sampling_seconds << " s)\n";

    // Stage 2: train independent codebooks for each chunk.
    const auto training_start = Clock::now();
    std::vector<float> mean;
    const bool center = (cfg.metric == Metric::L2);
    compute_mean_and_center(train_data, result.train_points, dim, mean, center);
    std::vector<float> pivots(cfg.num_centroids * dim, 0.0f);

    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t begin = offsets[chunk];
        const std::size_t width = offsets[chunk + 1] - begin;
        std::vector<float> chunk_data(result.train_points * width);
        std::vector<float> chunk_centers(cfg.num_centroids * width);
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(result.train_points); ++i)
            std::memcpy(chunk_data.data() + static_cast<std::size_t>(i) * width,
                        train_data.data() + static_cast<std::size_t>(i) * dim + begin,
                        width * sizeof(float));

        kmeanspp_select(chunk_data.data(), result.train_points, width, chunk_centers.data(),
                        cfg.num_centroids, cfg.kmeans_seed + static_cast<std::uint64_t>(chunk));
        const float loss = run_lloyds(chunk_data.data(), result.train_points, width, chunk_centers.data(),
                                      cfg.num_centroids, cfg.kmeans_iterations);
        for (std::size_t c = 0; c < cfg.num_centroids; ++c)
            std::memcpy(pivots.data() + c * dim + begin, chunk_centers.data() + c * width, width * sizeof(float));
        std::cout << "  trained chunk " << chunk << "/" << chunks - 1 << " (dim " << begin << ":" << begin + width
                  << ", final L2 loss=" << loss << ")\n";
    }
    // Save codebooks and metadata as ordinary matrix .bin files. The PQ-training
    // timer includes writing the trained pivots, matching the integration timer boundary.
    write_matrix_bin<float>(outdir / "pivots.bin", pivots.data(), static_cast<std::uint32_t>(cfg.num_centroids), h.cols);
    write_matrix_bin<float>(outdir / "centroid.bin", mean.data(), h.cols, 1);
    write_matrix_bin<std::uint32_t>(outdir / "chunk_offsets.bin", offsets.data(), static_cast<std::uint32_t>(offsets.size()), 1);
    result.training_seconds = elapsed_seconds(training_start, Clock::now());
    std::cout << "PQ Training: " << result.training_seconds << " s\n";

    // Stage 3: chunk-centric encoding. Input batches are streamed; no N x K x QD score table is materialized.
    const auto encoding_start = Clock::now();
    std::vector<CodebookView> codebooks(chunks);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t begin = offsets[chunk];
        const std::size_t width = offsets[chunk + 1] - begin;
        prepare_codebook(codebooks[chunk], pivots.data(), dim, begin, width, cfg.num_centroids, cfg.metric);
    }

    std::ifstream data_in(input_path, std::ios::binary);
    if (!data_in) throw std::runtime_error("Cannot reopen input file: " + input_path);
    const MatrixHeader h2 = read_header(data_in, input_path);
    if (h2.rows != h.rows || h2.cols != h.cols) throw std::runtime_error("Input matrix header changed before encoding");
    std::ofstream codes_out(outdir / "codes.bin", std::ios::binary);
    if (!codes_out) throw std::runtime_error("Cannot create codes.bin");
    write_bin_header(codes_out, h.rows, static_cast<std::uint32_t>(chunks));

    const std::size_t batch = std::min<std::size_t>(cfg.batch_size, n);
    std::vector<float> block(batch * dim);
    std::vector<std::uint8_t> block_codes(batch * chunks);
    for (std::size_t start = 0; start < n; start += batch) {
        const std::size_t count = std::min(batch, n - start);
        data_in.read(reinterpret_cast<char *>(block.data()), static_cast<std::streamsize>(count * dim * sizeof(float)));
        if (!data_in) throw std::runtime_error("Unexpected EOF during encoding");
        if (center) {
#pragma omp parallel for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
                float *row = block.data() + static_cast<std::size_t>(i) * dim;
                for (std::size_t d = 0; d < dim; ++d) row[d] -= mean[d];
            }
        }
        std::fill(block_codes.begin(), block_codes.begin() + count * chunks, 0);
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            const std::size_t begin = offsets[chunk];
            assign_batch_strided(block.data(), count, dim, begin, codebooks[chunk],
                                 block_codes.data(), chunks, chunk);
        }
        codes_out.write(reinterpret_cast<const char *>(block_codes.data()),
                        static_cast<std::streamsize>(count * chunks));
        if (!codes_out) throw std::runtime_error("Failed writing codes.bin");
        std::cout << "  encoded vectors [" << start << ", " << start + count << ")\n";
    }
    result.encoding_seconds = elapsed_seconds(encoding_start, Clock::now());
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    std::cout << "PQ Encoding: " << result.encoding_seconds << " s\n";
    std::cout << "PQ Total (sampling + training + encoding): " << result.total_seconds << " s\n";

    write_config(outdir / "config.json", cfg, result);
    std::ofstream summary(outdir / "summary.txt");
    summary << "RCS-PQ standalone construction\n"
            << "N=" << n << "\nD=" << dim << "\nchunks=" << chunks << "\nsubdim_requested=" << cfg.subdim
            << "\ncentroids=" << cfg.num_centroids << "\ntrain_points=" << result.train_points
            << "\nmetric=" << metric_name(cfg.metric) << "\nSIMD=" << result.simd_backend
            << "\nsampling_seconds=" << result.sampling_seconds
            << "\ntraining_seconds=" << result.training_seconds
            << "\nencoding_seconds=" << result.encoding_seconds
            << "\ntotal_seconds=" << result.total_seconds << "\n";
    return result;
}

} // namespace rcspq
