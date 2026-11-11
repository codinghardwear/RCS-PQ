# RCS-PQ: Standalone PQ Construction

This repository provides a standalone C++ implementation of the RCS-PQ construction path. It trains product-quantization codebooks and encodes vectors into PQ codes **without building an ANN graph, disk index, or search structure**. It does not require DiskANN, FAISS, Starling, MKL, or TSL/robin-map.

The implementation is based on the RCS-PQ/CS-PQ source files supplied for this project and the standalone-relevant PQ path in the DiskANN 0.7.0 integration. DiskANN-derived portions retain their upstream MIT notice; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Included techniques

1. **Ranking-aware scoring reformulation.** For L2 assignment, the kernel minimizes `0.5 * ||c||^2 - x dot c`, omitting the candidate-independent `||x||^2` term. During Lloyd training, the selected score is combined with the precomputed vector norm to recover the L2 residual used for termination.
2. **Centroid-parallel SIMD.** SIMD lanes represent candidate centroids. The codebook is stored in a dimension-major SoA layout `[subdimension][padded_centroid]`; AVX-512 uses 16 centroid lanes, AVX2 uses 8, and a scalar fallback is provided. For subdimensions up to 8, the dimension loop has a compiler-unrolled path.
3. **Cache-friendly chunk-centric encoding.** For each input batch, the implementation iterates over PQ chunks, reuses the prepared codebook for all vectors in that batch, and writes only the selected centroid IDs. It does not materialize an `N x K` distance matrix or an `N x K x chunks` score tensor.

## Requirements

- Linux (tested build path), C++17 compiler (GCC 9+ recommended)
- CMake 3.16+
- OpenMP
- AVX-512F or AVX2 CPU for SIMD acceleration; scalar fallback otherwise

No external numerical library is required.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRCSPQ_NATIVE=ON
cmake --build build -j
```

`RCSPQ_NATIVE=ON` compiles for the machine on which compilation occurs. This is recommended for local experiments, but the resulting binary may not run on a CPU with a different instruction set. Set `-DRCSPQ_NATIVE=OFF` for a portable scalar build, or compile with explicit target flags suitable for the deployment CPU.

Check the selected backend:

```bash
./build/rcspq --help
```

At runtime, the program prints the selected SIMD backend.

## Input format

The input must be a DiskANN-style float32 matrix file:

```text
uint32 number_of_vectors
uint32 vector_dimension
float32 values in row-major order
```

The file must contain exactly `8 + number_of_vectors * dimension * 4` bytes. The header and values are little-endian, as on the x86-64 systems used in the experiments. The current standalone CLI accepts float32 input only.

## Run

Example for SIFT10M-256D, with 16 dimensions per subvector and 40 OpenMP threads:

```bash
./build/rcspq \
  --input /path/to/sift10m256dfloat.bin \
  --output ./results/sift10m256d_subdim16_t40 \
  --subdim 16 \
  --centroids 256 \
  --train-ratio 0.0256 \
  --sample-seed 20261111 \
  --kmeans-seed 20261111 \
  --kmeans-iters 15 \
  --threads 40 \
  --batch-size 262144 \
  --metric l2
```

Useful thread sweep:

```bash
for t in 1 10 20 40 80; do
  ./build/rcspq \
    --input /path/to/sift10m256dfloat.bin \
    --output "./results/sift10m256d_subdim16_t${t}" \
    --subdim 16 --centroids 256 --train-ratio 0.0256 \
    --sample-seed 20261111 --kmeans-seed 20261111 \
    --kmeans-iters 15 --threads "$t" --batch-size 262144 --metric l2
 done
```

Set `OMP_PLACES=cores` and `OMP_PROC_BIND=spread` for controlled CPU placement, for example:

```bash
OMP_PLACES=cores OMP_PROC_BIND=spread ./build/rcspq ...
```

### Run a thread sweep

The package includes `scripts/run_thread_sweep.sh`. It runs one PQ-only construction per thread count, stores each run's output separately, and writes a summary CSV:

```bash
INPUT=/path/to/sift10m256dfloat.bin \
THREADS_LIST="1 10 20 40 80" \
SUBDIM=16 CENTROIDS=256 TRAIN_RATIO=0.0256 \
SAMPLE_SEED=20261111 KMEANS_SEED=20261111 \
OMP_PLACES=cores OMP_PROC_BIND=spread \
bash scripts/run_thread_sweep.sh
```

For a detached run:

```bash
RUN_ID="sift256_thread_$(date -u +%Y%m%dT%H%M%SZ)"
nohup env INPUT=/path/to/sift10m256dfloat.bin THREADS_LIST="1 10 20 40 80" \
  RUN_ID="$RUN_ID" bash scripts/run_thread_sweep.sh \
  > "${RUN_ID}.nohup.log" 2>&1 < /dev/null &
echo "PID=$! RUN_ID=$RUN_ID"
```

## Parameters

| Option | Default | Meaning |
|---|---:|---|
| `--subdim` | 16 | Target subvector dimensionality; number of chunks is `ceil(D/subdim)` |
| `--centroids` | 256 | Centroids per chunk; limited to 256 because codes are one byte |
| `--train-ratio` | 0.0256 | Per-vector Bernoulli sampling probability for codebook training |
| `--sample-seed` | 20261111 | Seed for deterministic training-sample selection |
| `--kmeans-seed` | 20261111 | Base seed for deterministic k-means++ initialization; chunk ID is added |
| `--kmeans-iters` | 15 | Maximum Lloyd iterations per chunk |
| `--threads` | 0 | OpenMP thread count; 0 uses the OpenMP runtime default |
| `--batch-size` | 262144 | Number of vectors encoded in one batch |
| `--metric` | l2 | `l2`, `ip`, or `cosine` scoring during final encoding |

The sample is selected with the same Bernoulli-sampling model as the DiskANN integration. The actual selected sample count is printed and recorded in `config.json`; it is not guaranteed to equal `N * train_ratio` exactly.

### Metric scope

The current evaluated RCS-PQ path and default configuration use L2. The encoding score supports:

- L2: `0.5 * ||c||^2 - x dot c`
- Inner product: `-x dot c`
- Cosine: `-x dot (c / ||c||)`

As in the supplied DiskANN integration, codebook training uses standard L2 Lloyd assignment and mean updates. For `ip` and `cosine`, the training data is not mean-centered and the final encoding assignment uses the selected metric score; this does **not** claim to implement an IP-optimal or spherical-k-means training objective. Use `l2` for reproducing the paper's SIFT experiments.

## Output format

All `.bin` outputs use a simple matrix format: two little-endian `uint32` values (rows, columns), followed by row-major payload.

| File | Matrix shape | Payload |
|---|---|---|
| `pivots.bin` | `K x D` | Float32 centroids. Each row contains all chunk centroids in their original dimension positions. |
| `centroid.bin` | `D x 1` | Float32 global mean subtracted from L2 vectors; zeros when centering is disabled. |
| `chunk_offsets.bin` | `(M+1) x 1` | UInt32 start/end offsets for each chunk. |
| `codes.bin` | `N x M` | UInt8 centroid ID for each vector and chunk. |
| `config.json` | — | Configuration and timing metadata. |
| `summary.txt` | — | Human-readable run summary. |

The output files are sufficient to reconstruct the PQ approximation: for vector `i`, chunk `j` uses centroid `codes[i,j]` from the corresponding dimension range in `pivots.bin`; add `centroid.bin` back for L2-centered data. The standalone `pivots.bin` is a simple matrix file, not the packed DiskANN pivot metadata container.

## Timing definitions

- `sampling_seconds`: reading the input once and selecting the training sample. This is reported separately.
- `training_seconds`: mean/centering, per-chunk k-means++ initialization, and Lloyd training. This excludes sample extraction, matching the PQ-training timer boundary in the DiskANN integration.
- `encoding_seconds`: codebook preparation, a second input scan, centering of each L2 batch, and final PQ assignment/encoding.
- `total_seconds`: end-to-end elapsed time measured by the standalone process, including sampling, training, output writing, and encoding.

## Reproducibility notes

- Fix both `--sample-seed` and `--kmeans-seed` when comparing thread counts.
- Keep the compiler, CPU target flags, OpenMP runtime, and input file identical across runs.
- The implementation is deterministic in its sample selection and k-means++ random choices for a fixed standard-library implementation. Floating-point reduction order can still vary with thread count and compiler/runtime.
- Do not compare a cold-cache run with a warm-cache run without reporting the difference. The standalone program does not prewarm the input file automatically.

## Source navigation

See [docs/algorithm_mapping.md](docs/algorithm_mapping.md) for a direct mapping from the paper's three techniques to the corresponding standalone functions.

## Scope and limitations

- This is a standalone **PQ construction** tool, not an ANN index builder or search engine.
- No FAISS, Starling, or DiskANN integration is included; this package is the independent PQ builder only. It does not depend on those frameworks.
- Input is currently float32 DiskANN-style `.bin` only.
- OPQ, residual PQ, GPU execution, and multi-byte codes (`K > 256`) are not implemented in this standalone package.
- The codebook training path is the PQ-specific L2 Lloyd path. Metric-aware IP/cosine support is currently applied to final encoding as described above.

## Licensing

The DiskANN-derived portions are identified in `THIRD_PARTY_NOTICES.md`, with the MIT text included at `LICENSES/DiskANN-MIT.txt`. This package does not assert a project-wide license for RCS-PQ-specific contributions; choose and add one before a public release if required.
