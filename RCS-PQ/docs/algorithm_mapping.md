# RCS-PQ implementation map

This document maps the paper's three construction techniques to the standalone source. Function names refer to `src/rcspq.cpp`.

| Paper component | Standalone implementation | What to inspect |
|---|---|---|
| Ranking-aware scoring reformulation | `prepare_codebook()` and `assign_centroid()` | For L2, precomputes `0.5 * ||c||^2` and evaluates `bias - x dot c`; selected score is reused by `run_lloyds()` to reconstruct the L2 residual as `||x||^2 + 2 * selected_score`. |
| Centroid-parallel SIMD | `CodebookView`, `prepare_codebook()`, `assign_centroid()` | The SoA codebook stores `[subdimension][padded_centroid]`; AVX-512 lanes correspond to 16 candidate centroids, AVX2 to 8. For subdimensions `<= 8`, the inner dimension loop has an unrolled path. |
| Cache-friendly chunk-centric execution | `build()` encoding stage and `assign_batch_strided()` | Each chunk codebook is prepared once. For each input batch, the code iterates over chunks, streams vector rows through the assignment kernel, and writes one byte per selected centroid. No full pairwise score table is materialized. |
| Thread-scalable PQ training | `run_lloyds()`, `kmeanspp_select()`, `compute_mean_and_center()` | Balanced `schedule(static)` work distribution; per-thread centroid sums/counts; reduced updates; deterministic block-based k-means++ sampling. |

## Construction flow

1. `sample_training_data()` scans the input matrix and creates a deterministic Bernoulli sample.
2. `compute_mean_and_center()` computes the global mean and centers the training sample for L2.
3. `make_chunk_offsets()` divides the vector dimensions into balanced contiguous chunks.
4. For each chunk, `kmeanspp_select()` initializes centroids and `run_lloyds()` trains the codebook.
5. `prepare_codebook()` transforms each codebook into the centroid-parallel SoA layout.
6. The encoding loop in `build()` streams batches and processes one chunk at a time through `assign_batch_strided()`.
7. `pivots.bin`, `centroid.bin`, `chunk_offsets.bin`, and `codes.bin` are written to the output directory.
