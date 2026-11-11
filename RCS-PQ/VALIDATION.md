# Validation record

Validation was performed on a small synthetic float32 matrix (600 vectors, 12 dimensions) and an API smoke-test matrix (1,000 vectors, 12 dimensions). These tests verify buildability and the end-to-end PQ output format; they are not substitutes for rerunning the paper's large SIFT experiments on the target server.

Validated configurations in the build environment:

- Native x86 build with `-march=native` (host exposes AVX-512F): build passed; smoke test passed.
- AVX2 build with `-mavx2 -mfma`: build passed; smoke test passed.
- Portable baseline build without native ISA flags (scalar assignment fallback): build passed; smoke test passed.
- L2, inner-product, and cosine final-encoding paths: smoke test passed. The training objective remains L2 Lloyd for all three, as documented in `README.md`.
- CLI invocation: passed; output matrix headers and file sizes were checked for `pivots.bin`, `centroid.bin`, `chunk_offsets.bin`, and `codes.bin`.

Not validated in this environment:

- Full SIFT10M-256D/512D/1024D runs.
- Bit-for-bit identity with the DiskANN integration's pivot file container. The standalone package writes documented independent matrix files rather than DiskANN's packed pivot metadata file.
- Performance numbers on the user's Xeon Gold 6230 server.
