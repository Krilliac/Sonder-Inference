# Sonder Inference `ggml` / `llama.cpp` Patch Series

Sonder Inference compiles a single SHA-256-pinned `ggml-org/llama.cpp` source
archive (`cmake/llamacpp-pin.json`) into `sonder-infer` and applies its own
hash-verified patch series from this directory at configure time via
`cmake/ApplyGgmlPatches.cmake`. Separate `llama.cpp` forks, standalone
`llama-server` binaries, and external local-checkout overrides are retired.

## Rules for patches in this directory

1. **File naming and ordering:** `NNNN-<slug>.patch`, where `NNNN` is a strictly
   increasing 4-digit sequence (`0001`, `0002`, ...) and `<slug>` uses lowercase
   ASCII letters, digits, `.`, `_`, or `-`.
2. **Manifest registration (`cmake/llamacpp-pin.json`):** every `.patch` file in
   this directory must be listed in `tracks.mainline.patches` in exact `NNNN`
   order with:
   - `"file"`: filename in `third_party/ggml-patches/`;
   - `"sha256"`: 64-character lowercase SHA-256 digest of the `.patch` file;
   - `"summary"`: concise explanation of what changes and why;
   - `"upstream"`: upstream PR/issue URL (so the patch can be dropped on a
     future pin bump) or `"sonder-owned"` for SI-specific kernels.
   Unlisted `.patch` files, digest mismatches, path traversal (`..`, absolute
   paths, `.git/`), symlink modes (`120000`), and binary patches fail closed in
   both `cmake/ApplyGgmlPatches.cmake` and `python -I scripts/llamacpp_pin.py check`.
3. **Verification:**
   - `python -I scripts/llamacpp_pin.py check` verifies the manifest, patch
     hashes, directory hygiene, and documentation consistency.
   - `python -I scripts/llamacpp_pin.py verify --tarball <tar.gz> --git-dir <repo> --commit <sha>`
     verifies every blob in the upstream archive against the pinned git commit
     and verifies that the entire patch series applies cleanly (`git apply --check -p1`).
   - Every patched kernel or model loader behavior must be accompanied by a
     deterministic C++ unit or numeric parity test under `src/backends/llamacpp/tests/`.
4. **Licensing (`docs/LICENSE_REVIEW.md`):** upstream `llama.cpp`/`ggml` files
   remain MIT (`NOTICE`). Choosing the license of SI's patch series (before
   populating non-empty `.patch` files here) and any git subtree vendoring
   require a review and explicit owner approval recorded in
   `docs/LICENSE_REVIEW.md`. Do not copy third-party fork code without
   verifying its exact revision and license and recording owner approval there.

## Current series

Empty (`"patches": []`). Stock `v1.0.0` already includes `GGML_TYPE_Q2_0`
(group 64), `GGML_TYPE_Q1_0`, `GGML_TYPE_NVFP4`, `GGML_TYPE_F8_E4M3`, the
`prism.hadamard.*` loader, and CPU/CUDA/Vulkan FWHT kernels; `PQ2_0` (group 128)
checkpoints convert losslessly to stock `Q2_0` via `sonder-infer reblock-pq2`.
