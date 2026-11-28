// Thread-scoped CUDA stream priority state shared between sonder_backend_llamacpp
// and ggml-cuda (linked downstream of both targets to preserve static link order).
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

enum SonderCudaStreamPriorityHint {
    SONDER_CUDA_STREAM_PRIO_DEFAULT = 0,
    SONDER_CUDA_STREAM_PRIO_HIGH = 1,
    SONDER_CUDA_STREAM_PRIO_LOW = 2,
};

struct SonderGgmlCudaStreamStats {
    std::uint64_t high_priority_scopes;
    std::uint64_t low_priority_scopes;
    std::uint64_t default_priority_scopes;
    std::uint64_t priority_streams_created;
    std::uint64_t fallback_streams_created;
    int last_device;
    int last_numeric_priority;
    int preinclude_hook_compiled;
};

int sonder_ggml_cuda_stream_priority_hint(void) noexcept;
int sonder_ggml_cuda_set_stream_priority_hint(int hint) noexcept;
void sonder_ggml_cuda_record_hint_scope(int hint) noexcept;
void sonder_ggml_cuda_record_stream_created(int device, int numeric_priority, int used_priority_api) noexcept;
void sonder_ggml_cuda_mark_hook_compiled(void) noexcept;
void sonder_ggml_cuda_get_stream_stats(SonderGgmlCudaStreamStats* out) noexcept;
void sonder_ggml_cuda_reset_stream_stats(void) noexcept;

#ifdef __cplusplus
}
#endif
