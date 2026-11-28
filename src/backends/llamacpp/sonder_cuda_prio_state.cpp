// Thread-scoped CUDA stream priority state shared between sonder_backend_llamacpp
// and ggml-cuda (linked downstream of both targets to preserve static link order).
#include "sonder_cuda_prio_state.hpp"

#include <atomic>

namespace {

thread_local int g_tls_prio_hint = SONDER_CUDA_STREAM_PRIO_DEFAULT;

std::atomic<std::uint64_t> g_high_scopes{0};
std::atomic<std::uint64_t> g_low_scopes{0};
std::atomic<std::uint64_t> g_default_scopes{0};
std::atomic<std::uint64_t> g_priority_streams_created{0};
std::atomic<std::uint64_t> g_fallback_streams_created{0};
std::atomic<int> g_last_device{-1};
std::atomic<int> g_last_numeric_priority{0};
std::atomic<int> g_preinclude_hook_compiled{0};

}  // namespace

extern "C" {

int sonder_ggml_cuda_stream_priority_hint(void) noexcept {
    return g_tls_prio_hint;
}

int sonder_ggml_cuda_set_stream_priority_hint(int hint) noexcept {
    const int prev = g_tls_prio_hint;
    g_tls_prio_hint = hint;
    return prev;
}

void sonder_ggml_cuda_record_hint_scope(int hint) noexcept {
    if (hint == SONDER_CUDA_STREAM_PRIO_HIGH) {
        g_high_scopes.fetch_add(1u, std::memory_order_relaxed);
    } else if (hint == SONDER_CUDA_STREAM_PRIO_LOW) {
        g_low_scopes.fetch_add(1u, std::memory_order_relaxed);
    } else {
        g_default_scopes.fetch_add(1u, std::memory_order_relaxed);
    }
}

void sonder_ggml_cuda_record_stream_created(int device, int numeric_priority, int used_priority_api) noexcept {
    g_last_device.store(device, std::memory_order_relaxed);
    g_last_numeric_priority.store(numeric_priority, std::memory_order_relaxed);
    if (used_priority_api != 0) {
        g_priority_streams_created.fetch_add(1u, std::memory_order_relaxed);
    } else {
        g_fallback_streams_created.fetch_add(1u, std::memory_order_relaxed);
    }
}

void sonder_ggml_cuda_mark_hook_compiled(void) noexcept {
    g_preinclude_hook_compiled.store(1, std::memory_order_relaxed);
}

void sonder_ggml_cuda_get_stream_stats(SonderGgmlCudaStreamStats* out) noexcept {
    if (out == nullptr) return;
    out->high_priority_scopes = g_high_scopes.load(std::memory_order_relaxed);
    out->low_priority_scopes = g_low_scopes.load(std::memory_order_relaxed);
    out->default_priority_scopes = g_default_scopes.load(std::memory_order_relaxed);
    out->priority_streams_created = g_priority_streams_created.load(std::memory_order_relaxed);
    out->fallback_streams_created = g_fallback_streams_created.load(std::memory_order_relaxed);
    out->last_device = g_last_device.load(std::memory_order_relaxed);
    out->last_numeric_priority = g_last_numeric_priority.load(std::memory_order_relaxed);
    out->preinclude_hook_compiled = g_preinclude_hook_compiled.load(std::memory_order_relaxed);
}

void sonder_ggml_cuda_reset_stream_stats(void) noexcept {
    g_high_scopes.store(0u, std::memory_order_relaxed);
    g_low_scopes.store(0u, std::memory_order_relaxed);
    g_default_scopes.store(0u, std::memory_order_relaxed);
    g_priority_streams_created.store(0u, std::memory_order_relaxed);
    g_fallback_streams_created.store(0u, std::memory_order_relaxed);
    g_last_device.store(-1, std::memory_order_relaxed);
    g_last_numeric_priority.store(0, std::memory_order_relaxed);
}

}  // extern "C"
