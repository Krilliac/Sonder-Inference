// Pre-included into ggml-cuda translation units when GGML_CUDA=ON (non-HIP/MUSA).
//
// Verified against llama.cpp v1.0.0:
//   - ggml/src/ggml-cuda/common.cuh:1395 and allreduce.cu:445 create streams via
//     cudaStreamCreateWithFlags(..., cudaStreamNonBlocking) with default priority 0.
//   - ggml/src/ggml-cuda/ggml-cuda.cu:3386 and :5392 launch captured CUDA graphs on
//     cuda_ctx->stream() via cudaGraphLaunch(instance, cuda_ctx->stream()) without
//     overriding node-level cudaLaunchAttributePriority, so captured graphs inherit
//     the priority of cuda_ctx->stream().
//
// When Sonder Inference activates a thread-scoped stream priority hint around
// context creation or execution (HIGH for main decode, LOW for aux/embed/rerank/STT),
// this shim queries cudaDeviceGetStreamPriorityRange(&least, &greatest) and calls
// cudaStreamCreateWithPriority, falling back cleanly to cudaStreamCreateWithFlags
// if stream priorities are unsupported on the active device/driver.
#pragma once

#if defined(__CUDACC__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_runtime.h>

#if !defined(__CUDA_ARCH__)
extern "C" {
int sonder_ggml_cuda_stream_priority_hint(void) noexcept;
void sonder_ggml_cuda_record_stream_created(int device, int numeric_priority, int used_priority_api) noexcept;
void sonder_ggml_cuda_mark_hook_compiled(void) noexcept;
}

static inline cudaError_t sonder_ggml_cuda_stream_create_with_flags(cudaStream_t* p_stream,
                                                                    unsigned int flags) {
    sonder_ggml_cuda_mark_hook_compiled();
    const int hint = sonder_ggml_cuda_stream_priority_hint();
    if (hint == 1 || hint == 2) {
        int least_prio = 0;
        int greatest_prio = 0;
        if (::cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio) == cudaSuccess &&
            least_prio != greatest_prio) {
            const int hi = greatest_prio < least_prio ? greatest_prio : least_prio;
            const int lo = least_prio > greatest_prio ? least_prio : greatest_prio;
            const int target_prio = (hint == 1) ? hi : lo;
            const cudaError_t prio_err = ::cudaStreamCreateWithPriority(p_stream, flags, target_prio);
            if (prio_err == cudaSuccess) {
                int dev = -1;
                (void)::cudaGetDevice(&dev);
                sonder_ggml_cuda_record_stream_created(dev, target_prio, 1);
                return cudaSuccess;
            }
            (void)::cudaGetLastError();
        } else {
            (void)::cudaGetLastError();
        }
    }
    const cudaError_t err = ::cudaStreamCreateWithFlags(p_stream, flags);
    if (err == cudaSuccess) {
        int dev = -1;
        (void)::cudaGetDevice(&dev);
        sonder_ggml_cuda_record_stream_created(dev, 0, 0);
    }
    return err;
}

#define cudaStreamCreateWithFlags sonder_ggml_cuda_stream_create_with_flags
#endif  // !defined(__CUDA_ARCH__)
#endif  // defined(__CUDACC__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
