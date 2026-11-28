#!/usr/bin/env bash
# Configure, build, and (optionally) test Sonder Inference on Linux / WSL2.
#
# Examples:
#   ./scripts/build.sh --preset ci-linux --test
#   ./scripts/build.sh --preset linux-cuda-blackwell --jobs 8 --test
#   ./scripts/build.sh --preset linux-rocm-xt --clean -- -DSONDER_WARNINGS_AS_ERRORS=ON
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

PRESET="linux-release"
RUN_TESTS=0
CLEAN=0
JOBS=""
EXTRA_CMAKE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/build.sh [options] [-- <extra-cmake-args>...]

Options:
  --preset <name>         CMake preset name (default: linux-release)
  --test                  Run ctest --preset <name> after building
  --clean                 Remove build/<preset> before configuring
  --jobs, -j <N>          Parallel build/test worker count (> 0)
  --help, -h              Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            [[ $# -ge 2 ]] || { echo "error: --preset requires a value" >&2; exit 2; }
            PRESET="$2"
            shift 2
            ;;
        --preset=*)
            PRESET="${1#*=}"
            shift
            ;;
        --test)
            RUN_TESTS=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --llamacpp-src|--llamacpp-src=*)
            echo "error: --llamacpp-src is retired: Sonder Inference builds only the pinned llama.cpp archive + third_party/ggml-patches/ (use -DSONDER_LLAMACPP_URL=/path/to/archive.tar.gz for offline builds)" >&2
            exit 2
            ;;
        --jobs|-j)
            [[ $# -ge 2 ]] || { echo "error: $1 requires a positive integer" >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_CMAKE_ARGS+=("$1")
                shift
            done
            ;;
        -D*)
            EXTRA_CMAKE_ARGS+=("$1")
            shift
            ;;
        *)
            echo "error: unknown option '$1' (pass '--' before arbitrary CMake arguments)" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${PRESET}" ]]; then
    echo "error: --preset must not be empty" >&2
    exit 2
fi
if [[ "${PRESET}" == *"/"* || "${PRESET}" == *"\\"* || "${PRESET}" == *".."* ]]; then
    echo "error: invalid preset name '${PRESET}'" >&2
    exit 2
fi
if [[ -n "${JOBS}" ]]; then
    if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: --jobs must be a positive integer (got '${JOBS}')" >&2
        exit 2
    fi
fi

# Guard against stray agent-shell CC/CXX values pointing at non-compilers
# (e.g. CC=claude.exe inherited into WSL), while preserving valid compiler overrides.
for var_name in CC CXX; do
    val="${!var_name:-}"
    if [[ -n "${val}" ]]; then
        base="$(basename "${val}")"
        if [[ "${base}" == claude* ]]; then
            echo "[sonder-build] clearing non-compiler ${var_name}=${val}" >&2
            unset "${var_name}"
        fi
    fi
done

print_gpu_toolchains() {
    case "${PRESET}" in
        *cuda*|*rocm*|*vulkan*) ;;
        *) return 0 ;;
    esac

    echo "[sonder-build] GPU preset '${PRESET}' toolchain probe:"
    if [[ -e /dev/dxg ]]; then
        echo "  WSL2   : /dev/dxg detected (WSL2 GPU paravirtualization)"
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path nvcc_rel
        nvcc_path="$(command -v nvcc)"
        nvcc_rel="$(nvcc --version 2>/dev/null | grep -E 'release [0-9]+\.[0-9]+' | tail -n 1 | sed 's/^[[:space:]]*//' || true)"
        echo "  CUDA   : ${nvcc_path}${nvcc_rel:+ (${nvcc_rel})}"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        echo "  CUDA   : /usr/local/cuda/bin/nvcc (not on PATH)"
    else
        echo "  CUDA   : not detected"
    fi

    local kfd_state="absent"
    if [[ -r /dev/kfd && -w /dev/kfd ]]; then
        kfd_state="accessible (/dev/kfd)"
    elif [[ -e /dev/kfd ]]; then
        kfd_state="present (check render/video group permissions on /dev/kfd)"
    fi
    if command -v hipcc >/dev/null 2>&1; then
        echo "  ROCm   : $(command -v hipcc) [kfd: ${kfd_state}]"
    elif [[ -x /opt/rocm/bin/hipcc ]]; then
        echo "  ROCm   : /opt/rocm/bin/hipcc (not on PATH) [kfd: ${kfd_state}]"
    else
        echo "  ROCm   : not detected [kfd: ${kfd_state}]"
    fi

    if command -v glslc >/dev/null 2>&1; then
        echo "  Vulkan : $(command -v glslc)${VULKAN_SDK:+ (VULKAN_SDK=${VULKAN_SDK})}"
    elif [[ -n "${VULKAN_SDK:-}" ]]; then
        echo "  Vulkan : VULKAN_SDK=${VULKAN_SDK} (glslc not on PATH)"
    elif command -v vulkaninfo >/dev/null 2>&1; then
        echo "  Vulkan : runtime detected ($(command -v vulkaninfo)); glslc not on PATH"
    else
        echo "  Vulkan : not detected"
    fi
}

print_gpu_toolchains

CONFIGURE_ARGS=(--preset "${PRESET}")
if [[ ${#EXTRA_CMAKE_ARGS[@]} -gt 0 ]]; then
    CONFIGURE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
fi

BUILD_ARGS=(--build --preset "${PRESET}")
TEST_ARGS=(--preset "${PRESET}")
if [[ -n "${JOBS}" ]]; then
    BUILD_ARGS+=(--parallel "${JOBS}")
    TEST_ARGS+=(--parallel "${JOBS}")
fi

BINARY_DIR="${REPO_ROOT}/build/${PRESET}"
if [[ "${CLEAN}" -eq 1 && -d "${BINARY_DIR}" ]]; then
    rm -rf "${BINARY_DIR}"
fi

cd "${REPO_ROOT}"
cmake "${CONFIGURE_ARGS[@]}"
cmake "${BUILD_ARGS[@]}"
if [[ "${RUN_TESTS}" -eq 1 ]]; then
    ctest "${TEST_ARGS[@]}"
fi
