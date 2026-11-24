"""ctypes mirrors of the structs in include/sonder_inference.h."""

from __future__ import annotations

import ctypes

c = ctypes


class EngineOptions(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("telemetry_level", c.c_int),
        ("telemetry_jsonl_path", c.c_char_p),
        ("capture_text", c.c_int32),
    ]


class CLogitBias(c.Structure):
    _fields_ = [("token", c.c_int32), ("bias", c.c_float)]


# Original layout (ABI v1 as first published): fields up to max_tokens.
_SAMPLING_V1_FIELDS = [
    ("struct_size", c.c_uint32),
    ("temperature", c.c_float),
    ("top_p", c.c_float),
    ("top_k", c.c_int32),
    ("min_p", c.c_float),
    ("repeat_penalty", c.c_float),
    ("has_seed", c.c_int32),
    ("seed", c.c_uint64),
    ("max_tokens", c.c_int32),
]


class CSamplingConfigV1(c.Structure):
    _fields_ = _SAMPLING_V1_FIELDS


class CSamplingConfig(c.Structure):
    _fields_ = _SAMPLING_V1_FIELDS + [
        ("typical_p", c.c_float),
        ("presence_penalty", c.c_float),
        ("frequency_penalty", c.c_float),
        ("repeat_last_n", c.c_int32),
        ("num_ctx", c.c_int32),
        ("logit_bias", c.POINTER(CLogitBias)),
        ("logit_bias_count", c.c_size_t),
    ]


#: struct_size a caller built against the original layout would send.
SAMPLING_CONFIG_V1_SIZE = c.sizeof(CSamplingConfigV1)
SAMPLING_CONFIG_SIZE = c.sizeof(CSamplingConfig)


class CGenerationStats(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("outcome", c.c_int),
        ("prompt_tokens", c.c_uint64),
        ("completion_tokens", c.c_uint64),
        ("chunks", c.c_uint64),
        ("ttft_ms", c.c_double),
        ("total_ms", c.c_double),
    ]


# int (*)(void* user_data, const char* text, size_t length). The text is not
# NUL-terminated, so it is received as a raw pointer (c_void_p), not c_char_p.
TOKEN_CALLBACK = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_size_t)

# void (*)(void* user_data, const char* json_line, size_t length).
# The JSON line is not guaranteed NUL-terminated, so receive as c_void_p.
TELEMETRY_CALLBACK = c.CFUNCTYPE(None, c.c_void_p, c.c_void_p, c.c_size_t)


MAX_CHAT_MESSAGES = 1024


class CChatMessage(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("role", c.c_char_p),
        ("content", c.c_char_p),
    ]


class CSessionMetadata(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("session_id", c.c_char_p),
        ("run_id", c.c_char_p),
        ("agent_id", c.c_char_p),
        ("task_id", c.c_char_p),
    ]


_REQUEST_METADATA_V1_FIELDS = [
    ("struct_size", c.c_uint32),
    ("parent_request_id", c.c_char_p),
]


class CRequestMetadataV1(c.Structure):
    _fields_ = _REQUEST_METADATA_V1_FIELDS


class CRequestMetadata(c.Structure):
    _fields_ = _REQUEST_METADATA_V1_FIELDS + [
        ("workload_class", c.c_int32),
        ("priority", c.c_int32),
    ]


REQUEST_METADATA_V1_SIZE = c.sizeof(CRequestMetadataV1)
REQUEST_METADATA_SIZE = c.sizeof(CRequestMetadata)


class CGpuInfo(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("id", c.c_char * 64),
        ("name", c.c_char * 256),
        ("vendor", c.c_char * 32),
        ("backend_hint", c.c_char * 32),
        ("source", c.c_char * 32),
        ("vendor_id", c.c_uint32),
        ("device_id", c.c_uint32),
        ("dedicated_total_bytes", c.c_uint64),
        ("dedicated_free_bytes", c.c_uint64),
        ("shared_total_bytes", c.c_uint64),
        ("effective_vram_mib", c.c_uint64),
        ("effective_free_vram_mib", c.c_uint64),
        ("has_dedicated_total", c.c_int32),
        ("has_dedicated_free", c.c_int32),
        ("has_shared_total", c.c_int32),
        ("has_effective_vram", c.c_int32),
        ("has_effective_free_vram", c.c_int32),
        ("is_integrated_uma", c.c_int32),
    ]


GPU_INFO_SIZE = c.sizeof(CGpuInfo)
