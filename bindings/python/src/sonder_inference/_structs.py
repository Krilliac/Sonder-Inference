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


MAX_CHAT_MESSAGES = 1024


class CChatMessage(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("role", c.c_char_p),
        ("content", c.c_char_p),
    ]
