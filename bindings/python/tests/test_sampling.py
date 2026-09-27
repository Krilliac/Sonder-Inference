import ctypes
import math

import pytest

import sonder_inference as si
from sonder_inference import _structs as S


def test_defaults_come_from_library(lib):
    d = si.SamplingConfig.defaults()
    assert d.temperature == pytest.approx(0.8)
    assert d.top_k == 40
    assert d.typical_p == 1.0
    assert d.repeat_last_n == 64
    assert d.presence_penalty == 0.0 and d.frequency_penalty == 0.0
    assert d.num_ctx == 0
    assert d.logit_bias is None
    d.validate()
    si.SamplingConfig().validate()
    si.SamplingConfig.greedy(16, 3).validate()


def test_unset_fields_keep_library_defaults(lib):
    c = si.SamplingConfig(top_k=7)._to_c(lib).struct
    assert c.top_k == 7
    assert c.temperature == pytest.approx(0.8)
    assert c.has_seed == 0
    assert c.struct_size == S.SAMPLING_CONFIG_SIZE


def test_new_fields_and_logit_bias_marshal(lib):
    cfg = si.SamplingConfig(typical_p=0.9, presence_penalty=0.5, frequency_penalty=-0.25,
                            repeat_last_n=-1, num_ctx=4096, seed=2**63 + 5,
                            logit_bias={5: 2.5, 9: -math.inf})
    m = cfg._to_c(lib)
    c = m.struct
    assert c.typical_p == pytest.approx(0.9)
    assert c.repeat_last_n == -1 and c.num_ctx == 4096
    assert c.has_seed == 1 and c.seed == 2**63 + 5
    assert c.logit_bias_count == 2
    assert (c.logit_bias[0].token, c.logit_bias[0].bias) == (5, 2.5)
    assert c.logit_bias[1].token == 9 and math.isinf(c.logit_bias[1].bias)
    cfg.validate()
    # Sequence-of-pairs form is accepted too.
    si.SamplingConfig(logit_bias=[(1, 1.0), (2, -1.0)]).validate()


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"top_p": 0.0}, "top_p"),
        ({"temperature": -1.0}, "temperature"),
        ({"typical_p": 0.0}, "typical_p"),
        ({"presence_penalty": 3.0}, "presence_penalty"),
        ({"frequency_penalty": float("nan")}, "frequency_penalty"),
        ({"repeat_last_n": -2}, "repeat_last_n"),
        ({"num_ctx": -1}, "num_ctx"),
        ({"logit_bias": {-1: 1.0}}, "logit_bias"),
        ({"logit_bias": [(3, 1.0), (3, 2.0)]}, "logit_bias"),
        ({"logit_bias": {3: 101.0}}, "logit_bias"),
        ({"max_tokens": 0}, "max_tokens"),
    ],
)
def test_library_validation_errors(lib, changes, field):
    with pytest.raises(si.InvalidArgumentError) as info:
        si.SamplingConfig(**changes).validate()
    assert field in str(info.value)
    assert info.value.status == si.Status.INVALID_ARGUMENT


def test_python_side_type_and_range_checks(lib):
    with pytest.raises(si.InvalidArgumentError, match="int32"):
        si.SamplingConfig(num_ctx=2**31).validate()
    with pytest.raises(TypeError):
        si.SamplingConfig(top_k=1.5).validate()  # type: ignore[arg-type]
    with pytest.raises(si.InvalidArgumentError, match="seed"):
        si.SamplingConfig(seed=-1).validate()
    with pytest.raises(TypeError, match="pair"):
        si.SamplingConfig(logit_bias=[1, 2]).validate()  # type: ignore[list-item]


def test_legacy_struct_size_ignores_appended_fields(lib):
    # An old caller sends the original struct_size; the library must ignore
    # the tail, so an otherwise-invalid typical_p passes validation.
    m = si.SamplingConfig(typical_p=0.0)._to_c(lib, struct_size=S.SAMPLING_CONFIG_V1_SIZE)
    assert lib.cdll.sonder_sampling_config_validate(si._lib.ctypes.byref(m.struct)) == 0
    m = si.SamplingConfig()._to_c(lib, struct_size=S.SAMPLING_CONFIG_V1_SIZE - 1)
    assert lib.cdll.sonder_sampling_config_validate(si._lib.ctypes.byref(m.struct)) == int(
        si.Status.INVALID_ARGUMENT)


def test_old_library_rejects_extended_fields_client_side(lib, monkeypatch):
    monkeypatch.setattr(lib, "_extended_sampling", False)
    c = si.SamplingConfig(top_k=5)._to_c(lib).struct
    assert c.struct_size == S.SAMPLING_CONFIG_V1_SIZE
    with pytest.raises(si.UnsupportedError, match="predates"):
        si.SamplingConfig(num_ctx=1024)._to_c(lib)


def test_defaults_from_pre_extension_library_leave_appended_fields_unset(lib, monkeypatch):
    # A pre-extension ABI-v1 library's init writes only the original prefix,
    # leaving the appended fields zeroed (typical_p=0.0 is invalid).
    # (typical_p sits inside the V1 struct's tail padding, so the old
    # library's `*config = {}` zeroes it.)
    real_init = lib.cdll.sonder_sampling_config_init
    tail = S.CSamplingConfig.typical_p.offset

    def old_init(ptr):
        real_init(ptr)
        c = ptr._obj
        ctypes.memset(ctypes.addressof(c) + tail, 0, ctypes.sizeof(c) - tail)
        c.struct_size = S.SAMPLING_CONFIG_V1_SIZE

    monkeypatch.setattr(lib.cdll, "sonder_sampling_config_init", old_init)
    monkeypatch.setattr(lib, "_extended_sampling", False)
    d = si.SamplingConfig.defaults(lib)
    assert d.temperature == pytest.approx(0.8) and d.top_k == 40
    for name in ("typical_p", "presence_penalty", "frequency_penalty", "repeat_last_n", "num_ctx"):
        assert getattr(d, name) is None, name
    assert not d.uses_extended_fields()
    d.validate(lib)  # must not raise UnsupportedError


def test_to_dict_and_replace():
    cfg = si.SamplingConfig.greedy(8).replace(logit_bias=[(1, 2.0)])
    d = cfg.to_dict()
    assert d["max_tokens"] == 8 and d["logit_bias"] == {1: 2.0}
    assert not si.SamplingConfig().uses_extended_fields()
    assert si.SamplingConfig(num_ctx=1).uses_extended_fields()
