import pytest

from hxr_analysis.preprocessing.background_subtract import (
    BackgroundSubtractParams,
    background_subtract_one,
)
from hxr_analysis.workbench.waveform_display.np_compat import np


def _payload(time, channels):
    return {"data": {"time": np.asarray(time), "channels": channels}, "meta": {}}


def test_bg_subtract_equal_time_by_key():
    exp = _payload([0, 1, 2], {"A": np.asarray([1.0, 2.0, 3.0])})
    bg = _payload([0, 1, 2], {"A": np.asarray([0.5, 0.5, 0.5])})
    params = BackgroundSubtractParams()

    result = background_subtract_one(exp, bg, params=params)
    out = result["data"]["channels"]["A"]
    assert hasattr(np, "allclose") and np.allclose(out, [0.5, 1.5, 2.5]) or list(out) == [0.5, 1.5, 2.5]


def test_bg_subtract_missing_channel_skip():
    exp = _payload([0, 1], {"A": np.asarray([1.0, 2.0]), "B": np.asarray([3.0, 4.0])})
    bg = _payload([0, 1], {"A": np.asarray([0.5, 0.5])})
    params = BackgroundSubtractParams(missing_channel_policy="skip")

    result = background_subtract_one(exp, bg, params=params)
    channels = result["data"]["channels"]
    assert "A" in channels and "B" in channels
    assert hasattr(np, "allclose") and np.allclose(channels["A"], [0.5, 1.5]) or list(channels["A"]) == [0.5, 1.5]
    assert hasattr(np, "allclose") and np.allclose(channels["B"], [3.0, 4.0]) or list(channels["B"]) == [3.0, 4.0]


def test_bg_subtract_time_interp():
    exp = _payload([0.0, 1.0, 2.0], {"A": np.asarray([10.0, 11.0, 12.0])})
    bg = _payload([0.0, 2.0, 4.0], {"A": np.asarray([0.0, 2.0, 4.0])})
    params = BackgroundSubtractParams(time_align="interp_bg_to_exp")

    result = background_subtract_one(exp, bg, params=params)
    out = result["data"]["channels"]["A"]
    expected = [10.0, 10.0, 10.0]
    assert hasattr(np, "allclose") and np.allclose(out, expected) or list(out) == expected


def test_bg_subtract_records_history_and_bg_ref():
    exp = _payload([0, 1], {"A": np.asarray([1.0, 2.0])})
    bg = _payload([0, 1], {"A": np.asarray([0.5, 0.5])})
    params = BackgroundSubtractParams()

    result = background_subtract_one(exp, bg, params=params)
    meta = result.get("meta", {})
    assert "__history__" in meta
    history = meta["__history__"]
    assert any(entry.get("op_name") == "background_subtract" for entry in history)
    assert meta.get("__background__", {}).get("uid") is not None
