from hxr_analysis.workbench.waveform_display.path import PathResolutionError, resolve_path


def test_resolve_nested_dict():
    payload = {"data": {"time": [1, 2, 3], "channels": {"A": [4, 5, 6]}}}
    assert resolve_path(payload, "data.time") == [1, 2, 3]
    assert resolve_path(payload, "data.channels.A") == [4, 5, 6]
    assert resolve_path(payload, "data.channels['A']") == [4, 5, 6]


def test_resolve_errors():
    payload = {"data": {}}
    try:
        resolve_path(payload, "data.missing")
    except PathResolutionError as exc:
        assert "Missing key" in str(exc)
    else:
        assert False, "Expected PathResolutionError"

    try:
        resolve_path(payload, "")
    except PathResolutionError:
        pass
    else:
        assert False, "Expected error on empty path"
