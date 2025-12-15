import pytest

from hxr_analysis.workbench.waveform_display.transform import IdSpec, MergeSpec, merge_payloads


def test_merge_collision_attach_id():
    payload1 = {
        "data": {"time": [1, 2], "channels": {"A": [1, 2]}},
        "meta": {"shot": 1},
    }
    payload2 = {
        "data": {"time": [1, 2], "channels": {"A": [3, 4]}},
        "meta": {"shot": 2},
    }

    spec = MergeSpec(
        target_path="data.channels",
        merge_mode="dict_union",
        id_spec=IdSpec(["meta.shot"]),
        collision_policy="attach_id",
        collision_template="{key}@{uid}",
    )

    merged = merge_payloads([payload1, payload2], spec=spec)
    assert set(merged["data"]["channels"].keys()) == {"A@1", "A@2"}


def test_merge_timebase_mismatch_raises():
    payload1 = {"data": {"time": [1, 2], "channels": {}}, "meta": {}}
    payload2 = {"data": {"time": [1, 3], "channels": {}}, "meta": {}}

    spec = MergeSpec(target_path="data.channels", merge_mode="dict_union", id_spec=IdSpec(["meta.shot"]))

    with pytest.raises(Exception):
        merge_payloads([payload1, payload2], spec=spec)
