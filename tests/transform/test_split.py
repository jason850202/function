from hxr_analysis.workbench.waveform_display.transform import IdSpec, SplitSpec, split_payload


def test_split_channels_dict():
    payload = {
        "data": {"channels": {"A": [1, 2], "B": [3, 4]}},
        "meta": {"shot": 1, "scope": "S"},
    }
    spec = SplitSpec(
        source_path="data.channels",
        split_mode="dict_keys",
        child_payload_target_path="data.channels",
        id_spec=IdSpec(["meta.shot", "meta.scope"]),
    )

    outputs = split_payload(payload, spec=spec)
    assert set(outputs.keys()) == {"1__S__A", "1__S__B"}
    for child_id, child_payload in outputs.items():
        channels = child_payload["data"]["channels"]
        assert len(channels) == 1
        assert child_payload["meta"]["__uid__"] == child_id
