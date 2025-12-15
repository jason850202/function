from hxr_analysis.workbench.waveform_display.transform import IdSpec, compute_identifier


def test_identifier_from_meta_paths():
    payload = {"meta": {"shot": 123, "scope_id": "S1"}}
    spec = IdSpec(["meta.shot", "meta.scope_id"])
    uid = compute_identifier(payload, spec)
    assert uid == "123__S1"
