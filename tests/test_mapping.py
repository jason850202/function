from hxr_analysis.workbench.waveform_display.mapping import (
    MappingValidationError,
    PlotMapping,
    validate_and_resolve,
)
from hxr_analysis.workbench.waveform_display.models import PayloadStore, create_waveform_payload
from hxr_analysis.workbench.waveform_display.np_compat import np


def build_store():
    time = np.linspace(0, 1, 3)
    channels = {"A": np.asarray([1, 2, 3]), "B": np.asarray([1, 2])}
    store = PayloadStore()
    store.add("demo", create_waveform_payload(time=time, channels=channels))
    return store


def test_validate_success():
    store = build_store()
    mapping = PlotMapping(
        plot_type="curve",
        payload_id="demo",
        x_path="data.time",
        y_path="data.channels.A",
        style="{\"pen\": \"r\"}",
    )
    resolved = validate_and_resolve(mapping, store)
    assert resolved.x.shape[0] == resolved.y.shape[0]
    assert resolved.style["pen"] == "r"


def test_validate_errors():
    store = build_store()
    bad_mapping = PlotMapping(plot_type="curve", payload_id="demo", x_path="data.time", y_path="data.missing")
    try:
        validate_and_resolve(bad_mapping, store)
    except MappingValidationError as exc:
        assert "Y path error" in str(exc)
    else:
        assert False, "Expected failure"

    bad_mapping2 = PlotMapping(plot_type="curve", payload_id="demo", x_path="data.channels.B", y_path="data.time")
    try:
        validate_and_resolve(bad_mapping2, store)
    except MappingValidationError as exc:
        assert "lengths differ" in str(exc)
    else:
        assert False, "Expected failure"
