import pytest

pytest.importorskip("pyqtgraph")

from hxr_analysis.workbench.waveform_display.style import (  # noqa: E402
    parse_style,
    style_to_pen,
)


def test_parse_style_accepts_json():
    style = parse_style('{"color": "#ff0000", "width": 2}')
    assert style["color"] == "#ff0000"
    assert style["width"] == 2


def test_parse_style_accepts_literal_dict():
    style = parse_style("{'color': 'blue', 'style': 'dash'}")
    assert style["color"] == "blue"
    assert style["style"] == "dash"


def test_parse_style_invalid():
    with pytest.raises(ValueError):
        parse_style("not a dict")


def test_style_to_pen_defaults():
    pen = style_to_pen({})
    assert pen.color().name() == "#000000"
