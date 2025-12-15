import math

from hxr_analysis.template.detect_candidate_peaks import DetectCandidatePeaksParams, detect_candidate_peaks
from hxr_analysis.workbench.waveform_display.np_compat import np


def _make_payload(time, channels):
    return {"type": "waveform_bundle", "data": {"time": time, "channels": channels}, "meta": {}}


def _sin_array(factor, t, scale: float = 1.0):
    return np.asarray([factor * math.sin(scale * float(val)) for val in t])


def _cos_array(factor, t, scale: float = 1.0):
    return np.asarray([factor * math.cos(scale * float(val)) for val in t])


def _gaussian(amp, center, width, t):
    return np.asarray([amp * math.exp(-0.5 * ((tv - center) / width) ** 2) for tv in t])


def test_detect_single_pulse():
    t = np.linspace(0, 1, 500)
    noise = _sin_array(0.01, t, scale=2 * math.pi)
    pulse = _gaussian(5.0, 0.5, 0.01, t)
    y = np.asarray([n + p for n, p in zip(noise, pulse, strict=False)])

    payload = _make_payload(t, {"A": y})
    result = detect_candidate_peaks(payload)

    peaks = result["events"]["candidate_peaks"]["by_channel"]["A"]
    assert len(peaks["i"]) == 1
    assert abs(peaks["t"][0] - 0.5) < 0.01
    assert peaks["amp"][0] > 4.0


def test_dead_time_removes_double_detection():
    t = np.linspace(0, 1, 400)
    pulse1 = _gaussian(3.0, 0.2, 0.003, t)
    pulse2 = _gaussian(6.0, 0.204, 0.003, t)
    base = _cos_array(0.01, t, scale=5)
    y = np.asarray([b + p1 + p2 for b, p1, p2 in zip(base, pulse1, pulse2, strict=False)])

    payload = _make_payload(t, {"A": y})
    params = DetectCandidatePeaksParams(min_distance_samples=20)
    result = detect_candidate_peaks(payload, params)
    peaks = result["events"]["candidate_peaks"]["by_channel"]["A"]

    assert len(peaks["i"]) == 1
    assert abs(peaks["t"][0] - 0.204) < 0.005
    assert peaks["amp"][0] > 5.0


def test_auto_polarity_detects_negative_pulses():
    t = np.linspace(0, 1, 300)
    pulse = _gaussian(-4.0, 0.7, 0.005, t)
    base = _sin_array(0.01, t, scale=3)
    y = np.asarray([b + p for b, p in zip(base, pulse, strict=False)])

    payload = _make_payload(t, {"A": y})
    params = DetectCandidatePeaksParams(polarity="auto")
    result = detect_candidate_peaks(payload, params)

    peaks = result["events"]["candidate_peaks"]["by_channel"]["A"]
    assert len(peaks["i"]) == 1
    assert abs(peaks["t"][0] - 0.7) < 0.01
    assert peaks["amp"][0] > 3.5


def test_noise_mad_threshold_behavior():
    t = np.linspace(0, 1, 200)
    noise = _cos_array(0.01, t, scale=10)
    weak_pulse = _gaussian(0.2, 0.4, 0.01, t)
    y = np.asarray([n + p for n, p in zip(noise, weak_pulse, strict=False)])

    payload = _make_payload(t, {"A": y})

    high_sigma = DetectCandidatePeaksParams(threshold_sigma=20.0)
    result_high = detect_candidate_peaks(payload, high_sigma)
    peaks_high = result_high["events"]["candidate_peaks"]["by_channel"]["A"]
    assert len(peaks_high["i"]) == 0

    low_sigma = DetectCandidatePeaksParams(threshold_sigma=5.0)
    result_low = detect_candidate_peaks(payload, low_sigma)
    peaks_low = result_low["events"]["candidate_peaks"]["by_channel"]["A"]
    assert len(peaks_low["i"]) == 1
    assert abs(peaks_low["t"][0] - 0.4) < 0.01
