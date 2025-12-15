from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hxr_analysis.template.detect_candidate_peaks import (
    DetectCandidatePeaksParams,
    detect_candidate_peaks,
)
from hxr_analysis.workbench.waveform_display.importers import ImportErrorWithContext, import_files
from hxr_analysis.workbench.waveform_display.models import Payload, PayloadStore
from hxr_analysis.workbench.waveform_display.np_compat import np
from hxr_analysis.workbench.waveform_display.path import PathResolutionError, resolve_path


class DetectCandidatePeaksController:
    def __init__(self, store: PayloadStore | None = None):
        self.store = store or PayloadStore()
        self.latest_outputs: dict[str, Payload] = {}

    # ---------- Payload handling ----------
    def import_payloads(self, paths: Iterable[str | Path]):
        imported = import_files(paths)
        for pid, payload in imported.items():
            self.store.add(pid, payload)
        return imported

    def payload_ids(self):
        return list(self.store.ids())

    def get_payload(self, payload_id: str) -> Payload:
        return self.store.get(payload_id)

    def available_channels(self, payload_id: str) -> list[str]:
        payload = self.get_payload(payload_id)
        try:
            channels = resolve_path(payload, "data.channels")
        except PathResolutionError:
            return []
        if not isinstance(channels, dict):
            return []
        return list(channels.keys())

    # ---------- Detection ----------
    def run_detection(self, payload_id: str, params: DetectCandidatePeaksParams) -> Payload:
        payload = self.get_payload(payload_id)
        output = detect_candidate_peaks(payload, params)
        self.latest_outputs[payload_id] = output
        return output

    def get_output(self, payload_id: str) -> Payload | None:
        return self.latest_outputs.get(payload_id)

    # ---------- Persistence ----------
    def save_output(self, payload_id: str, target_dir: str | Path) -> Path:
        if payload_id not in self.latest_outputs:
            raise ValueError("No detection results available to save")
        payload = self.latest_outputs[payload_id]
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{payload_id}.bundle.npz"
        np.savez(path, payload=payload)
        return path


__all__ = ["DetectCandidatePeaksController", "ImportErrorWithContext"]
