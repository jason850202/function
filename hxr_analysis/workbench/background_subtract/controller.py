from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from hxr_analysis.preprocessing.background_subtract import (
    BackgroundSubtractParams,
    background_subtract_many,
)
from hxr_analysis.workbench.waveform_display.importers import ImportErrorWithContext, import_files
from hxr_analysis.workbench.waveform_display.models import Payload, PayloadStore
from hxr_analysis.workbench.waveform_display.np_compat import np


class BackgroundSubtractController:
    def __init__(self, store: PayloadStore | None = None):
        self.store = store or PayloadStore()
        self.background_id: str | None = None

    # ---------- Payload handling ----------
    def import_payloads(self, paths: Iterable[str | Path]) -> Dict[str, Payload]:
        imported = import_files(paths)
        for pid, payload in imported.items():
            self.store.add(pid, payload)
        return imported

    def set_background(self, payload_id: str) -> None:
        if payload_id not in self.store.payloads:
            raise KeyError(f"Unknown payload id: {payload_id}")
        self.background_id = payload_id

    def selected_background(self) -> Payload:
        if not self.background_id:
            raise ValueError("No background payload selected")
        return self.store.get(self.background_id)

    def compute(self, experiment_ids: List[str], params: BackgroundSubtractParams) -> Dict[str, Payload]:
        if not experiment_ids:
            return {}
        if not self.background_id:
            raise ValueError("Background payload not set")
        bg_payload = self.store.get(self.background_id)
        exp_payloads = [self.store.get(pid) for pid in experiment_ids]
        outputs = background_subtract_many(exp_payloads, bg_payload, params=params)
        return {pid: out for pid, out in zip(experiment_ids, outputs, strict=False)}

    def save_outputs(self, outputs: Dict[str, Payload], target_dir: str | Path) -> Path:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, str | None] = {}
        for pid, payload in outputs.items():
            manifest[pid] = payload.get("meta", {}).get("__uid__")
            path = target / f"{pid}.bundle.npz"
            np.savez(path, payload=payload)

        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path


__all__ = ["BackgroundSubtractController", "ImportErrorWithContext"]
