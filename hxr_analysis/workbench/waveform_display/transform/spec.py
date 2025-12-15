from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List


@dataclass
class IdSpec:
    id_paths: List[str]
    fallback: str = "payload"
    joiner: str = "__"
    sanitize: bool = True


@dataclass
class SplitSpec:
    source_path: str
    split_mode: Literal["dict_keys", "list_items"]
    child_payload_target_path: str | None = None
    id_spec: IdSpec | None = None
    child_key_template: str = "{key}"
    attach_id_to_meta_path: str = "meta.__uid__"
    copy_policy: Literal["shallow", "deep"] = "shallow"


@dataclass
class MergeSpec:
    target_path: str
    merge_mode: Literal["dict_union", "stack", "concat"]
    id_spec: IdSpec | None = None
    collision_policy: Literal["error", "attach_id", "overwrite", "suffix_counter"] = "attach_id"
    collision_template: str = "{key}@{uid}"
    require_same_timebase: bool = True
    time_path: str = "data.time"
    source_map_path: str = "meta.__sources__"
    collision_suffix_separator: str = "_"
