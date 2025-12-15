from __future__ import annotations

from typing import Any, List


class PathResolutionError(Exception):
    pass


def _parse_path(path: str) -> List[str]:
    if not path or path.strip() == "":
        raise PathResolutionError("Path is empty")

    tokens: List[str] = []
    i = 0
    n = len(path)
    while i < n:
        ch = path[i]
        if ch == ".":
            i += 1
            if i >= n:
                raise PathResolutionError("Path cannot end with '.'")
            continue
        if ch == "[":
            end = path.find("]", i)
            if end == -1:
                raise PathResolutionError("Missing closing bracket in path")
            content = path[i + 1 : end].strip()
            if len(content) < 2 or content[0] not in {'"', "'"} or content[-1] != content[0]:
                raise PathResolutionError("Bracket path must be a quoted string, e.g. ['key']")
            tokens.append(content[1:-1])
            i = end + 1
            if i < n and path[i] not in {".", "["}:
                raise PathResolutionError("Unexpected characters after bracket expression")
            continue

        start = i
        while i < n and path[i] not in {".", "["}:
            i += 1
        token = path[start:i]
        if not token:
            raise PathResolutionError("Empty path segment")
        tokens.append(token)
    return tokens


def resolve_path(payload: Any, path: str) -> Any:
    tokens = _parse_path(path)
    current = payload
    traversed: List[str] = []

    for token in tokens:
        traversed.append(token)
        if isinstance(current, dict):
            if token not in current:
                raise PathResolutionError(
                    f"Missing key '{token}' after traversing: {'/'.join(traversed[:-1])}"
                )
            current = current[token]
        else:
            raise PathResolutionError(
                f"Cannot traverse into '{token}' because object at {'/'.join(traversed[:-1])}"
                f" is of type {type(current).__name__}, expected dict"
            )
    return current


__all__ = ["resolve_path", "PathResolutionError"]
