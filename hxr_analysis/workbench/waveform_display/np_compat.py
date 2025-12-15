from __future__ import annotations

"""Lightweight compatibility layer for numpy.

If real numpy is available, it is used. Otherwise a minimal fallback is
provided to keep unit tests running in constrained environments. The fallback
supports only a small subset of numpy APIs used by this workbench (asarray,
linspace, sin, cos, load for simple .npz files).
"""

from pathlib import Path
import ast
import math
import struct
import zipfile

try:  # pragma: no cover - prefer real numpy when available
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback for test-only environments

    class SimpleArray(list):
        @property
        def shape(self):
            return (len(self),)

    def asarray(seq):
        if isinstance(seq, SimpleArray):
            return seq
        return SimpleArray(seq)

    def linspace(start, stop, num):
        if num <= 1:
            return asarray([start])
        step = (stop - start) / (num - 1)
        return asarray([start + i * step for i in range(num)])

    def sin(seq):
        return asarray([math.sin(x) for x in seq])

    def cos(seq):
        return asarray([math.cos(x) for x in seq])

    def _load_npy_bytes(buf: bytes):
        magic_len = 6
        hlen = struct.unpack('<H', buf[magic_len + 2 : magic_len + 4])[0]
        header = buf[magic_len + 4 : magic_len + 4 + hlen].decode('latin1').strip()
        header_dict = ast.literal_eval(header)
        shape = header_dict.get('shape', (len(buf),))
        count = 1
        for dim in shape:
            count *= dim
        data_bytes = buf[magic_len + 4 + hlen :]
        data = struct.unpack('<%dd' % count, data_bytes)
        return asarray(data)

    def load(path, allow_pickle=False):  # noqa: ARG001
        path = Path(path)
        if path.suffix == '.npz':
            out = {}
            with zipfile.ZipFile(path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.npy'):
                        out[Path(name).stem] = _load_npy_bytes(zf.read(name))
            # mimic numpy.load output for npz
            class NPZ(dict):
                @property
                def files(self):
                    return list(self.keys())
            return NPZ(out)
        raise ImportError("Fallback numpy only supports .npz loading")

    class _Fallback:
        asarray = staticmethod(asarray)
        linspace = staticmethod(linspace)
        sin = staticmethod(sin)
        cos = staticmethod(cos)
        load = staticmethod(load)

    np = _Fallback()  # type: ignore
else:  # pragma: no cover
    np = np

__all__ = ["np"]
