"""Compatibility shim for the author's `_ternary_ref` GGUF reader.

check_row_order.py and check_assembly.py import `Gguf` from `_ternary_ref`, a module that lived in
the author's workspace and is not part of the patch bundle. The packer's reader is the same
seek-based GGUF parser -- with the corrected tensor-info-list `header_end` that keeps every payload
read on the right offset (verified by the scale-distribution fingerprint in `pack.py check`) -- so
re-export it and add the two audit-facing accessors the scripts expect:

  * `tensors`  : dict name -> (ne, type, offset)
  * `raw(name, rows)` : the raw GGUF block bytes for `rows` rows
"""
from __future__ import annotations

import sys

_PACK = r"H:/ninfer-ternary/pack"
if _PACK not in sys.path:
    sys.path.insert(0, _PACK)

from pack import Gguf as _PackerGguf  # noqa: E402


class Gguf(_PackerGguf):
    """pack.py's reader plus the audit-facing accessors."""

    def __init__(self, path):
        super().__init__(str(path))
        self.tensors = self.tensor

    def raw(self, name: str, rows: int) -> bytes:
        n, _gpr, _block, row_bytes, _raw = self.blocks(name)
        if rows > n:
            raise SystemExit(f"{name}: asked for {rows} rows, tensor has {n}")
        return self.payload(name)[: rows * row_bytes]


__all__ = ["Gguf"]
