"""Decide how a ternary payload is actually arranged inside a ninfer artifact.

Two candidate arrangements for a row of `groups` 128-weight groups:

  A) row-split planes (what ninfer's row-split-k128-v1 GEMM assumes)
        [codes: rows*groups*32][high: rows*groups*0][scales: rows*groups*2]
  B) verbatim GGUF PQ2_0 blocks (what a byte-for-byte payload copy would produce)
        per group: {fp16 d; uint8 qs[32]}, i.e. 34 bytes per group, interleaved

The decoded *semantics* of a group are identical either way, so byte-round-trip checks cannot
tell them apart -- only the distribution fingerprint can.  A correctly located fp16 scale plane
is tight and all-positive (median ~0.012-0.016, high byte variety 11..32/256); reading raw code
bytes as fp16 instead gives scattered exponents, negatives and NaNs.

Usage: check_payload_order.py <artifact.ninfer> <object-name-substring>
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"H:/ninfer-ternary/src/ninfer")
from tools.artifact import container  # noqa: E402


def stats(label: str, values: np.ndarray) -> None:
    finite = np.isfinite(values)
    pos = values[finite & (values > 0)]
    if pos.size == 0:
        print(f"  {label}: no finite positives")
        return
    raw = pos.astype(np.float32).view(np.uint8).reshape(-1, 4)
    high = np.unique(raw[:, 1]).size
    print(
        f"  {label}: n={values.size} finite={int(finite.sum())} "
        f"neg={int((values[finite] < 0).sum())} nan={int((~finite).sum())} "
        f"median={np.median(pos):.6g} min={pos.min():.6g} max={pos.max():.6g} "
        f"high_byte_kinds={high}/256"
    )


def main() -> int:
    path, needle = sys.argv[1], sys.argv[2]
    with container.Artifact.open(path) as art:
        names = [o.name for o in art.objects
                 if needle in o.name and getattr(o, "format", "") in ("PQ2_0_G128", "PTQ1_0_G128")]
        if not names:
            print("no matching ternary object")
            return 1
        name = names[0]
        obj = art.find(name)
        payload = np.frombuffer(bytes(art.payload(obj)), dtype=np.uint8)
        rows, cols = int(obj.shape[0]), int(obj.shape[1])
        print(f"object {name}  fmt={obj.format}  shape=({rows}, {cols})  bytes={payload.size}")

    groups = cols // 128
    code_bytes = 32 if obj.format == "PQ2_0_G128" else 24

    # A) plane order
    plane_a_scale_off = rows * groups * code_bytes
    a_count = rows * groups
    a = payload[plane_a_scale_off:plane_a_scale_off + a_count * 2].view(np.float16)

    # B) interleaved GGUF blocks: fp16 d then codes, 34 bytes per group
    block = 34 if obj.format == "PQ2_0_G128" else 28
    if rows * groups * block == payload.size:
        idx = (np.arange(rows * groups) * block)
        b = np.ascontiguousarray(payload[idx[:, None] + np.arange(2)[None, :]]).view(np.float16)
    else:
        b = np.zeros(0, dtype=np.float16)

    print("scale-plane candidates:")
    stats("A row-split planes", a)
    stats("B interleaved blocks", b)

    # codes fingerprint: the 2-bit codes of a correct read are not uniform, and the *zero* code
    # (code == 1) has a known theoretical share of 32.776% for PQ2_0.
    if obj.format == "PQ2_0_G128":
        for label, off in (("A", 0), ("B", 2)):
            if off == 0:
                codes = payload[: rows * groups * 32]
            else:
                idx = np.arange(rows * groups) * block
                codes = np.ascontiguousarray(
                    payload[(idx[:, None] + off + np.arange(32)[None, :]).ravel()]
                )
            q = codes.reshape(-1, 4)
            low = (q[:, 0] & 0b11).astype(np.int32)
            share = float((low == 1).mean())
            print(f"  codes {label}: byte_kinds={np.unique(codes).size} "
                  f"first_code_zero_share={share:.4f} (PQ2_0 theory 0.3278)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
