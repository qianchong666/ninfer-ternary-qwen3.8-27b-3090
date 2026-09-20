"""Verify the two INFERRED row-order rules by exact row fingerprints.

MAPPING.json flags exactly two rules as inferred rather than measured, and both are row
permutations of whole quantized rows -- which means each row's bytes survive untouched and can be
matched exactly between the GGUF and the artifact:

  gdn_value_z                    "risk: if the engine test disagrees, flip this single permutation"
  attn_q_per_head_interleave     "INFERRED for main layers from the MTP layer"

Nothing about the rotated basis matters here: rows are permuted, not transformed, so a byte-level
row fingerprint is an exact test of the rule (this is the "structural self-proof" the handoff asks
for instead of value correlation, which is meaningless across bases).

Usage: check_row_order.py <artifact.ninfer> <pq2.gguf>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"H:/ninfer-ternary/src/ninfer")
sys.path.insert(0, r"H:/ninfer-ternary/src/ninfer")
sys.path.insert(0, r"H:/ninfer-ternary/verify")
from _ternary_ref import Gguf  # noqa: E402
from tools.artifact import container  # noqa: E402

GROUPS = 40
CODE_BYTES = 32
SCALE_BYTES = 2


def artifact_row_keys(art, name: str) -> list[bytes]:
    """Fingerprint every row of a PQ2_0 row-split object as (codes, scales) digests."""
    obj = art.find(name)
    payload = np.frombuffer(bytes(art.payload(obj)), dtype=np.uint8)
    rows = int(obj.shape[0])
    assert int(obj.shape[1]) == GROUPS * 128, obj.shape
    codes = payload[: rows * GROUPS * CODE_BYTES]
    scale_start = rows * GROUPS * CODE_BYTES
    scales = payload[scale_start: scale_start + rows * GROUPS * SCALE_BYTES]
    keys: list[bytes] = []
    for r in range(rows):
        c = codes[r * GROUPS * CODE_BYTES:(r + 1) * GROUPS * CODE_BYTES]
        s = scales[r * GROUPS * SCALE_BYTES:(r + 1) * GROUPS * SCALE_BYTES]
        keys.append(hashlib.sha1(bytes(c) + bytes(s)).digest())
    return keys


def gguf_row_keys(gguf: Gguf, name: str) -> list[bytes]:
    """De-interleave GGUF {fp16 d; uint8 qs[32]} blocks into the same (codes, scales) fingerprint."""
    ne, tt, _ = gguf.tensors[name]
    assert tt == 142, (name, tt)  # PQ2_0
    rows = int(ne[1])
    raw = gguf.raw(name, rows)
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, GROUPS, 34)
    codes = np.ascontiguousarray(block[:, :, 2:34]).reshape(rows, GROUPS * CODE_BYTES)
    scales = np.ascontiguousarray(block[:, :, 0:2]).reshape(rows, GROUPS * SCALE_BYTES)
    return [hashlib.sha1(codes[r].tobytes() + scales[r].tobytes()).digest() for r in range(rows)]


def report(label: str, art_keys: list[bytes], gguf_subset: list[bytes],
           src_index: list[int], expected) -> bool:
    """Check the artifact rows two ways.

    The multiset test proves the packer took the right source rows. The permutation test is done by
    comparing the artifact row at index a directly against the source row the rule NAMES for a --
    never by searching for a matching fingerprint first, because identical quantized rows do occur
    and a first-hit search would then report a harmless false mismatch.
    """
    table: dict[bytes, list[int]] = {}
    for r, key in enumerate(gguf_subset):
        table.setdefault(key, []).append(r)
    matched = sum(1 for key in art_keys if key in table)
    ok_set = matched == len(art_keys)
    unique_art = len(set(art_keys))
    unique_src = len(set(gguf_subset))
    print(f"{label}: artifact rows={len(art_keys)} source rows={len(gguf_subset)} "
          f"matched={matched} -> {'row SET matches' if ok_set else 'ROW SET MISMATCH'}")
    print(f"    distinct fingerprints: artifact={unique_art} source={unique_src} "
          f"(duplicates make fingerprint-first search unreliable, hence the direct check)")

    direct = [a for a, src in enumerate(src_index)
              if art_keys[a] != gguf_subset[src]]
    perm_ok = not direct
    print(f"    documented rule holds row-by-row: {perm_ok}"
          + (f"  ({len(direct)} of {len(art_keys)} rows disagree)" if not perm_ok else ""))

    # Is the artifact a genuine PERMUTATION of the source rows?  A repack that reshaped the wrong
    # axis repeats some rows and drops others, which a "does every row exist somewhere" test cannot
    # see: every row still matches something.  Compare the multisets explicitly.
    from collections import Counter
    art_counts = Counter(art_keys)
    src_counts = Counter(gguf_subset)
    repeated = {k: v for k, v in art_counts.items() if v > 1}
    missing = [k for k in src_counts if k not in art_counts]
    extra_rows = sum(v - 1 for v in repeated.values())
    print(f"    multiset: artifact {len(art_keys)} rows / {len(art_counts)} distinct; "
          f"source {len(gguf_subset)} rows / {len(src_counts)} distinct")
    print(f"    rows DUPLICATED in the artifact: {len(repeated)} fingerprints covering "
          f"{extra_rows} redundant rows;  source rows MISSING from the artifact: {len(missing)}")
    if missing or repeated:
        print("    -> NOT a permutation: the packing permutation duplicates and drops rows")
    if not perm_ok:
        print("    first divergent rows (artifact row -> named source vs what it actually equals):")
        shown = 0
        for a in direct[:6]:
            hits = table.get(art_keys[a], [])
            print(f"      {a} -> named {src_index[a]}, actually equals source rows {hits[:4]}")
            shown += 1
        print(f"    artifact first 24 -> named sources: {src_index[:24]}")
    return ok_set and perm_ok


def tiled_to_grouped_index(grouped_row: int, heads: int = 48, per: int = 3) -> int:
    """grouped row (nk, rep, hd) -> tiled source row (rep, nk, hd), with 128 values per head."""
    hd = grouped_row % 128
    rest = grouped_row // 128
    nk, rep = rest // per, rest % per
    return rep * (heads // per) * 128 + nk * 128 + hd


def main() -> int:
    art_path, gguf_path = sys.argv[1], sys.argv[2]
    gguf = Gguf(Path(gguf_path))
    ok = True
    with container.Artifact.open(art_path) as art:
        # ---- rule: gdn_value_z (48 layers) ----------------------------------
        art_keys = artifact_row_keys(art, "text/layers/0/gdn/value_z")
        value_src = gguf_row_keys(gguf, "blk.0.attn_qkv.weight")[4096:10240]   # 6144 V rows
        gate_src = gguf_row_keys(gguf, "blk.0.attn_gate.weight")[0:6144]        # 6144 z rows
        print("== gdn_value_z: artifact rows [0:6144] vs gguf attn_qkv[4096:10240]")
        ok &= report("   value part", art_keys[0:6144], value_src,
                     [tiled_to_grouped_index(r) for r in range(6144)], None)
        print("== gdn_value_z: artifact rows [6144:12288] vs gguf attn_gate[0:6144]")
        ok &= report("   z part", art_keys[6144:12288], gate_src,
                     [tiled_to_grouped_index(r) for r in range(6144)], None)

        # ---- rule: attn_q_per_head_interleave (16 layers) -------------------
        qk_keys = artifact_row_keys(art, "text/layers/3/attention/query_key")
        attn_q = gguf_row_keys(gguf, "blk.3.attn_q.weight")   # 12288 rows: 48 chunks of 256
        even = [r for chunk in range(0, 48, 2) for r in range(chunk * 256, chunk * 256 + 256)]
        print("== attn_q_per_head_interleave: artifact query rows [0:6144] vs even chunks of attn_q")
        ok &= report("   query part", qk_keys[0:6144], [attn_q[r] for r in even],
                     list(range(6144)), None)
        attn_k = gguf_row_keys(gguf, "blk.3.attn_k.weight")
        print("== attn key part: artifact rows [6144:7168] vs gguf attn_k[0:1024]")
        ok &= report("   key part", qk_keys[6144:7168], attn_k[0:1024], list(range(1024)), None)

        gv_keys = artifact_row_keys(art, "text/layers/3/attention/gate_value")
        odd = [r for chunk in range(1, 48, 2) for r in range(chunk * 256, chunk * 256 + 256)]
        print("== attn_q_per_head_interleave: artifact gate rows [0:6144] vs odd chunks of attn_q")
        ok &= report("   gate part", gv_keys[0:6144], [attn_q[r] for r in odd],
                     list(range(6144)), None)
    print(f"\nRESULT: {'all row-order rules hold' if ok else 'ROW-ORDER RULE MISMATCH FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
