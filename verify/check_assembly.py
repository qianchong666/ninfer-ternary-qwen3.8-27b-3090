"""Assembly audit: is EVERY ternary object a faithful, correctly ordered copy of its source?

Byte accounting cannot see a row permutation (sizes, row counts and totals are conserved), and a
set-membership test cannot see it either (every duplicated row still finds a source). This audit
therefore compares MULTISETS and, more importantly, checks artifact row a DIRECTLY against the
source row the mapping rule names for a.

Sources are de-interleaved GGUF PQ2_0 blocks; artifacts are row-split planes. Both are reduced to
the same per-row fingerprint (codes bytes + scale bytes), so the comparison is exact and
independent of the rotated basis (rows are permuted, never transformed).

Usage: check_assembly.py <artifact.ninfer> <pq2.gguf>
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
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
V_HEAD_DIM = 128


def artifact_rows(art, name: str) -> list[bytes]:
    obj = art.find(name)
    payload = np.frombuffer(bytes(art.payload(obj)), dtype=np.uint8)
    rows = int(obj.shape[0])
    groups = int(obj.shape[1]) // 128
    codes = payload[: rows * groups * CODE_BYTES]
    off = rows * groups * CODE_BYTES
    scales = payload[off: off + rows * groups * SCALE_BYTES]
    return [
        hashlib.sha1(codes[r * groups * CODE_BYTES:(r + 1) * groups * CODE_BYTES].tobytes()
                     + scales[r * groups * SCALE_BYTES:(r + 1) * groups * SCALE_BYTES].tobytes()
                     ).digest()
        for r in range(rows)
    ]


def gguf_rows(gguf: Gguf, name: str) -> list[bytes]:
    ne, tt, _ = gguf.tensors[name]
    if tt != 142:
        raise SystemExit(f"{name}: expected PQ2_0 (142), got {tt}")
    rows = int(ne[1])
    groups = int(ne[0]) // 128          # ggml ne[0] is the contiguous K extent
    raw = gguf.raw(name, rows)
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, groups, 34)
    codes = np.ascontiguousarray(block[:, :, 2:34]).reshape(rows, groups * CODE_BYTES)
    scales = np.ascontiguousarray(block[:, :, 0:2]).reshape(rows, groups * SCALE_BYTES)
    return [hashlib.sha1(codes[r].tobytes() + scales[r].tobytes()).digest() for r in range(rows)]


def perm48(i: int) -> int:
    return (i % 3) * 16 + (i // 3)


def head_perm(rows: int) -> list[int]:
    """48 heads x 128 rows: the permutation must be applied at head granularity."""
    out = []
    for r in range(rows):
        head, inner = divmod(r, V_HEAD_DIM)
        out.append(perm48(head) * V_HEAD_DIM + inner)
    return out


def audit(label: str, art_keys: list[bytes], src_keys: list[bytes]) -> bool:
    art_counts, src_counts = Counter(art_keys), Counter(src_keys)
    repeated = sum(v - 1 for v in art_counts.values() if v > 1)
    missing = sum(1 for k in src_counts if k not in art_counts)
    order_bad = [a for a in range(len(art_keys)) if art_keys[a] != src_keys[a]]
    ok = (repeated == 0) and (missing == 0) and not order_bad
    print(f"{'OK  ' if ok else 'FAIL'} {label}")
    print(f"       rows art={len(art_keys)} src={len(src_keys)} | "
          f"distinct art={len(art_counts)} src={len(src_counts)} | "
          f"duplicated={repeated} missing={missing} | "
          f"order mismatches={len(order_bad)}/{len(art_keys)}")
    if order_bad:
        print(f"       first bad rows: {order_bad[:8]}")
    return ok


def main() -> int:
    art_path, gguf_path = sys.argv[1], sys.argv[2]
    gguf = Gguf(Path(gguf_path))
    results = []
    with container.Artifact.open(art_path) as art:
        def A(name):
            return artifact_rows(art, name)
        def G(name):
            return gguf_rows(gguf, name)

        # --- globals ---------------------------------------------------------
        results.append(audit("text/token_embedding  <- token_embd.weight",
                             A("text/token_embedding"), G("token_embd.weight")))
        results.append(audit("text/output_head      <- output.weight",
                             A("text/output_head"), G("output.weight")))

        for layer, kind in ((0, "gdn"), (3, "attention")):
            pre = f"blk.{layer}."
            la = f"text/layers/{layer}/"
            print(f"--- layer {layer} ({kind}) ---")
            results.append(audit(f"{la}mlp/down      <- ffn_down",
                                 A(la + "mlp/down"), G(pre + "ffn_down.weight")))
            results.append(audit(f"{la}mlp/gate_up   <- concat(ffn_gate, ffn_up)",
                                 A(la + "mlp/gate_up"),
                                 G(pre + "ffn_gate.weight") + G(pre + "ffn_up.weight")))
            if kind == "gdn":
                results.append(audit(f"{la}gdn/query_key <- attn_qkv[0:4096]",
                                     A(la + "gdn/query_key"),
                                     G(pre + "attn_qkv.weight")[:4096]))
                results.append(audit(f"{la}gdn/output    <- ssm_out",
                                     A(la + "gdn/output"), G(pre + "ssm_out.weight")))
                # value/z halves: compare against the source reordered by the head permutation
                vz = A(la + "gdn/value_z")
                v_src = G(pre + "attn_qkv.weight")[4096:10240]
                z_src = G(pre + "attn_gate.weight")[:6144]
                perm = head_perm(6144)
                results.append(audit(f"{la}gdn/value_z   <- attn_qkv[4096:10240] (head perm)",
                                     vz[:6144], [v_src[p] for p in perm]))
                results.append(audit(f"{la}gdn/value_z   <- attn_gate[0:6144] (head perm)",
                                     vz[6144:], [z_src[p] for p in perm]))
            else:
                aq = G(pre + "attn_q.weight")
                even = [c * 256 + r for c in range(0, 48, 2) for r in range(256)]
                odd = [c * 256 + r for c in range(1, 48, 2) for r in range(256)]
                qk = A(la + "attention/query_key")
                gv = A(la + "attention/gate_value")
                results.append(audit(f"{la}attention/query_key <- attn_q even chunks",
                                     qk[:6144], [aq[r] for r in even]))
                results.append(audit(f"{la}attention/query_key <- attn_k",
                                     qk[6144:], G(pre + "attn_k.weight")[:1024]))
                results.append(audit(f"{la}attention/gate_value <- attn_q odd chunks",
                                     gv[:6144], [aq[r] for r in odd]))
                results.append(audit(f"{la}attention/gate_value <- attn_v",
                                     gv[6144:], G(pre + "attn_v.weight")[:1024]))
                results.append(audit(f"{la}attention/output <- attn_output",
                                     A(la + "attention/output"), G(pre + "attn_output.weight")))

    passed = sum(1 for r in results if r)
    print(f"\nRESULT: {passed}/{len(results)} assembly rules OK")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
