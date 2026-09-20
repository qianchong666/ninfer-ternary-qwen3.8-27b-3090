"""Oracle for the ternary embedding path: gather (PQ2_0 decode) + inverse folded-basis mapping.

The residual stream enters layer 0 at the embedding output, so a break here makes every downstream
number meaningless -- which is exactly what a perplexity close to uniform looks like. This compares
the engine's dumped embedding activation against an independent numpy rebuild of

    h = s * (H * z),   z = decode(token_embedding_row[id])

where z comes straight out of the artifact payload (decoded independently of the engine).

Negative controls are mandatory: a near-uniform or all-zeros dump would otherwise "match" nothing,
and the unrotated variant would match if the engine simply skipped the mapping.

Usage: check_embedding.py <artifact.ninfer> <dump-file.T<N>>
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, r"<NINFER>\ninfer\src\ninfer-4090w-ternary")
sys.path.insert(0, r"E:\ninfer\src\ninfer-4090w-ternary")
from tools.artifact import container  # noqa: E402

BLOCK = 1024
GROUPS = 40
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount32(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.uint32)
    return (POP[v & 0xFF].astype(np.uint16) + POP[(v >> 8) & 0xFF].astype(np.uint16)
            + POP[(v >> 16) & 0xFF].astype(np.uint16) + POP[(v >> 24) & 0xFF].astype(np.uint16))


def hadamard(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    parity = popcount32(idx[:, None] & idx[None, :]) & 1
    return np.where(parity == 1, -1.0, 1.0) / np.sqrt(float(n))


def main() -> int:
    art_path, dump_path = sys.argv[1], sys.argv[2]
    raw = np.fromfile(dump_path, dtype=np.uint8)
    tokens, hidden, id_count = np.frombuffer(raw[:12], dtype=np.int32)
    ids = np.frombuffer(raw[12:12 + id_count * 4], dtype=np.int32)
    bf16 = np.frombuffer(raw[12 + id_count * 4:], dtype=np.uint16).reshape(tokens, hidden)
    # The op writes out as [hidden, T] row-major, so element (k, t) sits at k*T + t. Reshaping the
    # flat buffer as (T, hidden) would reinterpret that as t*hidden + k -- a silent transpose that
    # decorrelates everything for T > 1 and looks like a total engine failure.
    got = (bf16.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(hidden, tokens)
    print(f"dump: tokens={tokens} hidden={hidden} ids={id_count}")
    print(f"ids[:12]={ids[:12].tolist()}")

    with container.Artifact.open(art_path) as art:
        obj = art.find("text/token_embedding")
        payload = np.frombuffer(bytes(art.payload(obj)), dtype=np.uint8).copy()
        signs_all = np.frombuffer(bytes(art.payload("text/hadamard_signs")), dtype=np.float32)
        widths = np.frombuffer(bytes(art.payload("text/hadamard_widths")), dtype=np.int32)
        rows = int(obj.shape[0])
    off = 0
    sign_offsets = {}
    for w in widths:
        sign_offsets[int(w)] = off
        off += int(w)
    signs = signs_all[sign_offsets[hidden]:sign_offsets[hidden] + hidden].astype(np.float64)

    codes = payload[: rows * GROUPS * 32].reshape(rows, GROUPS, 32)
    scale_off = rows * GROUPS * 32
    scales = payload[scale_off: scale_off + rows * GROUPS * 2].view(np.float16).astype(
        np.float64).reshape(rows, GROUPS)

    def decode_row(index: int) -> np.ndarray:
        q = codes[index]                                     # (groups, 32 bytes)
        code = (q[:, :, None] >> (2 * np.arange(4, dtype=np.uint8))[None, None, :]) & 3
        flat = code.reshape(-1).astype(np.float64)            # 32 bytes x 4 codes x groups
        return (flat - 1.0) * np.repeat(scales[index], 128)

    H = hadamard(BLOCK)
    n_block = hidden // BLOCK

    def inverse(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        for b in range(n_block):
            lo = b * BLOCK
            out[lo:lo + BLOCK] = signs[lo:lo + BLOCK] * (H @ vec[lo:lo + BLOCK])
        return out

    def forward(vec: np.ndarray) -> np.ndarray:
        out = np.empty_like(vec)
        for b in range(n_block):
            lo = b * BLOCK
            out[lo:lo + BLOCK] = H @ (signs[lo:lo + BLOCK] * vec[lo:lo + BLOCK])
        return out

    checked = 0
    worst = 0.0
    for t in range(min(tokens, 8)):
        z = decode_row(int(ids[t]))
        ref = inverse(z)
        col = got[:, t]
        if t < 3:
            def stats(name, v):
                print(f"    {name}: ||v||={np.linalg.norm(v):.6g} mean={v.mean():+.4g} "
                      f"std={v.std():.4g} min={v.min():+.4g} max={v.max():+.4g} "
                      f"nan={int(np.isnan(v).sum())}")
            print(f"  token {t} id={int(ids[t])}")
            stats("engine dump", col)
            stats("decoded row z", z)
            stats("expected s*(H*z)", ref)
            # is the engine maybe returning a DIFFERENT token's row, or a mis-strided gather?
            for shift in (-2, -1, 1, 2):
                if 0 <= t + shift < tokens:
                    other = inverse(decode_row(int(ids[t + shift])))
                    c = float(np.dot(col, other) /
                              max(np.linalg.norm(col) * np.linalg.norm(other), 1e-30))
                    print(f"      cos vs token{shift:+d} row = {c:+.4f}")
        rel = float(np.linalg.norm(col - ref) / np.linalg.norm(ref))
        cos = float(np.dot(col, ref) / (np.linalg.norm(col) * np.linalg.norm(ref)))
        ctrl_raw = float(np.dot(col, z) / (np.linalg.norm(col) * np.linalg.norm(z)))
        ctrl_fwd = forward(z)
        ctrl_fwd_cos = float(np.dot(col, ctrl_fwd) / (np.linalg.norm(col) * np.linalg.norm(ctrl_fwd)))
        if t < 4:
            print(f"    rel_l2={rel:.4e} cos={cos:+.6f} | "
                  f"control cos(unrotated)={ctrl_raw:+.4f} cos(signs-first)={ctrl_fwd_cos:+.4f}")
        worst = max(worst, rel)
        checked += 1
    print(f"\nchecked {checked} tokens, worst rel_l2 = {worst:.4e}")
    print(f"RESULT: {'PASS' if worst <= 0.02 else 'FAIL'} (engine embedding == independent rebuild)")
    return 0 if worst <= 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
