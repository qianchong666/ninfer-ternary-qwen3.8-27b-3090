"""Dump / check the ternary GEMM + rotation integration against an independent reference.

Two modes:

  dump  <artifact> <object> <dump-dir>
        Writes the object's row-split payload, the sign block for its input width, and a
        deterministic activation, so a standalone nvcc harness can run the REAL kernels.

  check <artifact> <object> <dump-dir>
        Decodes the payload in pure numpy (the arrangement was already proven by
        check_payload_order.py), rebuilds  y = W' * (H * (s * (P * x)))  explicitly in float64 and
        compares against the kernel dump.

The point is to cover what no byte/size/oracle check so far covers: that the plane pointers, the
2-bit decode, the sign block for THIS width, and the rotation compose into the documented math.

Usage: gemm_oracle.py dump|check <artifact.ninfer> <object-name> <dump-dir>
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, r"<NINFER>\ninfer\src\ninfer-4090w-ternary")
sys.path.insert(0, r"E:\ninfer\src\ninfer-4090w-ternary")
from tools.artifact import container  # noqa: E402

BLOCK = 1024
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount32(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.uint32)
    return (
        POPCOUNT[v & 0xFF].astype(np.uint16)
        + POPCOUNT[(v >> 8) & 0xFF].astype(np.uint16)
        + POPCOUNT[(v >> 16) & 0xFF].astype(np.uint16)
        + POPCOUNT[(v >> 24) & 0xFF].astype(np.uint16)
    )


def hadamard(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.uint32)
    parity = popcount32(idx[:, None] & idx[None, :]) & 1
    return np.where(parity == 1, -1.0, 1.0) / np.sqrt(float(n))


def make_x(k: int, tokens: int, seed: int = 12345) -> np.ndarray:
    state = np.uint32(seed)
    out = np.empty(k * tokens, dtype=np.float32)
    for i in range(out.size):
        state = np.uint32(state * np.uint32(1664525) + np.uint32(1013904223))
        out[i] = np.float32((int(state >> 8) & 0xFFFF) / 32768.0 - 1.0)
    return out.reshape(k, tokens)


def load(artifact_path: str, name: str):
    with container.Artifact.open(artifact_path) as art:
        obj = art.find(name)
        payload = np.frombuffer(bytes(art.payload(obj)), dtype=np.uint8).copy()
        signs_all = np.frombuffer(bytes(art.payload("text/hadamard_signs")), dtype=np.float32)
        widths = np.frombuffer(bytes(art.payload("text/hadamard_widths")), dtype=np.int32)
        rows, cols = int(obj.shape[0]), int(obj.shape[1])
    offsets, acc = {}, 0
    for w in widths:
        offsets[int(w)] = acc
        acc += int(w)
    signs = signs_all[offsets[cols]:offsets[cols] + cols].copy()
    return payload, signs, rows, cols, obj.format


def decode_pq2(payload: np.ndarray, rows: int, groups: int) -> np.ndarray:
    """Same arrangement the row-split layout declares, decoded exactly like ggml's PQ2_0."""
    codes = payload[: rows * groups * 32].reshape(rows, groups, 32)
    scales = payload[rows * groups * 32: rows * groups * 32 + rows * groups * 2].view(
        np.float16).astype(np.float64).reshape(rows, groups)
    codes = codes.reshape(rows, groups, 32, 1)
    codes = (codes >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4)) & 3  # [rows,groups,32,4]
    codes = codes.reshape(rows, groups, 128).astype(np.float64)
    weights = (codes - 1.0) * scales[:, :, None]
    return weights.reshape(rows, groups * 128)


def main() -> int:
    mode, artifact_path, name, dump_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    payload, signs, rows, cols, fmt = load(artifact_path, name)
    groups = cols // 128
    if fmt != "PQ2_0_G128":
        print(f"this probe only implements PQ2_0; got {fmt}")
        return 2
    x = make_x(cols, 1)
    os.makedirs(dump_dir, exist_ok=True)

    if mode == "dump":
        payload.tofile(os.path.join(dump_dir, "gemm.payload.bin"))
        signs.astype(np.float32).tofile(os.path.join(dump_dir, "gemm.signs.f32"))
        x.astype(np.float32).tofile(os.path.join(dump_dir, "gemm.x.f32"))
        print(f"dumped {name}: payload={payload.size} B rows={rows} cols={cols} groups={groups} "
              f"scale_plane_off={rows * groups * 32}")
        return 0

    weights = decode_pq2(payload, rows, groups)
    print(f"decoded W' {weights.shape}  finite={np.isfinite(weights).all()} "
          f"absmax={np.abs(weights).max():.6g}")

    H = hadamard(BLOCK)
    xr = np.empty_like(x, dtype=np.float64)
    for b in range(cols // BLOCK):
        lo = b * BLOCK
        xr[lo:lo + BLOCK, 0] = H @ (signs[lo:lo + BLOCK].astype(np.float64) * x[lo:lo + BLOCK, 0])
    y_ref = weights @ xr[:, 0]

    y_got = np.fromfile(os.path.join(dump_dir, "gemm.y.f32"), dtype=np.float32).astype(np.float64)
    if y_got.size != rows:
        print(f"kernel dump has {y_got.size} rows, expected {rows}")
        return 1

    rel = float(np.linalg.norm(y_got - y_ref) / np.linalg.norm(y_ref))
    diff = float(np.abs(y_got - y_ref).max())
    # negative controls: if the kernel disagreed with the documented math in any of these ways the
    # match would collapse, so report them to prove the comparison has teeth
    y_norot = weights @ x[:, 0]
    y_nosign = np.empty_like(x, dtype=np.float64)
    for b in range(cols // BLOCK):
        lo = b * BLOCK
        y_nosign[lo:lo + BLOCK, 0] = H @ x[lo:lo + BLOCK, 0]
    y_nosign = weights @ y_nosign[:, 0]
    y_unnorm = weights @ (xr[:, 0] * np.sqrt(BLOCK))
    print(f"  rel_l2 = {rel:.4e}   max|diff| = {diff:.4e}   {'PASS' if rel <= 0.02 else 'FAIL'}")
    for label, ref in (("no rotation", y_norot), ("no signs", y_nosign),
                       ("unnormalized H", y_unnorm)):
        r = float(np.linalg.norm(y_got - ref) / np.linalg.norm(ref))
        print(f"    control {label:16s} rel_l2={r:.4f}  "
              f"{'(differs OK)' if r > 0.05 else '(TOO CLOSE)'}")
    return 0 if rel <= 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
