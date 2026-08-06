#!/usr/bin/env python3
"""Dump the byte layout of a Qwen3.5/3.6 GGUF, per layer and per layer class.

This answers the structural questions in PLAN-qwen36-27b.md section 6 that I could
not answer from source alone:

  - is the trunk really 48 gated-delta-net + 16 full-attention layers, or does this
    checkpoint use a different full_attention_interval?
  - what is the actual byte cost of one layer (it differs by class, so "file size /
    n_layer" is wrong)
  - how many trailing layers fit in a given VRAM budget (this is the quantum the
    fitter is stuck with for dense models)
  - are the MTP layers inside this file, or in a sidecar?

Usage:
    python3 02-gguf-layout.py /path/to/model.gguf [--vram-mib 5000] [--json out.json]

No third-party dependencies. It parses the GGUF header itself (only the metadata and
tensor-info blocks - the tensor data is never touched) and pulls the block-size table
from the repo's own gguf-py/gguf/constants.py, which is numpy-free, so the sizes
cannot drift from the rest of the tree.
"""

import argparse
import importlib.util
import json
import os
import re
import struct
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def load_repo_constants():
    """Load gguf-py/gguf/constants.py directly, bypassing the package __init__.

    The package __init__ imports .lazy, which imports numpy. constants.py itself has
    no third-party imports, so loading it by path keeps this script dependency-free.
    """
    path = os.path.join(REPO_ROOT, "gguf-py", "gguf", "constants.py")
    if not os.path.isfile(path):
        sys.exit(f"error: cannot find {path} - run this from inside the repo checkout")
    spec = importlib.util.spec_from_file_location("_gguf_constants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load_repo_constants()
QUANT_SIZES = C.GGML_QUANT_SIZES
QUANT_NAMES = {int(t): t.name for t in C.GGMLQuantizationType}
VT = C.GGUFValueType

# GGUF scalar value type -> struct format character.
SCALAR_FMT = {
    int(VT.UINT8): "B", int(VT.INT8): "b",
    int(VT.UINT16): "H", int(VT.INT16): "h",
    int(VT.UINT32): "I", int(VT.INT32): "i",
    int(VT.FLOAT32): "f", int(VT.BOOL): "?",
    int(VT.UINT64): "Q", int(VT.INT64): "q",
    int(VT.FLOAT64): "d",
}


class GGUFHeader:
    """Minimal reader for the GGUF metadata and tensor-info blocks."""

    def __init__(self, path):
        self.path = path
        self.fields = {}
        self.tensors = []   # list of (name, dims, ggml_type, n_bytes)

        with open(path, "rb") as f:
            self.buf = f.read(64)
            magic = struct.unpack("<I", self.buf[:4])[0]
            if magic == C.GGUF_MAGIC:
                self.bo = "<"
            elif struct.unpack(">I", self.buf[:4])[0] == C.GGUF_MAGIC:
                self.bo = ">"
            else:
                sys.exit(f"error: {path} is not a GGUF file (bad magic)")
            f.seek(0)
            self._f = f
            self._read_all()

    # -- primitives -----------------------------------------------------
    def _r(self, fmt, size):
        data = self._f.read(size)
        if len(data) != size:
            sys.exit("error: unexpected end of file while parsing the GGUF header")
        return struct.unpack(self.bo + fmt, data)[0]

    def _u32(self):
        return self._r("I", 4)

    def _u64(self):
        return self._r("Q", 8)

    def _str(self):
        n = self._u64()
        return self._f.read(n).decode("utf-8", errors="replace")

    def _value(self, vtype):
        if vtype == int(VT.STRING):
            return self._str()
        if vtype == int(VT.ARRAY):
            elem = self._u32()
            count = self._u64()
            return [self._value(elem) for _ in range(count)]
        fmt = SCALAR_FMT.get(vtype)
        if fmt is None:
            sys.exit(f"error: unknown GGUF value type {vtype}")
        return self._r(fmt, struct.calcsize(fmt))

    # -- structure ------------------------------------------------------
    def _read_all(self):
        self._u32()                     # magic, already validated
        self.version = self._u32()
        if self.version < 2:
            sys.exit(f"error: GGUF v{self.version} uses 32-bit counts; only v2+ supported")
        n_tensors = self._u64()
        n_kv = self._u64()

        for _ in range(n_kv):
            key = self._str()
            self.fields[key] = self._value(self._u32())

        for _ in range(n_tensors):
            name = self._str()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            ggml_type = self._u32()
            self._u64()                 # data offset, unused here
            self.tensors.append((name, dims, ggml_type, self._nbytes(dims, ggml_type)))

    @staticmethod
    def _nbytes(dims, ggml_type):
        n_elems = 1
        for d in dims:
            n_elems *= d
        try:
            block_size, type_size = QUANT_SIZES[C.GGMLQuantizationType(ggml_type)]
        except (KeyError, ValueError):
            return 0
        return n_elems * type_size // block_size


# Metadata keys we care about, without the "<arch>." prefix (src/llama-arch.cpp).
HPARAM_KEYS = [
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.recurrent_layers",
    "full_attention_interval",
    "nextn_predict_layers",
    "ssm.conv_kernel",
    "ssm.inner_size",
    "ssm.state_size",
    "ssm.time_step_rank",
    "ssm.group_count",
    "rope.freq_base",
]

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def classify(tensor_names):
    """Classify a block by which tensors it owns (see src/models/qwen35.cpp)."""
    if any(n.startswith("nextn.") for n in tensor_names):
        return "MTP"
    if any(n.startswith("ssm_") for n in tensor_names):
        return "GDN"
    if any(n in ("attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight")
           for n in tensor_names):
        return "ATTN"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--vram-mib", type=int, default=5000,
                    help="VRAM budget to solve the trailing-layer question against")
    ap.add_argument("--json", metavar="PATH", help="also write the raw layout as JSON")
    args = ap.parse_args()

    g = GGUFHeader(args.model)
    arch = g.fields.get("general.architecture")

    print("=" * 72)
    print(f"file : {args.model}")
    print(f"size : {os.path.getsize(args.model) / GIB:.2f} GiB")
    print(f"arch : {arch}   (gguf v{g.version}, {len(g.tensors)} tensors)")
    print("=" * 72)
    print()

    # ---- metadata -------------------------------------------------------
    print("--- hparams ---")
    hp = {}
    for key in HPARAM_KEYS:
        full = f"{arch}.{key}" if arch else key
        if full not in g.fields:
            continue
        val = g.fields[full]
        hp[key] = val
        if isinstance(val, list) and len(val) > 12:
            shown = f"[{len(val)} values] {val[:12]} ..."
        else:
            shown = val
        print(f"  {key:34s} = {shown}")
    missing = [k for k in HPARAM_KEYS if k not in hp]
    if missing:
        print(f"  (absent from file: {', '.join(missing)})")
    print()

    # ---- quant mix ------------------------------------------------------
    # UD-* quants are mixed precision, and repack eligibility is per type
    # (ggml/src/ggml-cpu/repack.cpp:4573-4699), so the spread matters.
    by_type = defaultdict(lambda: {"n": 0, "bytes": 0})
    for _, _, t, nb in g.tensors:
        e = by_type[QUANT_NAMES.get(t, f"type{t}")]
        e["n"] += 1
        e["bytes"] += nb
    print("--- quant mix ---")
    repackable = {"Q4_0", "Q4_K", "Q2_K", "Q5_K", "Q6_K", "IQ4_NL", "MXFP4", "Q8_0"}
    for tname, e in sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"]):
        flag = "repackable" if tname in repackable else ""
        print(f"  {tname:10s} n={e['n']:5d}  {e['bytes'] / GIB:6.2f} GiB  {flag}")
    rp = sum(e["bytes"] for t, e in by_type.items() if t in repackable)
    tot = sum(e["bytes"] for e in by_type.values())
    print(f"  -> {rp / GIB:.2f} of {tot / GIB:.2f} GiB ({100.0 * rp / max(tot, 1):.0f}%) "
          f"is repack-eligible by type")
    print()

    # ---- tensors grouped by block --------------------------------------
    blocks = defaultdict(dict)
    non_block = {}
    for name, _, _, nb in g.tensors:
        m = BLK_RE.match(name)
        if m:
            blocks[int(m.group(1))][m.group(2)] = nb
        else:
            non_block[name] = nb

    if not blocks:
        print("no blk.* tensors found - is this a sidecar rather than the main checkpoint?")
        for name, nb in sorted(non_block.items(), key=lambda kv: -kv[1]):
            print(f"  {name:44s} {nb / MIB:9.1f} MiB")
        return

    print("--- non-block tensors ---")
    for name, nb in sorted(non_block.items(), key=lambda kv: -kv[1]):
        print(f"  {name:44s} {nb / MIB:9.1f} MiB")
    print()

    # ---- per-layer table -----------------------------------------------
    layer_ids = sorted(blocks)
    rows = []
    for il in layer_ids:
        tensors = blocks[il]
        nb = sum(tensors.values())
        ffn = sum(v for k, v in tensors.items() if k.startswith("ffn_"))
        rows.append({"il": il, "cls": classify(tensors.keys()), "bytes": nb,
                     "ffn_bytes": ffn, "other_bytes": nb - ffn})

    print("--- per-layer bytes ---")
    print(f"  {'il':>4} {'class':>6} {'total MiB':>10} {'ffn MiB':>9} {'attn/ssm MiB':>13}")
    for r in rows:
        print(f"  {r['il']:4d} {r['cls']:>6} {r['bytes'] / MIB:10.1f} "
              f"{r['ffn_bytes'] / MIB:9.1f} {r['other_bytes'] / MIB:13.1f}")
    print()

    # ---- class summary --------------------------------------------------
    print("--- by class ---")
    by_cls = defaultdict(lambda: {"n": 0, "bytes": 0})
    for r in rows:
        by_cls[r["cls"]]["n"] += 1
        by_cls[r["cls"]]["bytes"] += r["bytes"]
    for cls, agg in sorted(by_cls.items()):
        print(f"  {cls:6s} n={agg['n']:3d}  total={agg['bytes'] / GIB:6.2f} GiB  "
              f"mean/layer={agg['bytes'] / agg['n'] / MIB:7.1f} MiB")
    print(f"  {'ALL':6s} n={len(rows):3d}  total={tot / GIB:6.2f} GiB "
          f"(incl. {sum(non_block.values()) / GIB:.2f} GiB non-block)")
    print()

    attn_ids = [r["il"] for r in rows if r["cls"] == "ATTN"]
    gdn_ids = [r["il"] for r in rows if r["cls"] == "GDN"]
    mtp_ids = [r["il"] for r in rows if r["cls"] == "MTP"]
    print(f"  full-attention layers ({len(attn_ids)}): {attn_ids}")
    print(f"  gated-delta-net layers ({len(gdn_ids)}): "
          f"{gdn_ids[:8]}{' ...' if len(gdn_ids) > 8 else ''}")
    print(f"  MTP layers ({len(mtp_ids)}): {mtp_ids}")
    print()

    # ---- trailing-layer budget -----------------------------------------
    # The fitter fills the GPU back-to-front (common/fit.cpp:481), so what matters
    # is the cumulative size of the LAST n layers, not the mean layer size.
    budget = args.vram_mib * MIB
    print(f"--- trailing layers vs a {args.vram_mib} MiB weight budget ---")
    print("  (weights only - KV, recurrent state and compute buffers are on top)")
    tail = list(reversed(layer_ids))
    cum = 0
    cum_at = {}
    for n, il in enumerate(tail, start=1):
        cum += sum(blocks[il].values())
        cum_at[n] = cum
        if cum <= budget * 1.2:
            print(f"  last {n:3d} layers: {cum / MIB:9.1f} MiB"
                  f"{' <= fits' if cum <= budget else ''}")
        else:
            break

    chosen = max((n for n, b in cum_at.items() if b <= budget), default=0)
    used = cum_at.get(chosen, 0)
    print()
    print(f"  => at most {chosen} trailing layers of weights fit in {args.vram_mib} MiB "
          f"({used / MIB:.1f} MiB used, {(budget - used) / MIB:.1f} MiB left over)")
    if chosen < len(tail):
        nxt = tail[chosen]
        one_more = sum(blocks[nxt].values())
        print(f"  => the next layer (il={nxt}) costs {one_more / MIB:.1f} MiB and does not fit;")
        print(f"     that is the granularity the dense fitter is stuck with, so up to")
        print(f"     {(budget - used) / MIB:.1f} MiB of VRAM is stranded at this setting.")
    print()

    # ---- suggested -ot regexes ------------------------------------------
    if attn_ids:
        attn_re = r"blk\.(" + "|".join(str(i) for i in attn_ids) + r")\."
        print("--- suggested overrides ---")
        print("  keep full-attention layers (and their KV) on GPU, everything else on CPU:")
        print(f'    -ot "{attn_re}=CUDA0" -ot ".*=CPU"')
    if gdn_ids:
        gdn_ffn_re = r"blk\.(" + "|".join(str(i) for i in gdn_ids) + r")\.ffn_"
        print("  push only the FFN of the gated-delta-net layers to CPU:")
        print(f'    -ot "{gdn_ffn_re}.*=CPU"')
    print()
    print("  note: -ot to CPU triggers the repack path (PLAN section 1.2). Always")
    print("  re-check RssAnon with 04-residency.sh after changing overrides.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "file": args.model,
                "arch": arch,
                "gguf_version": g.version,
                "hparams": hp,
                "quant_mix": {k: v for k, v in by_type.items()},
                "non_block": non_block,
                "layers": rows,
                "attn_layers": attn_ids,
                "gdn_layers": gdn_ids,
                "mtp_layers": mtp_ids,
                "total_bytes": tot,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
