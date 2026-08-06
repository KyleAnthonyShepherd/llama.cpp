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
  - how big is the MTP module, and does it live in this file or a sidecar?

Usage:
    python3 02-gguf-layout.py /path/to/model.gguf [--vram-mib 5000]

Requires the repo's gguf-py on the path; the script adds it automatically.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "gguf-py"))

from gguf.gguf_reader import GGUFReader  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

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


def field_value(field):
    """Best-effort scalar/list extraction from a ReaderField."""
    try:
        val = field.contents()
    except Exception:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


def classify(tensor_names):
    """Classify a block by which tensors it owns (see src/models/qwen35.cpp)."""
    if any(n.startswith("nextn.") for n in tensor_names):
        return "MTP"
    if any(n.startswith("ssm_") for n in tensor_names):
        return "GDN"
    if any(n in ("attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight") for n in tensor_names):
        return "ATTN"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--vram-mib", type=int, default=5000,
                    help="VRAM budget to solve the trailing-layer question against")
    ap.add_argument("--json", metavar="PATH", help="also write the raw layout as JSON")
    args = ap.parse_args()

    reader = GGUFReader(args.model, "r")

    # ---- metadata -------------------------------------------------------
    arch = None
    for name, field in reader.fields.items():
        if name == "general.architecture":
            arch = field_value(field)
            break

    print("=" * 72)
    print(f"file : {args.model}")
    print(f"size : {os.path.getsize(args.model) / GIB:.2f} GiB")
    print(f"arch : {arch}")
    print("=" * 72)
    print()
    print("--- hparams ---")
    hp = {}
    for key in HPARAM_KEYS:
        full = f"{arch}.{key}" if arch else key
        field = reader.fields.get(full)
        if field is None:
            continue
        val = field_value(field)
        hp[key] = val
        if isinstance(val, (list, tuple)) and len(val) > 12:
            shown = f"[{len(val)} values] {list(val[:12])} ..."
        else:
            shown = val
        print(f"  {key:34s} = {shown}")

    missing = [k for k in HPARAM_KEYS if k not in hp]
    if missing:
        print(f"  (absent from file: {', '.join(missing)})")
    print()

    # ---- tensors grouped by block --------------------------------------
    blocks = defaultdict(dict)   # il -> {suffix: n_bytes}
    non_block = {}               # name -> n_bytes
    total = 0
    for t in reader.tensors:
        name = t.name
        total += int(t.n_bytes)
        m = BLK_RE.match(name)
        if m:
            blocks[int(m.group(1))][m.group(2)] = int(t.n_bytes)
        else:
            non_block[name] = int(t.n_bytes)

    if not blocks:
        print("no blk.* tensors found - is this the MTP sidecar rather than the main checkpoint?")
        for name, nb in sorted(non_block.items()):
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
        cls = classify(tensors.keys())
        nb = sum(tensors.values())
        ffn = sum(v for k, v in tensors.items() if k.startswith("ffn_"))
        rows.append({"il": il, "cls": cls, "bytes": nb, "ffn_bytes": ffn,
                     "other_bytes": nb - ffn})

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
        mean = agg["bytes"] / agg["n"] / MIB
        print(f"  {cls:6s} n={agg['n']:3d}  total={agg['bytes'] / GIB:6.2f} GiB  mean/layer={mean:7.1f} MiB")
    print(f"  {'ALL':6s} n={len(rows):3d}  total={total / GIB:6.2f} GiB "
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
    cum_at = {}   # n trailing layers -> cumulative bytes
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
    # Handy for hand-beating the fitter (PLAN section 2.3) - these feed the same
    # override array common_fit_params writes into.
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
                "hparams": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                            for k, v in hp.items()},
                "non_block": non_block,
                "layers": rows,
                "attn_layers": attn_ids,
                "gdn_layers": gdn_ids,
                "mtp_layers": mtp_ids,
                "total_bytes": total,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
