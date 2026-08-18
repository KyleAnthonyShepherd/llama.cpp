#!/usr/bin/env python3
"""Summarise one ngram-mod-ab.sh case: throughput, draft fire rate, expert union.

Reads the /completion response json and the server log. Everything here is
grepped out of stock -lv 4 output except the union field, which needs the
LLAMA_EXPERT_HITRATE reporting in llama_expert_hotstore::log_hit_rate.
"""

import json
import re
import sys
from collections import Counter

RE_STATS = re.compile(
    r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)")
RE_HIT = re.compile(r"expert hot hit rate: (\d+)/(\d+) = ([\d.]+)% \(m=(\d+), union ([\d.]+)/(\d+)\)")
RE_BYPASS = re.compile(r"expert tier bypassed: n_tokens=(\d+) \(max (\d+)\)")
RE_ENGAGED = re.compile(r"expert tier engaged: n_tokens=(\d+)")
RE_SIZING = re.compile(r"Expert hotstore sizing \(S=(\d+)\)")
RE_VERIF = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")


def main(json_path, log_path):
    d = json.load(open(json_path, encoding="utf-8"))
    t = d.get("timings", {})
    print("  pp %.2f t/s (%s tok) | tg %.2f t/s (%s tok)" % (
        t.get("prompt_per_second") or 0, t.get("prompt_n"),
        t.get("predicted_per_second") or 0, t.get("predicted_n")))

    log = open(log_path, encoding="utf-8", errors="replace").read()

    sizing = RE_SIZING.search(log)
    if sizing:
        print("  hot store: S=%s slots" % sizing.group(1))

    for m in RE_STATS.finditer(log):
        impl, _, n_call, _, n_gen, n_acc_d, n_gen_tok, n_acc_tok = m.groups()
        n_call, n_gen, n_gen_tok = int(n_call), int(n_gen), int(n_gen_tok)
        fire = 100.0 * n_gen / n_call if n_call else 0.0
        width = n_gen_tok / n_gen if n_gen else 0.0
        print("  %-12s fire rate %5.1f%% (%d/%d draft calls), mean draft width %.1f tok, "
              "accepted %s/%s tok" % (impl, fire, n_gen, n_call, width, n_acc_tok, n_gen_tok))

    v = RE_VERIF.search(log)
    if v:
        # a decode step either verifies a draft or emits one token, so total steps is
        # tokens produced minus the ones that came from an accepted draft token
        predicted = t.get("predicted_n") or 0
        accepted = int(v.group(2))
        steps = predicted - accepted
        print("  verify steps: draft acceptance %s, mean accepted len %s; "
              "%d of ~%d decode steps were wide batches (%.1f%%)" % (
                  v.group(1), v.group(4), _verif_steps(log), steps,
                  100.0 * _verif_steps(log) / steps if steps else 0.0))

    hits = [m.groups() for m in RE_HIT.finditer(log)]
    if hits:
        by_m = {}
        for h, tot, pct, m_tok, union, n_exp in hits:
            by_m.setdefault(int(m_tok), []).append((int(h), int(tot), float(union), int(n_exp)))
        for m_tok in sorted(by_m):
            rows = by_m[m_tok]
            h = sum(r[0] for r in rows)
            tot = sum(r[1] for r in rows)
            union = sum(r[2] for r in rows) / len(rows)
            n_exp = rows[0][3]
            print("  m=%-3d n=%-5d hit rate %5.1f%%  union %6.1f / %d experts (%.0f%%)" % (
                m_tok, len(rows), 100.0 * h / tot, union, n_exp, 100.0 * union / n_exp))

    eng = Counter(m.group(1) for m in RE_ENGAGED.finditer(log))
    if eng:
        print("  tier ENGAGED at n_tokens: %s" % ", ".join(sorted(eng, key=int)))
    byp = Counter(m.group(1) for m in RE_BYPASS.finditer(log))
    if byp:
        print("  tier bypassed at n_tokens: %s" % ", ".join(sorted(byp, key=int)))


def _verif_steps(log):
    m = re.search(r"#gen drafts =\s*(\d+)", log)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
