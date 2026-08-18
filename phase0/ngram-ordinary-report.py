#!/usr/bin/env python3
"""Aggregate one ngram-ordinary.sh arm: throughput over the whole prompt set, and what each
speculator actually did.

Rate is summed, not averaged per prompt: sum(predicted_n) / sum(predicted_ms). Averaging per
prompt would weight a 40 token answer the same as a 400 token one.
"""

import json
import re
import sys

RE_STATS = re.compile(
    r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)")


def main(jsonl_path, log_path):
    pred_n = pred_ms = prompt_n = prompt_ms = 0.0
    rows = 0
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line).get("timings", {})
        pred_n   += t.get("predicted_n") or 0
        pred_ms  += t.get("predicted_ms") or 0.0
        prompt_n += t.get("prompt_n") or 0
        prompt_ms += t.get("prompt_ms") or 0.0
        rows += 1

    tg = 1000.0 * pred_n / pred_ms if pred_ms else 0.0
    pp = 1000.0 * prompt_n / prompt_ms if prompt_ms else 0.0
    print("  %d prompts | pp %.2f t/s (%d tok) | tg %.2f t/s (%d tok)" % (rows, pp, prompt_n, tg, pred_n))

    log = open(log_path, encoding="utf-8", errors="replace").read()
    # the server prints cumulative per-impl stats after each request; the last block is the total
    last = {}
    for m in RE_STATS.finditer(log):
        impl, _, n_call, _, n_gen, n_acc_d, n_gen_tok, n_acc_tok = m.groups()
        last[impl] = (int(n_call), int(n_gen), int(n_gen_tok), int(n_acc_tok))
    for impl, (n_call, n_gen, n_gen_tok, n_acc_tok) in sorted(last.items()):
        fire = 100.0 * n_gen / n_call if n_call else 0.0
        width = n_gen_tok / n_gen if n_gen else 0.0
        acc = 100.0 * n_acc_tok / n_gen_tok if n_gen_tok else 0.0
        print("  %-12s fired %5.1f%% of %-5d calls | width %5.1f tok | accepted %5.1f%% (%d/%d)" % (
            impl, fire, n_call, width, acc, n_acc_tok, n_gen_tok))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
