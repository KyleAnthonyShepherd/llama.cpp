#!/usr/bin/env python3
"""Summarise spec-width-sweep.sh runs.

  one     <jsonl> [log]              one arm: mean and per-window throughput
  compare <jsonl> [jsonl ...]        per-window throughput of every arm, best marked
  cold    <log> [log ...]            cold experts against batch width, the linearity check

Window throughput comes from the streamed arrival times, so it needs no server support.
The cold and hit-rate numbers come from LLAMA_EXPERT_HITRATE lines in the server log and
are mapped back to a token index with the LLAMA_TRACE acceptance lines.
"""

import argparse
import json
import os
import re
import sys

RE_HIT = re.compile(
    r"expert hot hit rate: (\d+)/(\d+) = ([\d.]+)% \(m=(\d+), union ([\d.]+)/(\d+)\) cold (\d+)")
RE_ACC = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens")
RE_SIZING = re.compile(r"Expert hotstore sizing \(S=(\d+)\)")
RE_STATS = re.compile(
    r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), "
    r"#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)")
RE_ACC_POS = re.compile(r"acceptance per draft position:\s*(.*)")

WINDOW = 100


def load(jsonl):
    """-> (name, [t_us per token], final event or {})"""
    toks, final = [], {}
    for line in open(jsonl, encoding="utf-8"):
        ev = json.loads(line)
        if "tok" in ev:
            toks.append(ev["t_us"])
        else:
            final = ev
    name = os.path.basename(jsonl).replace(".jsonl", "")
    return name, toks, final


def windows(toks, size=WINDOW):
    """-> [(first_tok, tokens_per_second)] over closed windows of `size` tokens"""
    out = []
    for beg in range(0, len(toks) - size, size):
        dt = toks[beg + size] - toks[beg]
        if dt > 0:
            out.append((beg, 1e6*size/dt))
    return out


def overall(toks):
    if len(toks) < 2:
        return 0.0
    # the first token started the clock, so it is not one of the tokens this times
    return 1e6*(len(toks) - 1) / (toks[-1] - toks[0])


def log_series(log_path):
    """Walk the log in order, pairing each hit-rate line with the token it landed on.

    A hit-rate line is one decode. The acceptance line that follows it says how many of
    that decode's draft tokens survived, so the run advanced 1 + accepted tokens. Without
    LLAMA_TRACE there is no acceptance line and every decode counts as one token, which is
    right at width 0 and wrong above it.
    """
    rows = []
    tok = 0
    pending = None

    for line in open(log_path, encoding="utf-8", errors="replace"):
        h = RE_HIT.search(line)
        if h:
            if pending is not None:
                rows.append((tok, pending))
                tok += 1
            hits, total, _pct, m, union, n_exp, cold = h.groups()
            pending = {
                "m": int(m), "hits": int(hits), "total": int(total),
                "union": float(union), "n_exp": int(n_exp), "cold": int(cold),
            }
            continue

        a = RE_ACC.search(line)
        if a and pending is not None:
            rows.append((tok, pending))
            # a restored checkpoint rolls the step back and replays it, so the run does not
            # advance here - the replay has its own acceptance line and that one moves it.
            # On a hybrid target every partial acceptance takes this path, so it is the
            # common case, not a corner
            if "restore checkpoint" not in line:
                tok += 1 + int(a.group(1))
            pending = None

    if pending is not None:
        rows.append((tok, pending))

    return rows


def cmd_one(args):
    name, toks, final = load(args.jsonl)
    print("=== %s ===" % name)
    print("  %d tokens, %.2f t/s overall" % (len(toks), overall(toks)))

    t = final.get("timings", {})
    if t:
        print("  server timings: pp %.2f t/s (%s tok), tg %.2f t/s (%s tok)" % (
            t.get("prompt_per_second") or 0, t.get("prompt_n"),
            t.get("predicted_per_second") or 0, t.get("predicted_n")))
        if t.get("draft_n_accepted") is not None:
            print("  draft: %s/%s accepted" % (t.get("draft_n_accepted"), t.get("draft_n")))

    for beg, tps in windows(toks):
        print("    tok %4d-%4d  %6.2f t/s" % (beg, beg + WINDOW, tps))

    if not args.log or not os.path.exists(args.log):
        return

    log = open(args.log, encoding="utf-8", errors="replace").read()

    s = RE_SIZING.search(log)
    if s:
        print("  hot store: S=%s slots" % s.group(1))

    for m in RE_STATS.finditer(log):
        impl, _, n_call, _, n_gen, _n_acc_d, n_gen_tok, n_acc_tok = m.groups()
        n_call, n_gen, n_gen_tok = int(n_call), int(n_gen), int(n_gen_tok)
        print("  %-12s drafted on %d/%d calls, mean width %.1f tok, accepted %s/%s" % (
            impl, n_gen, n_call, n_gen_tok/n_gen if n_gen else 0.0, n_acc_tok, n_gen_tok))

    rows = log_series(args.log)
    if not rows:
        return

    print("  per window, from the log:")
    for beg in range(0, len(toks), WINDOW):
        sel = [r for tok, r in rows if beg <= tok < beg + WINDOW]
        if not sel:
            continue
        hits = sum(r["hits"] for r in sel)
        total = sum(r["total"] for r in sel)
        print("    tok %4d-%4d  hit %5.1f%%  union %6.1f  cold %6.1f  m %.2f  (%d decodes)" % (
            beg, beg + WINDOW,
            100.0*hits/total if total else 0.0,
            sum(r["union"] for r in sel)/len(sel),
            sum(r["cold"] for r in sel)/len(sel),
            sum(r["m"] for r in sel)/len(sel),
            len(sel)))


def cmd_compare(args):
    arms = [load(p) for p in args.jsonl]
    arms = [(n, t) for n, t, _ in arms if len(t) > WINDOW]
    if not arms:
        print("no arm has more than %d tokens" % WINDOW)
        return

    per_arm = {n: dict(windows(t)) for n, t in arms}
    names = [n for n, _ in arms]
    begs = sorted({b for w in per_arm.values() for b in w})

    print("  %-14s %s" % ("window", "".join("%12s" % n[-11:] for n in names)))
    n_best = {}
    for beg in begs:
        row = [per_arm[n].get(beg) for n in names]
        best = max((v for v in row if v is not None), default=None)
        cells = []
        for n, v in zip(names, row):
            if v is None:
                cells.append("%12s" % "-")
            elif v == best:
                n_best[n] = n_best.get(n, 0) + 1
                cells.append("%11.2f*" % v)
            else:
                cells.append("%12.2f" % v)
        print("  %-14s %s" % ("%d-%d" % (beg, beg + WINDOW), "".join(cells)))

    print("  %-14s %s" % ("overall", "".join("%12.2f" % overall(t) for _, t in arms)))
    print("  %-14s %s" % ("windows won", "".join("%12d" % n_best.get(n, 0) for n in names)))

    if len(n_best) > 1:
        print("\n  more than one arm wins a window: the best width moves inside the run, so a")
        print("  controller has something to win. See PLAN-adaptive-draft-width.md section 6, E2.")
    else:
        print("\n  one arm wins every window: the best width does not move. Re-tune the constant")
        print("  and stop - PLAN-adaptive-draft-width.md section 6 calls this the kill criterion.")


def cmd_cold(args):
    print("  %-24s %5s %8s %8s %8s" % ("arm", "m", "cold", "cold/m", "hit%"))
    for path in args.log:
        rows = log_series(path)
        if not rows:
            continue
        by_m = {}
        for _tok, r in rows:
            by_m.setdefault(r["m"], []).append(r)
        name = os.path.basename(path).replace(".log", "")
        for m in sorted(by_m):
            sel = by_m[m]
            cold = sum(r["cold"] for r in sel)/len(sel)
            total = sum(r["total"] for r in sel)
            hits = sum(r["hits"] for r in sel)
            print("  %-24s %5d %8.1f %8.2f %8.1f" % (
                name[-24:], m, cold, cold/m, 100.0*hits/total if total else 0.0))

    print("\n  cold/m flat across m means the cold count is linear in the batch, which is what")
    print("  the controller assumes. A falling cold/m means cold draws repeat across tokens and")
    print("  a wider draft is cheaper than the model says.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("one")
    p.add_argument("jsonl")
    p.add_argument("log", nargs="?")
    p.set_defaults(fn=cmd_one)

    p = sub.add_parser("compare")
    p.add_argument("jsonl", nargs="+")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("cold")
    p.add_argument("log", nargs="+")
    p.set_defaults(fn=cmd_cold)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
