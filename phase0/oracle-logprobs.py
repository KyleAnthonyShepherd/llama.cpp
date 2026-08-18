#!/usr/bin/env python3
"""Compare the first-token logprobs of two llama-server /completion responses.

Rounding and corruption look different here. Reassociating a sum moves every logit by a similar
tiny amount and leaves the ranking alone. Dropping or misrouting expert rows moves a few logits a
lot and reshuffles the top of the distribution.
"""

import json
import sys


def top(path):
    d = json.load(open(path, encoding="utf-8"))
    probs = d.get("completion_probabilities") or []
    if not probs:
        raise SystemExit("no completion_probabilities in %s (needs n_probs > 0)" % path)
    top_k = probs[0].get("top_logprobs") or probs[0].get("top_probs") or probs[0].get("probs") or []
    return {e["id"]: (e["token"], e["logprob"]) for e in top_k}


def main(ref_path, test_path):
    ref, test = top(ref_path), top(test_path)

    shared = sorted(set(ref) & set(test), key=lambda i: ref[i][1], reverse=True)
    if not shared:
        raise SystemExit("no shared tokens between the two top-k sets: badly diverged")

    diffs = [abs(ref[i][1] - test[i][1]) for i in shared]
    ref_rank = sorted(ref, key=lambda i: ref[i][1], reverse=True)
    test_rank = sorted(test, key=lambda i: test[i][1], reverse=True)

    # rank agreement over the shared set says more than the raw deltas: a shifted but
    # order-preserving distribution is reassociation, a reshuffled one is not
    ref_order = [i for i in ref_rank if i in set(shared)]
    test_order = [i for i in test_rank if i in set(shared)]
    inversions = sum(1 for a, b in zip(ref_order, test_order) if a != b)

    print("  shared tokens in top-k : %d of %d / %d" % (len(shared), len(ref), len(test)))
    print("  argmax                 : %r vs %r%s" % (
        ref[ref_rank[0]][0], test[test_rank[0]][0],
        "" if ref_rank[0] == test_rank[0] else "   <-- DIFFERENT"))
    print("  rank positions moved   : %d of %d" % (inversions, len(shared)))
    print("  max |dlogprob|         : %.3e" % max(diffs))
    print("  mean |dlogprob|        : %.3e" % (sum(diffs) / len(diffs)))

    print()
    print("  %-18s %12s %12s %10s" % ("token", "ref", "test", "delta"))
    for i in shared[:10]:
        print("  %-18r %12.6f %12.6f %10.2e" % (ref[i][0], ref[i][1], test[i][1], test[i][1] - ref[i][1]))

    print()
    if ref_rank[0] != test_rank[0] or len(shared) < 0.9 * min(len(ref), len(test)):
        print("  VERDICT: the argmax or the top-k set moved. Something is wrong: check the id")
        print("           remap for duplicates before reading anything else.")
    else:
        print("  VERDICT: top-k set and argmax agree, so no gross corruption.")
        print()
        print("  This test cannot go further than that. Deltas of ~1e-1 in logprob are far")
        print("  larger than last-bit rounding, but the hot path runs on the GPU and the cold")
        print("  path on the CPU, and a per-layer difference of ~1e-3 compounds through 40")
        print("  residual layers into exactly this range. Benign reassociation and a subtle")
        print("  routing error both land here. Use perplexity over a fixed corpus to separate")
        print("  them: reassociation leaves it flat, dropped expert rows do not.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
