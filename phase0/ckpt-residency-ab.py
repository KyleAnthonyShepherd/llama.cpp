#!/usr/bin/env python3
"""
Root/tail checkpoint A/B/C harness for llama-server.

Models the layered-context edit loop:

    root = system prompt + reference material           (never changes)
    tail = root + document + edit request               (changes per document)
    turn B appends the model-requested extra context    (pure append to tail)
    turn C swaps the document                           (diverges right after root)

Turn B is an exact prefix extension, so it costs nothing on any arm. Turn C is
the one that needs a rollback to the root boundary, and that is what the arms
differ on.

Arms
----
  none  baseline. no rollback point anywhere. turn C reprocesses everything.
  ram   automatic context checkpoints, host RAM. a checkpoint lands on the
        document boundary because we declare it via message_delimiters.
  disk  explicit named root state via /slots?action=save|restore.
  vram  NOT IMPLEMENTED. see the note at the bottom of this file.

Server launch, one per arm (same model/ngl/fit flags for all three):

  none  --ctx-checkpoints 0 --cache-ram 0
  ram   --ctx-checkpoints 8 --checkpoint-min-step 128 --cache-ram 0
  disk  --ctx-checkpoints 0 --cache-ram 0 --slot-save-path /path/to/states/

--cache-ram is held at 0 on every arm on purpose. It restores whole prompt
states independently and would mask what the arm under test is doing. Sweep it
separately once the arms are understood.

Metric: timings.prompt_n, the tokens actually processed. timings.cache_n is the
tokens reused. Total prompt = cache_n + prompt_n.

Stdlib only. Python 3.8+.

Two things your agent should know before reading the code
The root boundary is declared, not inferred. message_delimiters is a plain request field on /completion (server-context.cpp:4304), not chat-endpoint-only. Declaring <|im_start|>user\n is what makes the server break the batch and checkpoint at each user message start. Without it there are no message spans and only the {4 + n_ubatch, 4} tail ladder applies - which is the ~500 you started with. The marker strings in the script must match the GGUF's template byte for byte.

Disk restore is append-only. SLOT_RESTORE calls slot->prompt.clear() (:2666), dropping the checkpoint list. So a restored state whose tokens are not an exact prefix of the incoming prompt leaves nothing to fall back to and resets to zero - worse than not restoring. That is why the script restores root only before a divergent turn and never before an append.

What I have and haven't verified
Syntax-checked, and the vram arm path runs. The ram and disk arms are untested against a live server - I have no model loaded here. The one assumption I'd check first is n_predict: 0 for establishing a clean root: has_budget() returns false at n_remaining == 0 (:443) so it should stop right after the prompt, and the script warns if predicted_n > 0, but that warning firing would mean every saved root state has a generated token baked in.

The result I expect
ram and disk land on the same prompt_n, because they restore to the same boundary. If that happens, the A/B/C has already answered itself: the arms are not competing on prompt processing, only on io cost and page-cache pressure, and on the 16 GB box the disk arm's page-cache eviction is the only real differentiator. none should be dramatically worse on swap-doc and identical on append - that gap is the whole value of the mechanism, and it's worth having the number.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# Qwen ChatML markers. Adjust if the GGUF's template differs - these strings
# must match the template byte for byte or the delimiters will not be found.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"

# Declaring the user delimiter is what makes the server break the batch and
# checkpoint at each user message start (server-context.cpp:3558). Without it
# the request has no message spans and only the {4 + n_ubatch, 4} tail ladder
# applies. This field is request-settable on /completion, not just on the chat
# endpoint.
MESSAGE_DELIMITERS = [{"role": "user", "delimiter": IM_START + "user\n"}]


def msg(role, body):
    return IM_START + role + "\n" + body + IM_END


def build(system, reference, document=None, request=None, extra=None):
    """Build the raw prompt. Everything up to and including the reference
    message is the root; it must be byte-identical across every turn."""
    p = msg("system", system) + msg("user", reference)
    if document is not None:
        p += msg("user", document + "\n\n" + request)
    if extra is not None:
        p += msg("user", extra)
    if document is not None:
        p += IM_START + "assistant\n"
    return p


def root_prefix(system, reference):
    return build(system, reference)


class Server:
    def __init__(self, url, slot, timeout):
        self.url = url.rstrip("/")
        self.slot = slot
        self.timeout = timeout

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def completion(self, prompt, n_predict, delimiters=True):
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": True,
            "id_slot": self.slot,
            "temperature": 0.0,
            "stream": False,
        }
        if delimiters:
            payload["message_delimiters"] = MESSAGE_DELIMITERS
        t0 = time.perf_counter()
        r = self._post("/completion", payload)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        t = r.get("timings") or {}
        return {
            "cache_n": t.get("cache_n", -1),
            "prompt_n": t.get("prompt_n", -1),
            "prompt_ms": t.get("prompt_ms", float("nan")),
            "predicted_n": t.get("predicted_n", -1),
            "wall_ms": wall_ms,
        }

    def slot_save(self, filename):
        t0 = time.perf_counter()
        r = self._post("/slots/%d?action=save" % self.slot, {"filename": filename})
        r["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return r

    def slot_restore(self, filename):
        t0 = time.perf_counter()
        r = self._post("/slots/%d?action=restore" % self.slot, {"filename": filename})
        r["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return r

    def slot_erase(self):
        return self._post("/slots/%d?action=erase" % self.slot, {})


def establish_root(srv, arm, system, reference, state_file):
    """Get a rollback point at the end of the reference message.

    n_predict = 0 stops the slot right after the prompt (has_budget() returns
    false at n_remaining == 0), so slot.prompt.tokens holds the root and
    nothing else. Any generated token would be baked into the saved state.
    """
    r = srv.completion(root_prefix(system, reference), n_predict=0)
    if r["predicted_n"] > 0:
        print("WARN: n_predict=0 still generated %d token(s); the saved root "
              "state is contaminated" % r["predicted_n"], file=sys.stderr)

    if arm == "disk":
        s = srv.slot_save(state_file)
        return {"n_bytes": s.get("n_bytes"), "n_tokens": s.get("n_tokens"),
                "save_ms": s.get("t_ms"), "wall_ms": s["wall_ms"]}
    return {}


def prepare_turn(srv, arm, state_file):
    """Put the rollback point back in place before a divergent turn."""
    if arm != "disk":
        return {}
    r = srv.slot_restore(state_file)
    return {"n_bytes": r.get("n_bytes"), "restore_ms": r.get("t_ms"),
            "wall_ms": r["wall_ms"]}


def run(args):
    srv = Server(args.url, args.slot, args.timeout)

    system = "You are an editing assistant.\n" + ("Guidance line.\n" * args.system_lines)
    reference = "Reference material.\n" + ("Example paragraph.\n" * args.reference_lines)
    documents = [
        "Document revision %d.\n" % i + ("Body line.\n" * args.document_lines)
        for i in range(args.revisions)
    ]
    edit_request = "Rewrite the third paragraph to be more concise."
    extra_context = "Additional context you asked for: the style guide forbids passive voice."

    rows = []

    for rep in range(args.reps):
        srv.slot_erase()
        setup = establish_root(srv, args.arm, system, reference, args.state_file)
        if setup:
            print("rep %d root: %s" % (rep, json.dumps(setup)))

        for rev, document in enumerate(documents):
            # turn C (and the first turn A): document swap, diverges after root
            io = prepare_turn(srv, args.arm, args.state_file)
            tail = build(system, reference, document, edit_request)
            a = srv.completion(tail, n_predict=args.n_predict)
            rows.append(("swap-doc", rep, rev, a, io))

            # turn B: append the model-requested context. exact prefix, no
            # rollback needed on any arm - this one should be ~free everywhere.
            follow = build(system, reference, document, edit_request, extra_context)
            b = srv.completion(follow, n_predict=args.n_predict)
            rows.append(("append", rep, rev, b, {}))

    report(args, rows)


def report(args, rows):
    print()
    print("arm = %s" % args.arm)
    print("%-10s %4s %4s %10s %10s %11s %11s" %
          ("turn", "rep", "rev", "cache_n", "prompt_n", "prompt_ms", "io_ms"))
    for kind, rep, rev, t, io in rows:
        io_ms = io.get("wall_ms")
        print("%-10s %4d %4d %10d %10d %11.1f %11s" %
              (kind, rep, rev, t["cache_n"], t["prompt_n"], t["prompt_ms"],
               ("%.1f" % io_ms) if io_ms is not None else "-"))

    print()
    for kind in ("swap-doc", "append"):
        sel = [t["prompt_n"] for k, _, _, t, _ in rows if k == kind]
        if not sel:
            continue
        io = [i["wall_ms"] for k, _, _, _, i in rows if k == kind and "wall_ms" in i]
        line = "%-10s reprocessed tokens: median %d  min %d  max %d" % (
            kind, statistics.median(sel), min(sel), max(sel))
        if io:
            line += "   restore io: median %.1f ms" % statistics.median(io)
        print(line)

    print()
    print("What to expect if the arm is working:")
    print("  append   : prompt_n ~= len(extra context). true on every arm.")
    print("  swap-doc : none -> prompt_n ~= whole prompt.")
    print("             ram  -> prompt_n ~= len(document + edit request).")
    print("             disk -> same as ram, plus the restore io column.")
    print("  If ram and disk match on prompt_n, the only thing left to compare")
    print("  is io cost and page-cache pressure, not prompt processing.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--arm", required=True, choices=["none", "ram", "disk", "vram"])
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("--state-file", default="root.bin")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--revisions", type=int, default=3,
                   help="document swaps per rep")
    p.add_argument("--n-predict", type=int, default=32)
    p.add_argument("--system-lines", type=int, default=20)
    p.add_argument("--reference-lines", type=int, default=400,
                   help="the rarely varying medium chunk")
    p.add_argument("--document-lines", type=int, default=60,
                   help="the frequently varying small chunk")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    if args.arm == "vram":
        print(VRAM_NOTE, file=sys.stderr)
        return 2

    try:
        run(args)
    except urllib.error.HTTPError as e:
        print("HTTP %d: %s" % (e.code, e.read().decode(errors="replace")), file=sys.stderr)
        return 1
    return 0


VRAM_NOTE = """\
The vram arm is not reachable from the current server API.

Nothing in common/ or tools/ passes LLAMA_STATE_SEQ_FLAGS_ON_DEVICE (= 2).
Context checkpoints store into std::vector<uint8_t> (common/common.h:1133) and
every call site passes PARTIAL_ONLY or NONE, so they are host RAM. --cache-ram
is host RAM. /slots save is disk. There is no slot-to-slot copy endpoint.

Two candidate patches, both small. Whoever picks this up should read the code
first and own the choice:

 1. Device-resident checkpoint.
    Pass LLAMA_STATE_SEQ_FLAGS_ON_DEVICE alongside PARTIAL_ONLY in
    create_checkpoint() and in the matching load_tgt/load_dft calls
    (server-context.cpp:2412, :3390). The storage backend then becomes
    llama_io_write_device, which allocates a backend buffer of the same buffer
    type as the source tensors (llama-context.cpp:2916).
    Hard constraint: mem_storage is keyed by seq_id and reallocated on each
    get, so it is ONE snapshot per sequence, not a list. That is fatal for the
    automatic multi-rung ladder and fine for a single pinned root.

 2. Slot-to-slot copy.
    Expose llama_memory_seq_cp as a slot action. For the recurrent half this
    shares the donor's tail cell rather than copying (llama-memory-recurrent.cpp:263),
    so it is cheap. Costs one extra sequence's KV allocation to hold the donor,
    which at 20 KiB/token is ~640 MiB at 32k ctx on the 35B - likely
    unaffordable on the 6 GB box, plausible on the dev box.
    slot->prompt.tokens has to be copied alongside the memory or the LCP logic
    will disagree with the cache.

Before building either: the partial state here is ~62.8 MiB, so the host round
trip being eliminated is roughly 20 ms over PCIe 3.0. Against a document-swap
reprocess measured in seconds, that is a rounding error. Get the rollback point
right first, then decide whether the tier is worth a patch.
"""


if __name__ == "__main__":
    sys.exit(main())
