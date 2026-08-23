#!/usr/bin/env python3
"""Stream one /completion and record when every token arrived.

The server's own timings give one number for a whole generation, which cannot answer
"does the best draft width move inside a run". Streaming gives an arrival time per token,
so the reporter can cut the run into windows and time each one.

The arrival times carry the SSE and socket overhead on top of the decode. That is well
under a millisecond against a decode step of ~100 ms on the target box, and it is the
same for every arm, so it does not move a comparison.

Writes one json object per line:
  {"tok": 0, "t_us": 0, "text": "The"}          one per token
  {"timings": {...}, "content": "...", ...}     one at the end
"""

import argparse
import json
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8080")
    ap.add_argument("--prompt", required=True, help="file holding the prompt text")
    ap.add_argument("--n-predict", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt, encoding="utf-8").read()

    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": 0,
        "seed": args.seed,
        "stream": True,
        "cache_prompt": False,
    }

    req = urllib.request.Request(
        "http://127.0.0.1:%s/completion" % args.port,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    text = []
    t0 = None
    n = 0

    with urllib.request.urlopen(req, timeout=3600) as resp, open(args.out, "w", encoding="utf-8") as out:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue

            ev = json.loads(line[5:])
            now = time.monotonic()

            if t0 is None:
                # the first token carries the whole prompt-processing wait, so it starts
                # the clock instead of being timed by it
                t0 = now

            piece = ev.get("content", "")
            if piece:
                out.write(json.dumps({"tok": n, "t_us": int((now - t0)*1e6), "text": piece}) + "\n")
                text.append(piece)
                n += 1

            if ev.get("stop"):
                ev.pop("content", None)
                ev["content"] = "".join(text)
                out.write(json.dumps(ev) + "\n")

    print("  %d tokens streamed -> %s" % (n, args.out), file=sys.stderr)


if __name__ == "__main__":
    main()
