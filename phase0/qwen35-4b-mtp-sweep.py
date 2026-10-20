#!/usr/bin/env python3
"""Sweep MTP draft settings on llama-server: one server per config, fixed prompts, read server timings."""
import argparse, json, os, subprocess, sys, time, urllib.request

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get and put in O(1), using a dict and a doubly linked list. Include type hints, docstrings and five unittest test cases.",
    "prose": "Explain to a curious twelve-year-old how a refrigerator keeps food cold. Cover the compressor, the refrigerant, the coils and why the back of the fridge feels warm. Write about 400 words.",
    "extract": "Turn this into a JSON array of objects with fields name, role, start_year, city:\n\nMaria Chen joined Fernwood Labs as lead chemist in 2014 and works out of Portland. Dev Patel has been the CFO since 2019, based in Austin. In 2021 Sara Okafor became head of design in the Lagos office, and Tom Berg, a field engineer since 2011, splits time between Oslo and Bergen but is listed in Oslo. Lena Fischer was hired in 2023 as a data analyst in Berlin. Carlos Ruiz, operations manager, started in 2016 in Madrid. Aiko Tanaka, a researcher, started in 2020 in Osaka. Then give a one-paragraph summary of the team.",
    "mixed": "Write a short blog post for beginners about reading CSV files in Python. Start with two paragraphs on why CSV is everywhere and what can go wrong with it, then show a complete example using the csv module, then the same task with pandas, and end with a paragraph of advice on encodings and delimiters.",
    "reason": "A train leaves city A at 9:00 travelling at 80 km/h towards city B, 360 km away. A second train leaves B at 10:00 towards A at 100 km/h. At what time do they meet, and how far from A? Solve it step by step, then check the answer a second way.",
}

SAMPLING = {
    "greedy": {"temperature": 0, "top_k": 1},
    "qwen": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
}


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", required=True, help="semicolon-separated list of 'name=extra args'")
    ap.add_argument("--port", default="8090")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--sampling", default="greedy,qwen")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--base", default="-ngl 99 -fa on -c 8192 -np 1 -t 6 --no-webui")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    os.makedirs(a.logdir, exist_ok=True)
    out = open(a.out, "a", encoding="utf-8")
    for cfg in a.configs.split(";"):
        name, _, extra = cfg.partition("=")
        name = name.strip()
        args = [a.server, "-m", a.model, "--port", a.port] + a.base.split() + extra.split()
        log = open(f"{a.logdir}/server-{name}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_ready(a.port, proc):
                print(f"{name}: server failed to start", file=sys.stderr)
                continue
            post(a.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8})
            for samp in a.sampling.split(","):
                for seed in a.seeds.split(","):
                    for pname, ptext in PROMPTS.items():
                        if a.only and pname not in a.only.split(","):
                            continue
                        body = {"messages": [{"role": "user", "content": ptext}], "max_tokens": a.n, "seed": int(seed),
                                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
                        body.update(SAMPLING[samp])
                        r = post(a.port, "/v1/chat/completions", body)
                        t = r.get("timings", {})
                        rec = {"cfg": name, "sampling": samp, "seed": seed, "prompt": pname,
                               "n": t.get("predicted_n"), "tps": t.get("predicted_per_second"),
                               "pp_tps": t.get("prompt_per_second"),
                               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                               "content": r["choices"][0]["message"].get("content", "")}
                        out.write(json.dumps(rec) + "\n"); out.flush()
                        acc = (rec["draft_acc"] / rec["draft_n"]) if rec.get("draft_n") else None
                        print(f"{name:14s} {samp:6s} {pname:8s} n={rec['n']} tg={rec['tps']:.1f} acc={acc if acc is None else round(acc, 3)}", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(20)
            except Exception:
                proc.kill()
            log.close()


if __name__ == "__main__":
    main()
