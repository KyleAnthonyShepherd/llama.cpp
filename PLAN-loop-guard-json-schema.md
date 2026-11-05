# PLAN - loop guard notes break JSON-schema replies (HTTP 500 "peg-native")

A request with a JSON schema (`response_format: json_schema`, or anything else that builds an
output-format grammar) and thinking on fails when the loop guard adds a note inside the thinking
block. This plan fixes it in the server, so every client is safe; geoui (the document
transcription client in `W:\Kyle_Shepherd\geocrawler\geoui`) works around it today.

Working agreement as in `NOTES.md`: discovery before code, check every symbol and line number
below against the tree (they are from commit `1c9ddcb46`), small commits, log what you find.

---

## 1. The symptom

Seen by geoui on `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_S` with Kyle's flags (`--loop-guard
reflect`, `--chat-template-file models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja`,
`--reasoning-format deepseek --reasoning-preserve`, draft-mtp n-max 1, `-np 1`), on transcription
requests: page images, a JSON schema, thinking on, `reasoning_budget_tokens` 6000, streaming.

- The stream stops sending deltas right after the guard's `note` event; nothing more arrives
  until the end.
- The request ends with `{"code": 500, "message": "The model produced output that does not
  match the expected peg-native format", "type": "server_error"}`.
- The server log has `common_chat_peg_parse: unparsed peg-native output: ```json`, and the
  reply text holds a stray `</think>` among the JSON.

Reproduced on demand (section 6): the same request fails every time with the default
`loop_guard_think_after` off. With `loop_guard_think_after: true` in the request it does not.

## 2. The cause

The output-format grammar is not lazy, and it covers the thinking block: at sampler init the
grammar is fed the tokens of the generation prompt, so it starts in the right place.

- `common/sampling.cpp` `common_sampler_init` (around line 280-308): computes `prefill_tokens`
  from `params.generation_prompt` (dropping a leading space token some tokenizers add), and if
  `grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)` accepts them
  into `grmr`. `common_grammar_needs_prefill` (`common/common.h:219`) is true for
  `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT` and `COMMON_GRAMMAR_TYPE_TOOL_CALLS`.
- With thinking on the generation prompt ends in `<think>\n`, so the grammar starts inside the
  thinking block: it allows free text up to `</think>`, then only the schema.

The guard closes reply A and opens reply B without telling the grammar:

- `tools/server/server-context.cpp` `guard_inject()` (around line 2719): appends `add` (the
  note frame `guard_frame()`: `pre_think` or `pre_content`, the note, then `post_on` or
  `post_off`) to the prompt, and feeds those tokens only to `common_sampler_accept_prompt()`.
- `common_sampler_accept_prompt()` (`common/sampling.cpp:803`) feeds only `rbudget`, never
  `grmr`.
- With `loop_guard.think_after` off (the default, see `PLAN-loop-guard.md` section 9.3: a new
  thinking block walked back into the loop), B opens with `post_off`: thinking already closed
  (`<think>\n\n</think>\n\n`). B's tokens are content.
- The grammar never saw A's close, the system note or B's opening. It still believes it is
  inside A's thinking block, so it lets anything through: the model writes ```` ```json ````,
  the schema is not enforced, and to satisfy the grammar the model eventually emits
  `</think>`.
- The streaming parser for B (`task_result_state::guard_split()` in
  `tools/server/server-task.cpp:175`, generation prompt = `gen_off`) expects JSON content and
  cannot parse a fence: no deltas, and the final parse throws the 500.

With `think_after` on, B opens with `<think>\n`: the stale grammar state ("inside thinking")
happens to match again, which is why that request field works around it.

## 3. The fix

Restart the grammar for reply B the way `common_sampler_init` starts it for a request.

1. `common/sampling.cpp`: move the prefill-token computation out of `common_sampler_init` into a
   static helper (`generation_prompt` -> tokens, with the same leading-space rule), and use it
   in both places.
2. Add `void common_sampler_restart_grammar(struct common_sampler * gsmpl, const std::string &
   generation_prompt)` (declare it in `common/sampling.h` next to
   `common_sampler_accept_prompt`):
   - no `grmr`, or `params.grammar_lazy`: return. A lazy grammar (tool calls) is not
     triggered while the guard acts: the guard is paused in the `forced` phase
     (`PLAN-loop-guard.md` section 8.2, "The third phase is called `forced`").
   - `llama_sampler_reset(gsmpl->grmr)`: back to the root state
     (`llama_sampler_grammar_reset`, `src/llama-sampler.cpp:2700`, re-inits from the stored
     grammar string).
   - if `common_grammar_needs_prefill(params.grammar)`: accept the prefill tokens of
     `generation_prompt` into `grmr`. A user grammar (`COMMON_GRAMMAR_TYPE_USER`) is not
     prefilled at init either, so it only gets the reset: B is a new reply.
   - an exception while accepting: log it like init does, and leave the grammar reset (do not
     crash the slot). Decide in discovery whether the request should then fail cleanly instead.
3. `guard_inject()`: after the `common_sampler_accept_prompt` loop, call
   `common_sampler_restart_grammar(slot.smpl.get(), generation_prompt)` with the same
   `generation_prompt` it already picks (`think_on ? guard_tmpl.gen_on : guard_tmpl.gen_off`)
   and sends to the parser. `gen_on` / `gen_off` come from `res.generation_prompt` of the same
   `common_chat_templates_apply` call that makes a request's generation prompt (around line
   2164), so the grammar and the parser see the same text.
4. Check the other ways B starts:
   - the reflection path (`guard_begin_reflection` and its finish, around line 2861-3098) ends
     in `guard_inject()` too; confirm the request's own sampler (not the reflection's) is the
     one restarted, after the parked state is restored.
   - the non-stream path (`g.segments`, same function): same call, nothing extra.
   - speculative decoding (draft-mtp, ngram-mod): drafts are checked with the target sampler;
     confirm no draft sampler keeps its own grammar copy that also needs the restart.
   - `common_sampler_get_phase()` after the restart: for B with `gen_off` it must report
     content / forced, as now.
5. Leave `loop_guard_think_after` as it is (off by default). The fix makes both settings
   correct.

Alternative, if the restart turns out unsafe somewhere: force `think_after` on for requests
with a non-lazy grammar. Smaller, but it brings back the extra thinking block that `PLAN-loop-guard.md` section 9.3
found walks back into the loop, so only as a fallback.

## 4. Build

`build-cuda`, Release, target `llama-server` (plus `test-sampling` / any test you touch). On
this machine read the user's CLAUDE.md first: commands over ~7.7 KB fail in the agent shell, and
any JVM needs the loopback workaround (not relevant here). Work in a git worktree on a new
branch from the branch Kyle runs (ask if unsure; `git branch` shows several), so the checked-out
tree is not disturbed. Commit; do not push.

## 5. Tests

Cheap loop first, then the real model:

- **CPU, no GPU** (as in `PLAN-loop-guard.md` sections 6 and 8.6): `CUDA_VISIBLE_DEVICES=""`, a small
  Qwen3.5 GGUF from `I:\Kyle_Shepherd\huggingface\hub` (0.8B or 2B, both have mmproj; text is
  enough here), its loop guard template. The per-request fields `loop_guard_line_repeats: 2`
  and `loop_guard_rate: 0.2` make the guard trip early in thinking on almost any request; fix
  the seed.
- **The 35B** with Kyle's command above, `--api-key testkey`, port 8080.

Matrix (each with a JSON schema, thinking on, streaming unless noted):

| case | before the fix | after |
|---|---|---|
| `note`, think_after off | 500 peg-native, no deltas after the note | valid schema JSON, deltas after the note |
| `reflect`, think_after off | same | same as above |
| `note`, think_after on | works | still works |
| non-stream, think_after off | 500 | valid JSON, `loop_guard` segments in the result |
| two notes in one reply | 500 | valid JSON |
| thinking off, JSON schema | guard does not act (forced phase) | unchanged |
| tool-call request (lazy grammar), guard trips in thinking | works | unchanged |
| no grammar at all | works | unchanged |

Also re-run the cache check of `PLAN-loop-guard.md` section 9.4: the follow-up request that
holds the split messages still reuses the whole cache (prompt_n in the 20s). The restart must
not change any token in the prompt, only the sampler.

Done when every row of the matrix holds on the 35B, the cache check holds, and the existing
loop guard tests in `I:\Kyle_Shepherd\toolbag\tests\loop_guard` still pass. Add the short
findings to `PLAN-loop-guard.md` (a line in section 8.5, Known limits, pointing here).

## 6. Repro script

Talks to the server directly (Python 3, standard library only). `python repro.py note` fails
before the fix; `python repro.py note think_after` passes before and after; after the fix both
pass.

```python
import json, sys, time, urllib.request

URL, KEY = "http://127.0.0.1:8080/v1/chat/completions", "testkey"
schema = {"type": "object", "properties": {"samples": {"type": "array", "items": {"type": "object",
          "properties": {"sample": {"type": "string"}, "depth": {"type": "string"}, "n": {"type": "string"}},
          "required": ["sample", "depth", "n"]}}}, "required": ["samples"]}
q = ("A boring log lists samples S-1 to S-12 every 2.5 ft from 0 ft, N values 5, 8, 12, 15, 9, 22, 30, "
     "18, 11, 7, 25, 40. Check each sample's depth range and N twice, carefully, line by line, then give the JSON.")
body = {"messages": [{"role": "user", "content": q}], "stream": True,
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 6000, "reasoning_budget_tokens": 3000,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 1,
        "loop_guard": sys.argv[1] if len(sys.argv) > 1 else "note",
        "loop_guard_line_repeats": 2, "loop_guard_rate": 0.2}
if "think_after" in sys.argv:
    body["loop_guard_think_after"] = True
req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                              "Authorization": f"Bearer {KEY}"})
content, notes, after_note, t = [], 0, 0, time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            if (ev.get("loop_guard") or {}).get("event") == "note":
                notes += 1; content = []
                continue
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    after_note += notes > 0
                content.append(d.get("content") or "")
    text = "".join(content)
    json.loads(text)
    print(f"OK {time.time() - t:.0f}s notes={notes} deltas_after_note={after_note}: {text[:120]!r}")
except Exception as e:
    print(f"FAILED {time.time() - t:.0f}s notes={notes} deltas_after_note={after_note}: {e}")
```

A run with `notes=0` did not trip the guard: change the seed or lower `loop_guard_rate`.

## 7. Out of scope

- geoui's workaround (`loop_guard_think_after: true` on JSON requests with thinking, then one
  retry with `loop_guard: log` on this error or on a reply the guard stopped with no answer) can
  stay; it is harmless after the fix. Kyle may drop it later.
- What the guard counts as a loop, its thresholds, and the reflection prompt.

## 8. Results, 2026-10-26

Branch `loop-guard-json-schema` (from `expert-prefetch-adaptive-ub` at `1c9ddcb46`), worktree
`I:\Kyle_Shepherd\opt-llama-lgjs`. Not pushed.

Discovery (every symbol and line of sections 2-3 matched the tree):

- `common_sampler_restart_grammar` takes the vocab as well: `common_sampler` keeps no model or
  vocab, and the server has `vocab` at hand.
- A prefill the grammar refuses now fails the request cleanly (`send_error` + `release`, same as a
  failed restore after a reflection) instead of going on with a broken grammar. The restart runs
  at the top of `guard_inject()`, before the note event, so a refused prefill sends no half note.
  Not seen in any test.
- Reflection: `guard_end_reflection` moves the parked sampler back into `slot.smpl` before
  `guard_inject()`, so the request's sampler is the one restarted. The reflection's own sampler
  (user grammar) is dropped.
- Speculative decoding: the draft samplers (`common/speculative.cpp`) are built without a grammar,
  and `guard_act` only runs with no draft in flight. The verify step's clone/copy of the target
  sampler lives inside one verify. Nothing else to restart.
- Phase: `common_sampler_get_phase` reports forced for any non-lazy grammar once the reasoning
  budget is out of counting, before and after the restart. Unchanged.

Tests. 0.8B on CPU (`--loop-guard note`, `loop_guard_rate` 0.2-0.35, `line_repeats` 2, seed 1):
unchanged build = 500 peg-native for note, reflect and non-stream with `think_after` off; fixed
build = valid JSON in every one, deltas after the note. The toolbag `lg_client.py` cases (note,
reflect, stream, non-stream, phase content, thinking off, logprobs, `--followup`) and
`lg_cancel.py` pass; they need `--rate 0.3 --lines 2` to trip with the section 9 thresholds.

35B (unsloth UD-Q4_K_S, Kyle's flags + `--api-key testkey`, port 8080, fixed build), script of
section 6 with `--max 6000`, budget 3000:

| case | result |
|---|---|
| `note`, think_after off | valid JSON, 1 note, 521 deltas after it |
| `reflect`, think_after off | valid JSON, 1 note (reflection ran), 509 deltas after it |
| `note`, think_after on | valid JSON |
| non-stream, `note` / `reflect` | valid JSON, `loop_guard.messages` holds the split |
| two notes (think_after on, rate 0.17, seed 2) | valid JSON after 2 notes |
| thinking off, JSON schema | guard does not act, valid JSON |
| tool call (lazy grammar), trip in thinking | tool call, unchanged |
| no grammar | unchanged (2 notes, second one in content) |
| follow-up after `note` / `reflect` | 21 / 20 new prompt tokens, rest from cache |

No `unparsed` line and no error in the server log over all runs. Two notes with a schema only
happen with `think_after` on (with it off the reply after the note is in the forced phase and the
guard does not act); other seeds/rates there tripped a third time and ended with no answer, which is
the guard's normal stop, not this bug.
