# Plan: loop guard - detect a doom loop, reflect on it, steer the agent out with a system note

Branch: `Ternary` (written against commit `eb4bfa19d`; line numbers are approximate, function names are exact).
Rev 5, 2026-10-14. Steps 0-3 are built and uncommitted. **Section 9 has the tests on Qwen3.6-35B-A3B, which changed the trigger (word novelty instead of the 8-gram rate), the reflection, and the defaults.** Section 8 has the state after the first build on the small model. Sections 1-7 are the design as first reviewed; where the code differs, sections 8 and 9 say so.

## 0. What this document decides

How the server watches a slot's generated tokens for a "perception loop", what it does when the watch trips, and what a client has to do so that its history and the server's cache stay the same.

### 0.1 Decisions taken (Kyle, 2026-10-13)

| # | Decision | Effect on the design |
|---|---|---|
| 1 | No context checkpoints. Park the agent with the full slot state in RAM, the mechanism behind hats | The only state restore is from a buffer the slot owns. Every other hop is an append (3.6) |
| 2 | Expose token probabilities now | Step 0 logs surprisal and entropy next to the n-gram and line signals, on the plain and the speculative path (3.1) |
| 3 | The note is a real system message | The reply is closed, a system turn is appended, a new assistant turn is opened (3.4) |
| 4 | The guard covers thinking and normal output, not tool calls | 3.2 |
| 5 | Keep the loop text. The agent's context is never cut, only appended to | No rollback anywhere |
| 6 | It is a server feature behind a flag | 3.8 |
| 7 | The chat template is made to fit. Unusual is fine if it is documented | A loop guard template that renders a system message anywhere, checked by a probe at startup (3.3) |
| 8 | The reflection is streamed to the client, marked as reflection | `loop_guard` events in the stream (3.9) |
| 9 | The client holds both assistant messages with the note between them | The next request renders to the same tokens the cache holds. No hidden turn (3.9) |

### 0.2 Steps

| Step | What | Size | Decides |
|---|---|---|---|
| 0 | Token stats on both sampling paths, monitor, per-token trace file. Log only | ~250 lines, one new header | thresholds, false positives, what entropy is good for |
| 1 | Loop guard template + startup probe. `note` mode: fixed system note, reply split in two, `loop_guard` events, non-stream fields | ~350 lines + one template file | does a system note break the loop, does the next request reuse the cache |
| 2 | Web UI as an aware client: split the message, hold the note, show the reflection | ~150 lines of TS | the hidden-turn problem is gone for the UI |
| 3 | `reflect` mode: park in RAM, reflect on the same slot with the text streamed, restore, inject the reflector's note | ~250 lines | does a tailored note beat the fixed one |

Toolbag and other harnesses follow the client contract in 3.9. That work is planned in toolbag, not here.

---

## 1. The failure, on the sample

The pasted transcript is ~1475 word-level tokens (real token count is higher). A scratch prototype gave:

- Exact-period detector (`is_repeating`, `tools/server/server-context.cpp:58`, the `--repetition-stop` of 358034b3e), N = 32 / 64 / 128 / 256: **no trip at any point**. The loop is near-copies with new text in between, not one period repeated.
- Share of 8-grams already produced earlier in the same generation, over the last 128 tokens: 0.04 at token 128, 0.60 at 256, **1.00 at 576**, down to **0.05-0.12 at 704-896** (the model writes a new table), back to 0.65-0.92 from 1024 to the end.
- Same rate over the last 256 tokens: first reaches 0.5 at token 546, 0.6 at 572, 0.7 at 1109.
- The line `Wait, that still doesn't make sense. Let me look at the image one more time.` (20 tokens) occurs 6 times. Copies end at tokens 275, 572, 977, 1120, 1299, 1433. 43 distinct lines out of 60.

What this says about a trip rule:
- The rate dips to near zero in the middle of the loop. The rate alone must trip within one window, or a signal with memory must cover the dips.
- The repeated anchor line is that signal. It does not reset when the model writes a new table.
- First candidate: **rate(8-gram, last 256) >= 0.6, or one line of >= 8 tokens produced 4 times**. On the sample that is token 572 (rate) or 1120 (line).
- One transcript sets no threshold, and it has no probabilities. Step 0 exists to get both.

---

## 2. What the tree already has

### 2.1 Token loop
- Every generated token passes `process_token()` (`server-context.cpp:2083`), from the plain sampling path (`post_decode()`, `:4478-4525`) and from the speculative accept loop (`:4644-4662`).
- `slot.sampled` is the token just sampled. It is **not yet in the KV cache or in `slot.prompt.tokens`**. `handle_last_sampled_token()` (`:642`) adds it at the next `pre_decode()`.

### 2.2 Probabilities
- Plain path: `populate_token_probs()` (`:2242`), only when the request sets `n_probs`. The pre-sampling variant reads raw logits by batch index (`get_token_probabilities`, `server-common.cpp:1524`).
- Speculative path: nothing. `result.prob = 1.0f; // set later` and `// TODO: set result.probs` (`:4649-4651`). With `--spec-type draft-mtp` most tokens take this path, so today `logprobs` in the API is empty for them.
- The batch indices of the verified tokens are in `slot.spec_i_batch` until it is cleared (`:4553`), and the logits stay valid until the next decode.
- `n_probs` already turns backend sampling off for a request (`:2040-2045`). The guard does the same.

### 2.3 Phase of the reply
- The reasoning-budget sampler knows "inside the thinking block": state `REASONING_BUDGET_COUNTING` (`common/reasoning-budget.h:10-16`). It exists when the request has thinking tags and a lazy grammar, a budget, or `reasoning_control` (`common/sampling.cpp:311`). The web UI always sends `reasoning_control: true` (`tools/ui/src/lib/services/chat.service.ts:1155`), API clients may not.
- "Inside a tool call" is known too: the lazy grammar has triggered (`rep_paused()`, `common/sampling.cpp:477-480`).
- Missing: a public getter for the phase. A few lines in `common/sampling.{h,cpp}`.

### 2.4 Full slot state in RAM (the hats mechanism)
- `ram_store_save()` (`:2693`) copies the whole sequence state of the target and the draft context with `LLAMA_STATE_SEQ_FLAGS_NONE`, plus `slot.prompt.clone()`. `ram_store_restore()` (`:2749`) writes it back.
- On hybrid memory the full state is the attention KV **and** the recurrent state (`src/llama-memory-hybrid.cpp:190-202`). A restore removes the whole sequence first (`state_read_meta`, `src/llama-kv-cache.cpp:2982-2984`), so cells added after the save are gone after it.
- The named store never evicts, but a client can overwrite or drop a name, and it needs `--slot-ram-limit`. The guard keeps its state in a buffer on the slot and shares only the capture and restore code.
- State of testing as last recorded (2026-10-08): the RAM path was measured on the CPU test server with Qwen3.5-0.8B. It was **not yet run on the 35B with MTP**. Step 3 starts with that test.
- Size on Qwen3.6-35B-A3B: 20 KiB per token of f16 KV over the 10 attention layers (`PLAN-qwen36-35b-moe.md` section 0.3), plus the context-independent recurrent state. About 250 MiB at 12.5k tokens, 640 MiB at 32k. One D2H copy to park, one H2D copy to restore. Estimate, not measured.

### 2.5 Appending to a slot
- The prompt path handles a request whose tokens extend what the slot holds: common prefix (`:3779`), a batched decode of the new tokens, then `SLOT_STATE_DONE_PROMPT` -> sampling (`:4185-4197`).
- For a pure append on hybrid memory it does not touch a checkpoint. The checkpoint search runs only if `pos_min >= pos_min_thold` (`:3933`). `pos_min` is the recurrent state's position `n - 1` (`llama-memory-hybrid.cpp:172-175`) and the threshold is `n` when there are new tokens (`:3881`).
- The `seq_rm(p0 = n, -1)` at `:4032` removes nothing and the recurrent memory accepts it (`src/llama-memory-recurrent.cpp:194-206`).

### 2.6 Chat templates: three behaviors for a late system message

Read from the files, the two GGUF ones with `gguf-py`'s `GGUFReader` (field `tokenizer.chat_template`):

| Template | A system message after the first turn | Thinking of earlier assistant turns |
|---|---|---|
| `models/templates/Qwen3.5-4B.jinja` (repo) | **raises** `System message must be at the beginning.` (`:83-86`) | only after the last user query (`:100`) |
| `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (157 lines) | **dropped without a word**: the loop skips every `system` / `developer` message past the first one or two, which it merges into the top block (lines 45-57, 88) | always when `preserve_thinking` is true, which this fork sets by default (`common/arg.cpp:1006-1008`) |
| `llmfan46/...heretic-Native-MTP-Preserved-GGUF` (222 lines) | **rendered** as `<|im_start|>system\n...<|im_end|>\n` (lines 111-116) | when thinking is on, `preserve_thinking` is true, and the reasoning is not empty |

All three share the pieces the guard needs:
- an assistant turn with reasoning renders as `<|im_start|>assistant\n<think>\n` + reasoning + `\n</think>\n\n` + content + `<|im_end|>\n`. Reasoning and content are trimmed
- the generation prompt is `<|im_start|>assistant\n` + `<think>\n`, or + `<think>\n\n</think>\n\n` with thinking off

The silent drop is the dangerous one: a client could hold the note correctly and the model would still never see it again, with no error anywhere. This is why 3.3 has a probe.

### 2.7 Stream plumbing
- The HTTP thread keeps the chat parser state for a request (`task_result_state`, `tools/server/server-task.h:111`) and re-parses the accumulated text on each chunk (`update_chat_msg`, `server-task.cpp:170`). One state parses one assistant message.
- Extra top-level fields on a chunk are already in use and ignored by plain OpenAI clients: `timings`, `prompt_progress` (`server-task.cpp:1173-1178`).
- The `is_begin` marker may be sent once (`server-context.cpp:5276-5278` handles only the first result).
- The UI reads chunks in `chat.service.ts:626-665`.

---

## 3. Design

### 3.1 Token stats (decision 2)

One helper in the server: from the raw logits at a batch index and the chosen token, return `logprob` (of the chosen token, temperature 1, before any sampler) and `entropy` (of the whole distribution). Two passes over the vocabulary, no sort.

- Plain path: call it after `common_sampler_sample()` (`:4478`) with `tok_idx`.
- Speculative path: call it for each accepted token with its index from `slot.spec_i_batch`, before the clear at `:4553`, on the branch that goes on to `process_token()`. The checkpoint-restore replay branch (`:4565-4590`) returns first and samples the same tokens again next step.
- `completion_token_output` gets the two numbers. `result.prob` is then real on both paths, and with `n_probs > 0` the speculative path fills `result.probs` through the existing pre-sampling branch of `populate_token_probs()`. That closes the TODO and makes API `logprobs` work under MTP.
- Computed when the guard is on or `n_probs > 0`. Cost to measure in step 0.

Raw, not post-sampler, on purpose: the question is what the model believes, and the sampler chain would hide it.

Two readings to test, they predict opposite things:
- "the model copies its own text": surprisal and entropy **fall** inside the repeated stretches
- "the model is too unsure to commit": entropy is **high** at the points where it turns back ("Wait, ...")
The trace in step 0 shows which one is there. No entropy threshold is fixed before that.

### 3.2 Monitor (decision 4)

New header `tools/server/server-loop-guard.h`, one struct per slot, fed from `process_token()` with the token id, its text piece, and the two stats.

State:
- rolling hash of the last `n` tokens (n = 8), a hash set of every n-gram seen in this phase, a ring of the last `W` hit flags (W = 256) with a running sum -> `rate`
- line tracker: hash of the tokens since the last piece with `\n`. On a line end, if the line has >= 8 tokens, bump its count -> `line_repeats` = highest count. Keep the text of that line for the reflection prompt
- running means of surprisal and entropy over the last 64 tokens

Rules:
- Phases: `thinking`, `content`, `tool_call`. The monitor runs in `thinking` and `content`. It is paused in `tool_call`: a half-written call cannot be closed cleanly, and paths and code repeat honestly there. A loop in a tool call stays with `--repetition-stop`.
- **The n-gram set and the line counts are reset at each phase change.** A model that drafts in thinking and then copies the draft into the answer is then not a hit.
- Only n-grams of the **generated** tokens count. Quoting the prompt is not a hit.
- No trip before 256 tokens in the phase.
- Trip (first candidate): `rate >= 0.6` or `line_repeats >= 4`. Thresholds are request fields, so tuning needs no rebuild.
- When the guard is on, the server sets `reasoning_control` for the request (`server-common.cpp:1399`) so the phase is known for API clients too. A model without thinking tags has only `content`.
- A trip is only **recorded** in `process_token()`. It is acted on at the top of a later `pre_decode()`, where no draft is in flight, and only at a seam-safe token (3.4).

Not used: embedding similarity between chunks. The generation context is not in embedding mode, and a second model costs VRAM this box does not have.

### 3.3 The template (decision 7)

The guard needs one thing from the template: **a system message placed after an assistant message renders as its own system turn, at that place.** Two parts make that sure.

**A loop guard template file.** `models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja`, loaded with the existing `--chat-template-file`. It is the template of the GGUF in use with one change: a `system` / `developer` message that is not part of the top block renders as `<|im_start|>system\n` + content + `<|im_end|>\n` (the heretic template already does this, the unsloth one needs the branch added at its line 88). The file starts with a Jinja comment that says which GGUF it came from, what was changed, and that `--loop-guard` needs it. The `--loop-guard` help text names the file.

Loading a file on purpose, not patching the model's template in memory when the flag is set: a text patch on a template that upstream or a new GGUF has changed fails without a sign, a file is something a future reader can open and diff.

**A probe at startup**, in `init()` next to `chat_params` (`server-context.cpp:1701-1740`), run when the mode is `note` or `reflect`. It renders, with no tools:

| Render | Cut at the markers | Gives |
|---|---|---|
| `[user U, assistant {reasoning R, content ""}, system Z]` + generation prompt | between `R` and `Z` | `pre_note_think` |
| same | after `Z` | `post_note_on` (and `post_note_off` from a render with thinking off) |
| `[user U, assistant {reasoning R, content C}, system Z]` + generation prompt | between `C` and `Z` | `pre_note_content` |

- The probe fails if the render throws, if `Z` is missing, or if `Z` is not after `R`. Then the server **refuses to start** with a message that names the template file and this plan. A guard that is on but silently drops its note is worse than no guard.
- The strings are printed at startup. For the Qwen family they must read `\n</think>\n\n<|im_end|>\n<|im_start|>system\n`, `<|im_end|>\n<|im_start|>system\n`, `<|im_end|>\n<|im_start|>assistant\n<think>\n`.
- Because the strings come from the template's own render of the message list a client will send back, the injected tokens and the next request's render agree by construction. The one place they can still differ is the seam (3.4).

To check in step 1: the auto parser reads the template to find the tool call and reasoning formats (`docs/autoparser.md`). Compare its startup log lines for the stock and the loop guard template.

### 3.4 Injecting the note (decisions 3, 5)

At a trip the reply is split. In the context and in the client's history alike:

```
assistant A   the reply so far, closed where it stands (reasoning only, or reasoning + partial content)
system        the note
assistant B   a new, complete turn: its own thinking, its own answer, its own tool calls
```

B always opens with the request's normal generation prompt (`post_note_on`, or `post_note_off` if the request has thinking off). The client knows about the split, so B can think again after a trip in `content`.

Injected tokens: `pre_note_think` or `pre_note_content` (by phase) + note + `post_note_*`. The framing is tokenized with special tokens parsed, the note text with special tokens **not** parsed, so a note cannot close or open anything.

**Seam-safe token.** The template trims reasoning and content, and the tokenizer may merge the last characters of A with the first character of the close (`.` + `\n`). Either would make the next request's tokens differ from the cache at the end of A, and on hybrid memory any difference is a rewind. So after a trip is recorded, the server acts at the first token where both hold:
- the generated text does not end in whitespace
- `tokenize(tail text + pre_note)` equals `tail tokens + tokenize(pre_note)`, for a tail of the last 16 tokens
If neither holds within 32 tokens it acts anyway and logs that the seam may not match.

Mechanics, at the top of `pre_decode()`:
1. new task = copy of the running task's params, same `id` and `oaicompat_cmpl_id`, `cache_prompt = true`, `tokens` = `slot.prompt.tokens.clone()` + `slot.sampled` + injected tokens
2. swap it into `slot.task`, set `slot.state = SLOT_STATE_STARTED`
3. the prompt path decodes the new tokens in one batch and goes back to sampling

The sampler: the lazy grammar has not triggered (the guard never acts in a tool call), and the reasoning-budget sampler is set to match B's opening (inside thinking after `post_note_on`, done after `post_note_off`). Penalty and DRY history are kept.

Request state at the split: `slot.generated_text` so far is stored as segment A and cleared, with `n_sent_text` and the newline tracking. `stats`, `n_gen` and the budget counters run on.

Guards in the prompt path for this hop:
- **the common prefix must be the whole slot** (`n_past == slot.prompt.n_tokens()`). If not, fail the request with an error. Never fall through to a checkpoint restore or a silent reprocess
- no `update_prompt_start()`, no second `is_begin` (`:3693`, `:4005-4012`)
- no `slot.stats.n_gen = 0` (`:4194`), no `init_sampler()` (`:4197`)
- no shrink of the KV cache (`maybe_shrink_for_request`, `:4817`)

After the hop the monitor starts fresh, and the trip count of the request goes up by one.

### 3.5 Reflection agent (step 3)

An internal completion task on the same slot, appended after the parked context:
- `tokens` = parked tokens + `tok_park` + `pre_note_*` + reflection prompt + `post_note_off`. The reflector sees the agent's reply closed and a system message asking it to reflect
- same task `id` and `oaicompat_cmpl_id`, so cancel (`:2868`) and `get_slot_by_cmpl_id` (`:1783`) still find the slot
- own sampler: low temperature, a small fixed grammar, `n_predict` 256, `t_max_predict_ms`, `repetition_stop` on (the reflector can loop too)
- its tokens go to the client as `reflect_delta` events (3.9), never through the chat parser
- the same append-only prompt path and the same prefix guard as 3.4

Reflection prompt: the evidence from the monitor (repeat rate, the most repeated line and its count, tokens spent in the phase, mean entropy), then the task: say whether the reply above is stuck, why, and write the note for the agent.

Output, as lines, so the stream reads well and the server can parse it:

```
LOOPING: yes|no
CAUSE: one line
NOTE: one paragraph, at most ~80 words, addressed to the agent: what to stop doing, what to commit to
```

The note the agent gets is a fixed header plus the reflector's text, for example: `LOOP GUARD: your last reply was cut off because it repeated itself. <NOTE> Do not repeat the steps above. Continue the task now.`

Where the generation ends (`post_decode()`, the two `send_final_response(slot); slot.release();` sites at `:4518-4523` and `:4655-4659`): if the slot is reflecting, call `end_reflection(slot)` instead. `release()` and the error paths call it too.

### 3.6 Parking in RAM (decision 1, step 3)

`begin_reflection()`:
1. capture the full state of the target and draft context and `slot.prompt.clone()` into `slot.parked_state` (the refactored body of `ram_store_save`). If the allocation fails, skip the reflection and inject the fixed note
2. move `slot.task`, `slot.smpl` and the per-request fields into `slot.parked`
3. remember `n_park = slot.prompt.n_tokens()` and `tok_park = slot.sampled`

`end_reflection()`, on **every** way out (done, failed, timed out, cancelled, decode error):
1. restore `slot.parked_state` (the refactored body of `ram_store_restore`)
2. check `slot.prompt.n_tokens() == n_park`
3. move the request state back
4. `LOOPING: yes` -> inject the note as in 3.4. `LOOPING: no` -> decode `tok_park` and go on with the same reply, no split. Unparsable -> the fixed note

Why this is a guarantee:
- the buffer is owned by the slot. No eviction rule, no count cap, no min-step logic, no client can drop it
- it is the full state, so it does not depend on what the memory type can rewind
- the agent's own context is byte for byte what it was. The only thing the reflection leaves behind is the appended note
- restoring on cancel too means the slot is never left holding context + reflection

Cost: the two copies from 2.4, and host RAM for one state while the reflection runs.

### 3.7 Modes and escalation

`--loop-guard MODE`: `off` (default), `log`, `note`, `reflect`. Request field `loop_guard` with the same values, for A/B runs.

- `log`: step 0. One `SLT_INF` line per trip. With `--loop-guard-trace DIR`, one TSV file per completion: token index, id, piece, phase, logprob, entropy, rate, line_repeats, tripped.
- `note`: fixed system note.
- `reflect`: 3.5 and 3.6. After `LOOPING: no` the next trip needs more evidence (`line_repeats` two higher, a fresh full window for the rate).
- At most 2 interventions per request (`loop_guard_max`). The next trip ends the request with `STOP_TYPE_REPETITION`.
- Room: the reflection needs about 500 cells past the parked context, the note about 200. `maybe_grow_for_request()` (`:4781`) runs for the internal tasks. If the room is not there, the request ends with `STOP_TYPE_REPETITION`.

### 3.8 Scope (decision 6)

- `note` and `reflect` apply to `/v1/chat/completions`, stream and non-stream. That is what the web UI uses.
- `/v1/responses`, `/v1/messages` and raw `/completion` get `log` only for now. Their event shapes would each need their own mapping of the split.

### 3.9 Client contract (decisions 8, 9)

**Stream.** Events ride on ordinary chunks as a top-level `loop_guard` object, next to `timings`. The `delta` of such a chunk is empty.

| Event | Fields | An aware client |
|---|---|---|
| `reflect_start` | `phase`, `signals` (rate, line_repeats, entropy means) | shows that the server is reflecting. The current assistant message stays open |
| `reflect_delta` | `text` | appends to the reflection display. It is **not** part of any message |
| `resume` | - | nothing changes. The deltas that follow continue the same assistant message |
| `note` | `message: {role: "system", content}` | 1. closes the current assistant message as it stands (A). 2. appends `message` to its history, unchanged. 3. starts a new assistant message (B) for the deltas that follow |

In `note` mode only the `note` event is sent.

**Non-stream.** `choices[0].message` is B, the last assistant message. A top-level `loop_guard` object carries `messages` = `[A, note]` (or `[A1, note1, A2, note2]`), to be placed before `message` in the history, plus `reflection` and `signals`.

**The next request** from an aware client holds `..., A, note, B` and renders through the loop guard template to the tokens the cache holds. The cache is reused in full, the same as after an ordinary turn.

**A client that ignores the events** still works:
- stream: it sees one assistant message, A's text followed by B's. Its next request does not match the cache from the end of A on, and the note is gone from its history. One reprocess of that reply, on the model where that is expensive
- non-stream: it holds only B. Its next request does not match from the start of A on

Server side of the split: the HTTP thread's parser state parses one message, so a `note` event resets it (`chat_msg`, `generated_text`, tool call ids, block flags) with B's generation prompt. In non-stream mode the final result carries the stored segments and each is parsed with a fresh state.

---

## 4. Steps

### Step 0 - stats, monitor, trace. Log only
- `server-context.cpp`: the stats helper, both call sites (3.1), `process_token()` feeds the monitor. `server-task.h`: two floats on `completion_token_output`.
- `server-loop-guard.h`: the struct from 3.2. `server_slot`: one member, reset in `reset()` (`:497`).
- `common/sampling.{h,cpp}`: `common_sampler_phase()`.
- `common/arg.cpp` + `common/common.h`: `--loop-guard`, `--loop-guard-trace`. `server-schema.cpp` + `server-task.{h,cpp}`: `loop_guard`, `loop_guard_rate`, `loop_guard_line_repeats`.
- **Measure:** tokens per second with the guard on and off, plain and MTP. Then normal agent work for a few days with the trace on.
- **Exit:** a rule with zero trips on healthy runs and a trip within ~600 tokens of the loop onset on every looping run, and a verdict on entropy from the traces.

### Step 1 - template, probe, `note` mode, events
- The template file (3.3), from the GGUF Kyle runs. The probe in `init()`, the startup log line, the refusal to start.
- `pre_decode()` top: seam check, build and swap the task (3.4). Prompt path: the guards. Setter for the reasoning-budget sampler state.
- `server-task.{h,cpp}`: the `loop_guard` event on the partial result, the parser state reset in `update()`, the segments on the final result, the JSON for both.
- **Measure** on the repro (section 6), image prompt included, driven by a script that follows 3.9: loop escaped (no trip in the next 512 tokens), answer right, seconds added. Then send the follow-up request and read the server log: **the cached prefix must be the whole previous context.** That one line is the test of the template, the seam rule and the contract together.
- **Exit:** full cache reuse on the follow-up, on the 35B with MTP.

### Step 2 - web UI
- `chat.service.ts` (`:626-665`): read `parsed.loop_guard`, new callbacks. Chat store: on `note`, close the streaming assistant message, add the system message (marked as a loop guard note for display), start a new assistant message. A collapsed "reflection" block for `reflect_delta`.
- The UI already makes several assistant messages in one turn for tool calls, so the store has the shape for it. Read that path before sizing this step.
- **Exit:** the same cache-reuse check as step 1, from the UI.

### Step 3 - `reflect`
- First, with no guard code: save and restore a RAM state on the 35B with MTP, mid-reply, through the existing `/slots` actions, and check the next token matches a run without the round trip. Time both copies.
- Refactor `ram_store_save` / `ram_store_restore` into capture / restore of a `server_prompt_cache_state`, used by the named store and by the slot.
- `server_slot`: `parked`, `parked_state`, a phase enum, all cleared in `reset()`. `begin_reflection()`, `end_reflection()`, the finish sites, `release()`. The `reflect_*` and `resume` events.
- **Measure** against `note` on the same repro: loop escaped, answer right, seconds added, and the reflector's verdicts on healthy text that was made to trip.
- **Exit:** make `reflect` the mode in daily use only if it clearly beats `note`.

---

## 5. Where this is documented

For a future reader who meets a system message in the middle of a conversation and wonders why:
- this file, sections 2.6, 3.3 and 3.9
- the comment at the top of the loop guard template
- the `--loop-guard` help text, which names the template
- the startup error when the probe fails

---

## 6. Test rig

- **Cheap repro, no GPU:** the throwaway CPU server (`CUDA_VISIBLE_DEVICES=""`, Qwen3.5-0.8B, hybrid memory, same template family, with its own loop guard template since the stock one raises). Send the sample's first ~600 tokens as an assistant thinking prefill (`continue_final_message` with `reasoning_content`). A model handed three rounds of a loop tends to write the fourth. Fixed seed. For a content loop, the same text as a content prefill.
- **Real repro:** the request that produced the sample (page 26 image, the 35B, the usual flags), saved outside this repo.
- **Healthy set:** long runs that did not loop, including one that reads a table, one that writes code, and one that drafts in thinking and copies the draft into the answer.
- **Per mode, per run:** tripped at token, loop escaped, answer correct, seconds added, tokens added, cached prefix of the follow-up request.
- No new file under `tests/`. The monitor header has no llama dependency, so a scratch program can replay a trace file through it.

---

## 7. Risks and open points

- **False positives** are the main risk: a long table, similar list rows, boilerplate code. `reflect` has a second opinion, `note` does not. Default is `off`, and step 0 comes first.
- **The seam.** Trim and token merging at the end of A (3.4). The rule should make it exact. The follow-up check in step 1 proves it or shows where it breaks.
- **The RAM round trip on the 35B is unproven** (2.4). Expert hot store, KV spill to host, q8 KV and the MTP draft context all sit under it.
- **The model restarts its answer** in B after a trip in `content`. A is then a broken partial answer that stays in the history (decision 5).
- **The note stays in the history** for the rest of the conversation, like any message the client holds. If old notes turn out to mislead later turns, the client can drop a note together with its A once a new user message arrives. Not planned.
- **Images.** The sample came from an image task. `slot.prompt.tokens` holds media placeholders and the internal tasks are built from a clone of it. Appends never re-encode, but test with an image in the prompt first.
- **The reflector sees the loop as the last thing in context** and may be pulled in. Grammar, `n_predict` and `repetition_stop` bound it. A failed reflection gives the fixed note.
- **Other templates.** Any template that passes the probe works. One that fails stops the server at startup when the guard is on.
- **Time.** Estimate: two state copies, ~250 prompt tokens and up to 256 generated for the reflection, ~200 prompt tokens for the note. About 15-30 s at this box's speeds.
- **Merge surface.** One getter and one setter in `common/sampling.cpp`. A new template file. The rest is in `tools/server`.

Open:
1. **Which 35B GGUF is in daily use?** The unsloth MTP template drops a late system message, the heretic one renders it. The loop guard template is cut from the one in use.
2. **Reflection in context or not.** As planned, the reflection is run on a parked copy and only the note is appended. The alternative needs no park and no restore: leave the reflect prompt and the reflector's answer in the context as two more messages the client holds. It costs ~400 tokens of context per intervention and puts the evidence text in front of the agent for good. Planned: park and restore, as decided.

---

## 8. Status, 2026-10-13

Built in one session, not committed. Tested with a separate CPU-only build (`build-lg`, `-DGGML_CUDA=OFF -DLLAMA_BUILD_UI=OFF`) and `unsloth/Qwen3.5-0.8B-GGUF` (hybrid memory). `build-cuda` was not relinked. **Nothing has run on the 35B yet.**

### 8.1 What exists

| Piece | Where |
|---|---|
| Flags `--loop-guard off|log|note|reflect`, `--loop-guard-trace DIR` | `common/arg.cpp`, `common_params_loop_guard` in `common/common.h` |
| Request fields `loop_guard`, `loop_guard_rate`, `loop_guard_line_repeats` | `tools/server/server-schema.cpp` |
| Monitor | `tools/server/server-loop-guard.h` |
| Phase of the reply, and the hook that shows the budget sampler injected tokens | `common_sampler_get_phase()`, `common_sampler_accept_prompt()` in `common/sampling.cpp` |
| Token stats, both sampling paths. API `logprobs` now filled under speculative decoding | `token_stats()` and the two call sites in `post_decode()`, `tools/server/server-context.cpp` |
| Template probe, refusal to start | `guard_probe_template()`, called from `init()` |
| Note injection, seam check | `guard_act()`, `guard_seam_ok()`, `guard_inject()` |
| Reflection, parking | `guard_begin_reflection()`, `guard_end_reflection()`, `guard_restore_parked()`, `slot_state_capture()` / `slot_state_restore()` (shared with the RAM slot store) |
| Events and non-stream fields | `send_guard_event()`, `server_task_result_cmpl_partial::loop_guard`, `server_task_result_cmpl_final::guard_segments`, `task_result_state::guard_split()` in `tools/server/server-task.{h,cpp}` |
| Loop guard template for the unsloth GGUF | `models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja` |
| Web UI | `onLoopGuard` in `chat.service.ts`, the split in `stores/agentic/index.svelte.ts` and `stores/chat/index.svelte.ts` |

### 8.2 Where the code differs from sections 1-7

- The reflection answers `CAUSE:` / `STUCK: yes|no` / `NOTE:` in that order, not `LOOPING:` first. With the verdict first the small model wrote `no` and then explained why the text was stuck. The cause first fixed that.
- The third phase is called `forced`, not `tool_call`. It also covers the budget sampler while it forces its end sequence, and a grammar that is not lazy (a JSON schema reply).
- The fixed note quotes the most repeated line when there is one.
- A reflection that says "not stuck" uses up one of the `n_max` interventions.
- In the UI the reflection shows as toasts ("reflecting...", and the text if no note follows). The note is a system message in the chat. A real reflection block in the transcript is not built.
- `prompt_n` in the timings stays the prompt the client sent. The tokens of the note and of the reflection are not counted in it. The reflection time shows up as a gap in the generation time.

### 8.3 Measured on the 0.8B (CPU, one slot)

- Guard off: no change in behavior, no events.
- `log`: the sample as a thinking prefill trips at `n_gen = 335` on the rate rule, the same token on the plain path and with `--spec-type ngram-mod`. 402 of 500 tokens came through the speculative path and every one has a logprob and an entropy.
- In the loop the mean entropy falls to about 0.001-0.01 nats. So on this model the "copies its own text" reading holds. The 35B may differ.
- Cost per token: not visible at 40-70 t/s on a 248k vocabulary. Not measured properly.
- Stock template of the 0.8B GGUF: the server refuses to start with `--loop-guard note`. With the loop guard template the probe prints `\n</think>\n\n<|im_end|>\n<|im_start|>system\n`, `<|im_end|>\n<|im_start|>system\n`, `<|im_end|>\n<|im_start|>assistant\n<think>\n`.
- **The follow-up request reuses the whole cache** (24-25 new tokens processed, 1,650-2,570 from cache) in every case tried: `note` and `reflect`, stream and non-stream, a trip in thinking and a trip in content, thinking off, two notes in one reply, a reflection that found no loop, and with speculative decoding.
- The seam check never had to give up.
- Parking: 1,727 tokens, 97 MiB, 18 ms to save and 17 ms to restore.
- A connection dropped in the middle of a reflection: the parked context is put back at release, and the same prompt sent again is served from cache.
- A third trip ends the request with stop type `repetition`.
- Web UI in dev mode: the reply splits, the note shows as its own bubble, and the next request of the tool loop processed 103 new tokens (two tool results) on a ~2,090 token context.

The 0.8B does not stop looping because of a note. That says nothing about the 35B, it only means the mechanics could be tested many times.

### 8.4 Owed

1. ~~Rebuild `build-cuda` and run on the 35B~~ done, section 9.
2. ~~The RAM round trip on the 35B~~ done, section 9.4.
3. ~~The heretic GGUF with its own template~~ it passes the probe, but needs its own loop guard template for the cache, section 9.4.
4. **A prompt with an image.** The sample came from one. Not run.
5. **Step 0's real purpose**: days of `--loop-guard log --loop-guard-trace DIR` on normal work, to set the thresholds and count false trips. The defaults (0.6, 4 lines) come from one transcript.
6. Does a note break the loop on the 35B, and does `reflect` beat `note`. That is the point of the whole feature and is not known.

### 8.5 Known limits

- The UI path that re-attaches to a running stream after a page reload (`stores/chat/streams.svelte.ts`) does not know the events. A reply that was split while the page was reloading comes back as one message.
- The UI's two-request prefill (thinking prefill and response prefill together) does not split in its first request.
- `/v1/responses`, `/v1/messages`, `/completion` and requests with `n > 1` are only watched.
- A request whose template did not pass the probe is only watched, also when it asks for `note` itself.
- Toolbag and other harnesses have to follow 3.9, or they pay one reprocess per intervention.
- Fixed 2026-10-26: a JSON schema (any non-lazy output-format grammar) with thinking on failed with HTTP 500 "peg-native" after a note with `think_after` off, because the grammar still thought it was inside the closed thinking block. `guard_inject()` now restarts the grammar with the new reply's generation prompt (`common_sampler_restart_grammar`). See `PLAN-loop-guard-json-schema.md`. With `think_after` on, a third trip in the new thinking block still ends the request with no answer (the normal stop after two notes).

### 8.6 How to rerun the tests

```
cmake -B build-lg -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-lg --config Release --target llama-server
build-lg/bin/Release/llama-server -m <Qwen3.5-0.8B gguf> --jinja -c 16384 -np 1 --port 8091 \
    --loop-guard reflect --loop-guard-trace <dir> \
    --chat-template-file models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja
```

Then send a chat request with the loop transcript as `reasoning_content` of a last assistant message, `continue_final_message: true`, `add_generation_prompt: false`. End the prefill at a full line: a cut inside a word makes a token seam that no tokenizer produces, and the follow-up then misses the cache for a reason that has nothing to do with the guard. `LLAMA_SERVER_SLOTS_DEBUG=1 LLAMA_SERVER_SLOTS_N_DIFF=14` prints the tokens around a cache mismatch.

---

## 9. Tests on Qwen3.6-35B-A3B, night of 2026-10-13

Model: `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF:Q4_K_M` with its mmproj, on the GPU, `build-cuda`, Kyle's flags: temp 0.6, top-p 0.95, top-k 20, DRY 0.8 / 1.75 / allowed length 2 / last 4096, `--rep-skip-prompt --rep-skip-tool-calls`, draft-mtp n-max 1, `--reasoning-preserve`. About 90 runs. The rig, the runs and the per-token traces are in `I:\Kyle_Shepherd\toolbag\tests\loop_guard` (README there).

### 9.1 What makes this model loop

- **DRY with allowed length 2 is the main cause.** The model cannot write a number, a formula or a name a second time: DRY punishes every repeat longer than two tokens. It writes `7-17-9` where it read `7-17-16`, `x^x - x x = 4` for `x^2 - 5y^2 = 4`, a 12-digit number with digits swapped. Then it doubts its own reading and goes over it again. With DRY on, 14 of the 16 tasks ran to the 5000-token cap without an answer, including the clear version of the image and the "healthy" math, code and email tasks. With DRY off the image task finished in 1300-1500 tokens on both seeds.
- **DRY also breaks the cache across turns.** To get around the penalty the sampler picks rare token splits of the same text (`-` + `"?` instead of `-"` + `?`). The next request renders the text again, it tokenizes the normal way, and the cache misses from there. Measured on two of the runs. This has nothing to do with the guard and costs a reprocess on any turn.
- What loops with DRY off: the impossible puzzle, the self-counting sentence, and re-drafting (the model writes its whole answer in the thinking block, then writes it again).
- The DRY loops are paraphrases, not copies: 8-gram repeat rate 0.05-0.15, the line rule never fires.

### 9.2 Trigger v2

The 8-gram rule from section 3.2 was wrong both ways on this model: it fired on healthy structured work (truth tables, prime lists, partial products, code drafts) as early as on loops, and it never fired on a DRY loop. Section 1 had only one transcript, which was a copy loop.

What separates them is progress. A table, a list or a calculation keeps bringing words it has not written before (new numbers, row patterns, names). A loop goes over the same things. So:

- **word novelty**: the share of the last 300 words (lowercase, punctuation at the ends cut off, digits kept) that were never written before in this phase. Trip at 6% or less, after 600 words. Most healthy runs stayed at 13-45%. The DRY loops went to 2-6%.
- **exact lines**: one line of at least 12 tokens written 5 times (catches copy loops like the one of section 1, where novelty also falls below 6% at about 500 words).
- the 8-gram rate rule stays in the code, off by default (`loop_guard_rate`).

Novelty also falls when the model writes a draft it already wrote. On this model that is close to a loop anyway (it never got to the answer within 5000 tokens on those tasks), and in `reflect` mode the reflection judges it.

### 9.3 Interventions

| Change | Why |
|---|---|
| The reply after a note answers without a thinking block (`loop_guard_think_after`, default off) | A new thinking block started over ("let's look at Sample 5 again") and walked back into the loop every time: 0 of 5 escapes in the first round |
| The reflection grammar allows one sentence of cause and at most three sentences of note, 250 characters each | Free text let the reflection loop over the same digits, and its note carried the confusion back to the agent |
| The reflection prompt says it is not the assistant, must not work on the task, and must tell it to say so when the task cannot be done | It solved the puzzle itself in the note; the agent then forced an answer to an impossible puzzle |
| The note tells the agent to write the final answer now and mark unsure values with `(?)` | "Continue the task" was read as "keep checking" |
| The reflection samples without DRY and penalties | It has to quote values |
| After a note the penalties and DRY forget the reply so far (`loop_guard_rep_after` 1). Turning them off for the rest of the request (2) did not help on its own | The next reply has to state values the old one stated |
| A copy of the old note's last sentence in a new note is cut off | After a second trip the reflection copied it |
| Logprob and entropy are computed only with `--loop-guard-trace` | They cost about 5% of a step (36.8 -> 34.7 t/s) and the trigger does not use them |

Results with DRY on, after these changes:

| Task | Mode | Runs | Escaped the loop and answered | Notes per run |
|---|---|---|---|---|
| blurry boring log table | reflect | 5 | 5 | 1 |
| blurry boring log table | note | 3 | 3 | 1 |
| impossible seating puzzle | reflect | 4 | 4, all four say the clues contradict | 1-2 |
| count letters in a string | reflect | 1 | 1 | 1 |
| 12-digit multiplication | reflect | 2 | 1 escaped, 1 finished without a trip | 1 / 0 |
| four math problems | reflect | 2 | 1 answered after a note, 1 no trip (DRY garbled the arithmetic, but it kept trying new values: novelty 13-45%) | 1 / 0 |

Before these changes (first round, same settings otherwise): 0 of 5.

Healthy tasks with DRY off and `reflect`: math, essay and the CSV table did not trip. Code, the truth table and the email draft tripped while re-drafting, the reflection said stuck, and each then wrote a complete answer far sooner than without the guard. The prime list tripped once and the reflection said "not stuck", so the reply went on unchanged.

### 9.4 Cost and cache

- Parking on the 35B with MTP: 180-260 MiB in 72-110 ms, restore 46-72 ms (2.5k-6.3k tokens). Reflection: 126 tokens on average. A whole reflect step is about 5-8 s.
- The follow-up request reuses the whole cache (22-27 new tokens) when the client holds the split messages, in `note` and `reflect`, with one and two notes, after a trip in thinking and in the answer. The exceptions were the DRY token splits of 9.1.
- **The heretic template needs its own loop guard template**: `models/templates/Qwen3.6-35B-A3B-heretic-loop-guard.jinja`. The source renders an assistant message with empty reasoning without the empty thinking block that the generation prompt wrote. That missed the cache after every answer-only reply of the guard, and after every normal turn with thinking off. The unsloth template renders the block already.

### 9.5 Not covered

- The "word salad" of `I:\Kyle_Shepherd\Looping Example.txt` (presence penalty 1.5, repeat penalty 1.1: no repeats at all, no full stops). Not reproduced with Kyle's current flags. A "no sentence end in 200 tokens" rule would catch it; not built.
- A model that keeps trying new things without converging (the math run of 9.3). That is the reasoning budget's job.

### 9.6 Recommended launch

```
--loop-guard reflect --chat-template-file models/templates/Qwen3.6-35B-A3B-heretic-loop-guard.jinja
```

(or `Qwen3.6-35B-A3B-unsloth-loop-guard.jinja` for the unsloth GGUF). And reconsider DRY allowed length 2: it causes most of the loops the guard then has to break, and it costs cache hits on every turn.

---

## 10. Tests with plain sampling, 2026-10-14

Same model and build. Sampling as Qwen recommends, with every repetition control off: temp 1.0, top-p 0.95, top-k 20, min-p 0, presence 0, repeat 1.0, DRY 0. The idea: no penalty to fight loops, and the guard breaks the ones that happen. 36 baseline runs (guard watching) and 26 intervention runs. Data: `toolbag/tests/loop_guard/data/q_base`, `q_iv`, `q_final`; `srv35.sh` takes `SAMPLING=qwen` (default) or `SAMPLING=kyle`.

### 10.1 What loops without any penalty

The blurry table of section 9 no longer loops (finishes in 2200-2900 tokens). New tasks that do, on both seeds, to the 6000-token cap: a worse scan of the same page, counting 73 dots in an image, counting "the" in a paragraph (53), the impossible seating puzzle, a knights-and-knaves puzzle that has no solution, a 12-digit multiplication. The self-counting sentence runs long but ends on its own within 9000 tokens. A liar-paradox question ends on its own (3000-4400 tokens).

### 10.2 Trigger

The thresholds of section 9.2 hold without change. Replayed over the 36 baseline traces:

- trips on every looping task, at 1000-3700 tokens
- no trip: math, letter enumeration, CSV, essay, the blurry table that finished
- trips while the model rewrites a draft in its thinking: code, email draft, truth table, primes, and the clear image (it re-read a table it had right). With `reflect` the reflection judges these; on this model they did not reach an answer within 6000 tokens either

### 10.3 Interventions (`reflect`, think_after off)

| Task | Runs | Escaped and answered | Answer |
|---|---|---|---|
| worse scan | 2 + 1 note | 3 | a table, unsure cells marked |
| 73 dots | 2 + 1 note | 3 | 38, 73, 55 |
| count "the" (53) | 2 + 1 note | 3 | 50, 52, - |
| impossible seating puzzle | 2 | 2 | both say the clues contradict |
| knights, no solution | 2 + 1 note | 2 (the note run looped again in its answer and was stopped) | all named an assignment |
| 12-digit multiplication | 2 | 2 answered after 1-2 notes, kept working to the cap | wrong |
| re-drafting tasks | 4 | 4, each answered after one note | complete answers |

The guard does what it is for: it breaks the loop and gets an answer out. Whether the answer is right is up to the model, and on these tasks it often is not.

### 10.4 Notes, after this round

- The reflection may not give a value of its own (it made up `9-13-17` for `9-12-14`, and a 24-digit product). It may tell the agent to keep a result the agent itself reached more than once, or to mark it unclear. It still does it now and then ("mark D as Knave").
- Any copy of a footer sentence in the reflection's note is cut off (it copied from its own prompt too, not only from earlier notes).
- The fixed note quotes the most repeated line only when it came 3 times or more.

### 10.5 Cache

The follow-up request reused the whole cache in 20 of 22 runs. The two misses: at temp 1.0 the model sometimes samples a rare token split of plain text (`"` + `?).`, ` (` + `Sil` + `ty`); the client's history tokenizes the usual way, and the cache misses from there. This is not the guard, any multi-turn chat at this temperature can hit it. Fixed since: the server takes the cached tokens where the prompt spells the same text with other tokens (`server_tokens::align_to`, commit `919fe6ff5`).

### 10.6 Recommendation

With plain sampling the model loops less, and when it does the guard gets it out. `--loop-guard reflect` with the loop guard template of the GGUF in use. `note` works about as well for escaping, without the reflection's 5-8 s, but it cannot say "not stuck".

## 11. The unsloth GGUF, 2026-10-15

`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-Q4_K_S with `models/templates/Qwen3.6-35B-A3B-unsloth-loop-guard.jinja` (renamed from `Qwen3.6-35B-A3B-loop-guard.jinja`, so the file names the GGUF it belongs to), plain sampling, `reflect`, rig `srv35.sh` with `MODEL_NAME=unsloth`. The template probe passes, and the unsloth source already renders the empty thinking block, so the heretic fix is not needed here.

8 runs (img_smudge, img_dots, contradiction, img_table, 2 seeds each): 4 tripped, reflected and ended with an answer; the follow-up request of all 8 reused the cache (prompt_n 22-23). The reflections were on point ("stop examining the column headers", "you reached 70 twice, give it").

## 12. Qwen3.5-4B, 2026-10-20

`unsloth/Qwen3.5-4B-MTP-GGUF` Q4_K_S on the GPU, plain sampling, MTP, `reflect`. Its template is the unsloth Qwen3.6 one except for one line: earlier reasoning is shown only after the last user message, `preserve_thinking` is ignored.

- With a loop guard template cut from it as it is: all 12 runs tripped (15 reflections, on point), but every follow-up missed the cache from the start of the reply (prompt_n 275-3048), because the new user message makes the template drop the reply's reasoning. That is the stock behavior on every turn, guard or not.
- With `Qwen3.6-35B-A3B-unsloth-loop-guard.jinja`, which keeps the reasoning under `--reasoning-preserve`: 8 runs, 6 ended with an answer (2 at the 6000-token cap, one after two notes), all 8 follow-ups reused the cache (prompt_n 22-24). So the unsloth Qwen3.6 loop guard template is the one for unsloth's Qwen3.5 GGUFs too; no separate file.
- The 4B escapes less often than the 35B: after the note it sometimes loops again in the answer (a second trip in content in several runs).

## 13. LFM3-VL-2B, task-neutral texts, 2026-10-28

`LiquidAI/LFM3-VL-2B-GGUF` QAD-Q4_0, branch `lfm3`, `models/templates/LFM3-VL-2B-loop-guard.jinja`, LiquidAI sampling (temp 0.1, top-k 50, repeat penalty 1.1).

- Server fixes this model needed: the hybrid SWA cache reuse on an append (`d048cadf2`; before it every hop failed with "not an append"), and the reflection grammar (`f56153b9f`): LFM3 ended sentences with U+202F or no space, CAUSE took the STUCK and NOTE lines, and every reflection fell back to the fixed note.
- "Write 10,000 words of your choice" gave "I'm sorry, but I can't ...". The refusal is the model's own: guard off it refused 1 of 3, thought for all 16000 tokens 1 of 3; thinking off it refused 3 of 3 at once. The guard cut a real circle (2000-3300 thinking tokens on "is it feasible", 6% new words).
- The note, footer and reflection prompt spoke of a final answer, of what was worked out, of the image and the data. Now task-neutral: "Now write what the user asked for, in full: the answer, the text or the code." The "write now" order stays (9.3: "continue the task" was read as "keep checking"). The reflection's NOTE must start with "You": the 2B wrote it in the first person ("I will stop ...").
- Old vs new texts on the same binary, 2 seeds: boring log tables the same; count_chars a wash (old: a fast wrong count, new: still counting at the cap; true count 77); essay, story, email draft all answered under both, healthy essays and drafts tripped in thinking under both (LFM3 overthinks). Reflect, 10,000 words: old 0 of 3 wrote at once, new 1 of 3 (1 apologized then wrote, 1 refused).
- Not checked on the 35B.

### 13.1 Reasoning preserve, and the reflector writes the plan (2026-10-28, later)

- `--reasoning-preserve` does not change what the model sees after a note: the hop appends to the cached tokens (no render), so the looping reasoning is in context either way. Same seeds, flag on and off: the first segment identical, the reply identical where the note was. The runs that differed had different reflections, because the reflection sampled with a random seed. It now takes the request's seed.
- The flag matters on the next turn: off, LFM3's template drops the earlier reasoning, the follow-up re-processes the whole reply (cache_n 1) and sees only its own apology. "Please go ahead and write it.": 2 of 4 wrote with the flag on, 1 of 4 off.
- Texts: no more (?) (no model was seen to use it). The fixed note asks to write what the user asked for "from what you already have" and adds "Do not refuse it: do as much of it as you can". The reflect note is the header, the reflector's own note, and only the footer "Do not refuse what the user asked for: do as much of it as you can."
- The 2B reflector copies the wording of its own instruction into the note, and the model then does that instead of the task. Tried on img_table, img_smudge, essay, draft_copy (2 seeds) and the 10,000 words prompt (3 seeds):
  - "how to go on from here: what to keep, what to stop, what to write next": essay s1 answered with a "How to proceed" list.
  - "Name the next concrete thing ... for example the next section, the table, or the result": essays fine; 10,000 words s3 wrote a table instead of the essay.
  - "Name the next concrete thing" alone, "at most three short sentences": the note invented a sentence to copy ("I will now finish the report and send it by [date]." became the whole email), essay s1 answered in "three sentences".
  - Kept: "a short note to the assistant, starting with You. Name the next concrete thing it should write for the user, for example the next part of the answer, or the result it already found." The grammar still limits it to three sentences, so the prompt does not say so. All 8 rig tasks answered; 10,000 words refused 3 of 3 (the model's own belief).
- For a 2B, `note` gives the same outcomes as `reflect` without the reflector's copy risk. Not re-checked on the 35B.
