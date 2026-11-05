# Plan: hydra phase 2 - reliable and fast boring log and lab result transcription

Rev 1, 2026-10-29. Hand-off for an implementation agent. Read `AGENTS.md`, then `PLAN-hydra.md` (sections 3.3, 7 and 9: what the hydra is and what is built). ASCII only, short comments in simple English, no new files under `tests/` of opt-llama, test tools go in `I:\Kyle_Shepherd\toolbag\tests\hydra\`. **Stop and ask Kyle at each gate.** Kyle owns the design.

## 1. Goal

geocrawler's transcription of two document kinds with LFM3-VL-2B as a hydra:

- **boring logs** (`geoui/templates/boring_log.md`): header fields + Groundwater table + Samples table (sample, from_ft, to_ft, type, blows 0-6/6-12/12-18/18-24, N, RQD, rec, description, page);
- **lab results** (`geoui/templates/lab_results.md`): header + one Results row per printed value (`Sample | Depth (ft) | Table | Result | At | Value | Page`).

"Reliable" = cell-exact against a ground truth, no invented or dropped rows, `checks.py` clean. "Fast" = wall time per page well below today's single-pass call. Find the request layout (stages, what goes in the root, what each head does) that gets there, and make geoui use it.

## 2. What exists (do not rebuild)

All in worktree `I:\Kyle_Shepherd\opt-llama-hydra` (branch `hydra`, uncommitted - ask Kyle to commit before you change the server), build `build-cuda` (UI built in, `-DLLAMA_BUILD_UI=ON`, Node 22 on PATH).

- **Hydra request**: `POST /v1/chat/completions` with `"hydra": {"heads": [{"messages": [...tail...], ...overrides}]}`; `messages` is the root. A head overrides any field (`max_tokens`, `response_format`, `temperature`, `chat_template_kwargs`). Choice i = head i, stream and non-stream. `/props` has `hydra_max_heads`. The root is read once (one image encode per root image), each head forks from it (`seq_cp`, unified KV, shares the root's cells) and prefills only its tail; all heads decode in one batch. A later hydra with the same root reuses the root slot (no image encode, no prefill). Images only in the root; tails are text.
- **Budget**: each head's `max_tokens` is capped at its share of the KV pool (`--ctx-max` is the pool size when set); the pool grows on demand with `--ctx-max` on `-kvu` (also on LFM3 now).
- **Loop guard** per head (`--loop-guard reflect|note`), light park, notes show in the UI columns.
- **Speed facts** (laptop RTX 3060 6 GB, LFM3 QAD-Q4_0): 1 head ~155 t/s; 16 heads ~18.5 ms/step = ~860 t/s total; with `-bs` (backend sampling) 12.7 ms/step = ~1260 t/s. `-bs` does not apply to grammar sampling (`response_format`), which runs on the CPU per head: measure its cost (section 5). Prefill ~6400 t/s, page image encode 0.26-0.39 s, ~1500-2000 tokens per page image. `-np 17` costs only 44 MiB more VRAM than `-np 5`.
- **Greedy drift**: LFM3 output changes with batch shape (logprob drift ~0.02-0.04 nats). Compare against single requests with logprobs, not text alone; the GGUF default `repeat_penalty` 1.1 survives `temperature: 0`, send `repeat_penalty: 1.0` in A/B runs.
- **Harness**: `toolbag/tests/hydra/` (`srv.sh`, `hydra_check.py`, `greedy_check.py`, `budget_check.py`, `loop_check.py`, README).

geocrawler (`W:\Kyle_Shepherd\geocrawler\geoui`, read it, do not break it):

- `geoui/jobs.py` `build_transcribe_messages` (~656): system = `prompts/transcribe.md` + template ABOUT / RULES / LESSONS / reply format; up to 3 example pages (user: image + "Transcribe this example page." / assistant: JSON); then the target page(s) (image, optional engineer note, optional text layer of born-digital pages, `text_chars` 6000). This whole context is the natural root.
- `geoui/llm.py` `chat()` (streams, `response_format` aware), `sampling()`; `geoui/config.py` `transcribe`: px 1800, thinking on, temperature 0.3, max_tokens 48000, reasoning_budget 6000, example_px 1400.
- `geoui/checks.py`: `_boring` (depth order, N vs blows, recovery vs drive, ...), `_lab`, `_text_layer` (values vs the PDF text layer).
- Ground truth with exact answers: `templates/boring_log/01..04-*.png|.md` (4 boring log pages) and `templates/lab_results/01..03-*.png|.md` (summary table, sieve + hydrometer, consolidation). Leave the page under test out of the root's examples (leave-one-out). Synthetic boring logs (clear / blur / smudge) in `toolbag/tests/loop_guard/tasks.py` `boring_log_image`. More pages: `Micron-Draft-EIS-Appendix_E_Vol_2.pdf` (ask Kyle for reviewed transcriptions; geoui's Review tab on the work computer is the source of truth).
- `geoagent-client-v2/CRAWLER.md` "V2: one row per call": a 2B model miscounts cells in 12-column rows; named fields per row fix it. Hydra heads with a one-row JSON schema are the same idea, in parallel.

## 3. Layouts to compare

Same root for all (system + examples + page). Thinking off unless a step says otherwise (LFM3 overthinks; it read the table right with thinking off in the LFM3 evaluation). Each head gets a `response_format` JSON schema, so the grammar fixes the shape.

| Id | Stage 1 | Stage 2 |
|---|---|---|
| A | single request, today's geoui prompt (baseline, thinking off and on) | - |
| B | 1 head: skeleton = header + list of rows (sample id, from, to; for lab: sample, table, result names) | N heads on the kept root: head 0 = header + groundwater, head k = row k (or a group of rows), the skeleton in each tail |
| C | as B, but stage 1 itself is a hydra: header head + skeleton head (+ a second skeleton head with another wording, rows kept where they agree) | as B |
| D | no skeleton: one head per column group (depths, blows, descriptions) over all rows | merge by row index |
| E | as B with K votes per row (K heads per row, temperature > 0, majority per cell; ties -> greedy head) | |

Lab pages: the skeleton is the list of (Sample, Depth, Table, Result) keys; stage 2 heads read the Value (and At) for a group of keys, e.g. one head per table column or per sample. Start with the summary-table example (01), then sieve (02) and consolidation (03), which are two-way tables.

Template note: check that a second user turn after the root reads as well as one merged turn (PLAN-hydra 3.4). If a merged turn reads better, add a hydra template file next to `models/templates/LFM3-VL-2B-loop-guard.jinja` (root ends with an open user turn, the tail closes it) with a header comment; the server's prefix check must still pass.

## 4. Steps and gates

1. **Ground truth and scorer** (toolbag): load the 7 example pages + their `.md` truth with geoui's own template code (import `geoui.templates`, do not copy the parser); score per cell (exact after whitespace normalising, numbers as numbers), rows missed / invented, `checks.py` findings, wall time, tokens. Run A on all 7 pages (leave-one-out examples), thinking off and on, 3 seeds. **Gate 1**: baseline table; Kyle confirms the truth set and the metric.
2. **Layout B on boring logs**: client script in toolbag that builds the root exactly as `build_transcribe_messages` does (call it, do not copy it), runs stage 1, then stage 2 on the same root, merges, scores. Try row groups of 1, 2, 4 rows per head. Log per stage: root prefill time, stage times, tokens, reuse of the root (stage 2 root prompt_n near 0). **Gate 2**: B vs A, accuracy and time; Kyle confirms the "2-stage swarm" reading (PLAN-hydra 7.2 was never confirmed).
3. **Lab results**: B with lab keys; then C/D/E only where B leaves errors. **Gate 3**.
4. **Speed pass**: grammar sampling cost per head (CPU) vs `-bs`; head count per page (16 vs 8 vs rows); `--ctx-max` sizing; whether the examples in the root pay for themselves (try 0, 1, 3). Measure ABBA (the laptop slows as it heats). **Gate 4**.
5. **geoui integration**: a hydra transcription path next to `build_transcribe_messages` (setting `transcribe.mode`: `single` | `hydra`), using `llm.chat` with the `hydra` body, the merge, and `checks.py`; keep the single path as default until Kyle switches. Thinking per stage from config. Kyle's server list in `config.py` has `slots`; a hydra server needs `-np` >= heads + 1: read `hydra_max_heads` from `/props` and fall back to `single` when it is too small. **Gate 5**: end-to-end on a few Micron PDF pages in geoui, Kyle reviews.
6. Write results into `PLAN-hydra.md` section 9 (new 9.5) and a table per run under `toolbag/tests/hydra/`.

## 5. Things that will bite

- **Row identity**: heads must agree on which row is which. Key rows by (sample id, from depth) from the skeleton, not by position; a head returns its key with its values, and the merge rejects a mismatch.
- **Skeleton errors** propagate to every row: measure skeleton accuracy on its own (missed / invented rows) before blaming stage 2.
- **Grammar on the CPU**: 16 grammar samplers per step may cost more than the decode (~1.2 ms/step/head without grammar was the CPU sampling cost). If so, consider fewer, wider heads, or a lighter schema.
- **Descriptions** are long free text and wrap across lines on the page; a head per description may be worth it, or descriptions may need the text layer.
- **Two-page items** (a boring over 2 sheets): the root holds both page images; rows carry `page`.
- **Server side**: images only in the root; heads cannot add images. The root slot is reused by prefix match, so stage 2 must send the stage 1 root byte for byte (same image size, same text).
- **Loop guard**: run with `--loop-guard note` or `reflect`; a looping head on a JSON grammar restarts its grammar after a note (server handles it), but check the merged JSON.
- **Server launch** for the tests: `toolbag/tests/hydra/srv.sh start -np 17 -kvu -c 8192 --ctx-max 131072` (plus `--loop-guard ...`); Kyle may run his own server on 8080, use 8091.
