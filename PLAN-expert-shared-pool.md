# Testing one hot-store slot pool shared across layers

Written 2026-10-28 for an agent to carry out (reworked the same day: build and measure directly, one model, no gate). Background: PLAN-moe-cache-merge.md section 7.1. Base: branch `expert-prefetch-adaptive-ub` at 03ac05aa0.

## 0. The question

Our hot store gives every MoE layer the same `hot_s` slots. Heat is not spread evenly: after 128 tokens layer 0 had 225 of 256 experts warm and layer 9 only 126 (PLAN-qwen36-35b-moe.md 15.3). Build a mode where the slots form a pool shared across layers and go to the hottest (layer, expert) pairs anywhere in the model, at the same VRAM. Then measure it against today's equal split on the unsloth Qwen3.6-35B-A3B UD-Q4_K_S, and report. Kyle decides from the numbers. There is no pass/fail threshold, so report what you measure, including a null or negative result.

## 1. Ground rules

- Read AGENTS.md first. ASCII only in code, comments and docs. Concise comments.
- You have the laptop GPU to yourself: Kyle's server is not running. You can build in `build-cuda` and start or stop llama-server instances as needed. Before every measurement, check no other llama-server is running (`tasklist | grep llama-server`). A stray one spoils the numbers.
- Work on a new branch made from the current one in the main folder: `git switch -c expert-shared-pool`. The untracked PLAN files come along. Commit there as you go. Never push and never open a PR. At the end, leave the folder on that branch and say so in the report.
- Scripts and results go in `I:\Kyle_Shepherd\toolbag\tests\expert_pool\`. Do not add files under opt-llama's `tests/`.
- Bash commands are capped at about 7.7 KB. Write scripts with the Write tool and run them by path. PowerShell script execution is disabled; run commands inline.
- Measure only sustained runs (600+ generated tokens). The store warms up slowly, so short runs mislead (PLAN-qwen36-35b-moe.md 15.2).

## 2. Build

### 2.1 Read first

- `src/llama-expert-hotstore.{h,cpp}`: `allocate`, `copy_top_s`, `resync_top_s` (stable slots, empty slots filled first, hysteresis 1.3, dwell 0), `maybe_resync` (every 50 tokens of `tokens_total`), `update_luts`, `slot_of`, `count_cold`, `log_hit_rate`, `resize`, `slots_that_fit`, `plant_static`.
- `src/llama-expert-tier.cpp`: `llama_expert_tier_register` keys each model expert tensor to `{dst_hot, hot_lut, cold_mask}`. The graph gathers `hot_lut[e]` (the expert's slot, 0 when cold) per draw, runs `ggml_mul_mat_id(dst_hot, cur, ids_hot)` on the GPU, and runs `ggml_mul_mat_id_cold` on the CPU for the cold draws, masked. A cold draw lands on a real slot and the mask removes its contribution. Duplicate ids keep the tier at 4 tokens or fewer.
- `src/llama-context.cpp`: construction around 650-695. The per-batch heat update, resync and `LLAMA_EXPERT_HITRATE` are around 2455-2490. The KV auto-grow resize is around 2880. `set_n_ubatch` lends store VRAM to the large ubatch and to the MTP draft context (`ctx_other`).
- `common/fit.cpp` around 891-905: the autosize plans a per-layer `hot_s`.

### 2.2 Layout groups

`ggml_mul_mat_id` needs one tensor of one type and shape, so the pool is one bank per **layout group**: the layers whose gate, up and down tensors have the same types and shapes. UD-Q4_K_S mixes quant types across layers. List them with gguf-py (`python -I gguf-py/gguf/scripts/gguf_dump.py <model>`, or the equivalent in `gguf-py`), and write the groups into the report: layers, types, bytes per slot.

Each group gets `hot_s x n_layers_in_group` slots, so every group holds exactly the VRAM it holds today. Experts move between the layers inside a group, never between groups. Moving VRAM between groups would mean resizing banks at run time; leave it out. If the model turns out to have many small groups (say more than 4, or most groups with 1-3 layers), note in the report how much of the model's experts that limits to sharing. Then carry on.

### 2.3 Changes

Add `--expert-pool layer|shared` (env `LLAMA_ARG_EXPERT_POOL`), default `layer`. `layer` must behave exactly as today.

In `shared` mode:

- **Tensors.** One bank tensor per group and expert tensor (gate, up, down, and any chexps variants), shape `{ne0, ne1, n_slots_group}`, all in the one store buffer as today. `entry::dst` for every layer of a group points at that group's bank, and the tier registration does the same. `hot_lut[il][e]` holds the bank slot. The LUT is f32, which stays exact for any bank size here.
- **Ownership.** Replace the per-layer `slot_to_expert[il][p]` with a per-bank owner table `(layer, expert)` or empty. `slot_of(il, e)` should become a lookup, an `[n_layers][n_experts]` int array, rather than a scan of a bank of hundreds of slots.
- **Fill and resync.** Same heatmap, same cadence, same hysteresis, but the ranking is global within the group: the bank holds that group's top `n_slots_group` (layer, expert) pairs by heat. Each layer gets 8 draws per token, so heat is comparable across layers. Keep slots stable: an unchanged pair is never re-copied. A cold pair displaces the coldest resident only if its heat is at least `hyst` times that resident's, with the same dwell rule.
- **Resize** (KV growth, `-ubg` lending): the bank shrinks or grows to `new_hot_s x n_layers_in_group` and re-plants the top pairs from the current heat. Keep `slots_that_fit` and `bytes_resident` in the same units (per-layer `hot_s`), so the fit and the lending code need no changes.
- **A layer can end up with 0 hot experts.** The hot `mul_mat_id` then runs on all-cold ids that are fully masked. That is correct but wasted work. Leave it unless it shows up in the profile.
- **Log.** At each resync with `LLAMA_EXPERT_DEBUG`, and once at teardown always, print one line per group giving the hot experts per layer (for example `layers 0-39: 12 9 31 ...`). Kyle will want to see where the slots went.
- `count_cold`, `log_hit_rate` and `plant_static` must work in both modes. `plant_static` in shared mode can just give each layer `hot_s` consecutive slots.

### 2.4 Correctness checks before any speed run

- Run `llama-perplexity` on a fixed text, so every ubatch takes the tier: `-c 512 -b 512 -ub 4 --chunks 8`, with `-ehs N -fitt 256` on the unsloth model. Use wikitext under HF_HOME if present, otherwise any long plain-text file, and name it in the report. Compare `-ehs 0`, `--expert-pool layer` and `--expert-pool shared`; record all three. First check that perplexity accepts `-ub 4` with the store on (the tier only runs at 4 tokens or fewer). If it doesn't, find another way to force small ubatches and say what you used. Exact token match at temp 0 is not a valid test: the hot/cold split changes summation order, so runs drift apart.
- `--expert-pool layer` on the new build vs the old code: same slot count at load, same perplexity. Before changing any code, build `build-cuda` once on the base commit and record its slot count and the perplexity run above as the baseline.
- Both modes must survive a long prompt that triggers the `-ubg` switch and a resize, with no crash and no CUDA error in the log.

## 3. Measure

Model: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (the snapshot under `I:/Kyle_Shepherd/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/`).

Server: copy `I:\Kyle_Shepherd\toolbag\tests\loop_guard\srv35.sh` to `toolbag/tests/expert_pool/srv_pool.sh`, with `MODEL_NAME=unsloth` fixed and `SAMPLING=kyle`. Drop the `--loop-guard*` flags if they aren't needed. Add `-b 4096 -ub 4096 -ubg 512 -epf 2` (Kyle's prefill config since 2026-10-24), `--expert-pool $POOL`, and `LLAMA_EXPERT_HITRATE=1` in the environment. Keep the rest as it is: MTP n_max 1, `-ehs -1 -fitt 256`, FA on, `-np 1`.

Workload: the healthy long tasks from `toolbag/tests/loop_guard/tasks.py`, driven with `run.py` (see that folder's README), each producing 600+ generated tokens. Use the same tasks and seeds for both modes.

Runs:

- **Generation:** at least 3 runs per mode, alternating `layer`, `shared`, `layer`, ..., with a fresh server for each run. Per run, record:
  - generation tok/s;
  - the hit rate and distinct-cold figures from the `LLAMA_EXPERT_HITRATE` lines, averaged after the first 200 tokens;
  - the slot count and store MiB at load;
  - the teardown per-layer slot line (shared mode).
- **Smaller store:** repeat with a fixed `-ehs` at about half of what autofit picks. The home server's store is smaller, so this shows whether the gain grows or shrinks as VRAM gets tighter.
- **Prompt speed:** one 8k-token prompt per mode, prompt tok/s. It should not change. If it does, find out why (resize churn or lending).
- **Profile, only if the results are surprising:** for example a higher hit rate but no speedup, or a slowdown. Then take one nsys capture per mode of a decode stretch, and compare the hot `mul_mat_id` kernel time (the bank's larger expert count could make id compaction slower) and the CPU cold op time per layer. Moving slots to deep layers could leave one layer's CPU work as the long pole.

## 4. Report

Write `toolbag/tests/expert_pool/RESULTS.md`:

- The layout groups (2.2).
- The correctness results (2.4).
- A table per budget (auto, half): mode, runs, generation tok/s (mean, min, max), hit rate, distinct cold per token, store MiB.
- The prompt speed result.
- Where the slots went: the per-layer counts from shared mode, as a table or a small chart.
- The nsys findings, if you took them.
- Anything that broke or looked wrong.
- The branch name and the commits.

Add a short section 10 to PLAN-moe-cache-merge.md that points to it. Then stop and hand the results to Kyle. Do not change the default of `--expert-pool`.
