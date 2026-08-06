# Phase 0 - measurement scripts

Everything in `PLAN-qwen36-27b.md` is sized off these numbers. Nothing here changes
any code; it only builds, downloads, and measures.

Target machine: the 6 GB VRAM / 16 GB RAM Ubuntu server with CUDA.

## Order

```bash
cd phase0
chmod +x *.sh

./00-build.sh                                   # optional if you already built
HF_REPO=<user>/<model>:Q4_K_M ./01-download.sh
python3 02-gguf-layout.py "$(cat results/model-path.txt)" --vram-mib 5000 --json results/layout.json
./03-fit.sh
./04-residency.sh
NGL_LIST=0,4,8,12,16,20,24 ./05-ngl-sweep.sh
```

Shared settings live in `config.sh` - model path, `N_CTX`, prompt sizes, and
`DROP_CACHES`. Override by exporting before the call, e.g. `N_CTX=4096 ./03-fit.sh`.

`02-gguf-layout.py` needs **no third-party packages** - not even numpy. It parses the
GGUF metadata and tensor-info blocks itself with `struct`, and pulls the block-size
table from the repo's own `gguf-py/gguf/constants.py` (which has no third-party
imports) so the sizes cannot drift from the rest of the tree. The tensor data is
never read, so it is fast even on an 18 GiB file.

**Set `DROP_CACHES=1` if you have passwordless sudo.** Without dropping the page
cache between runs, run 2 inherits run 1's cached weights and the residency
comparison in `04` loses most of its meaning.

## What each script answers

| script | plan section | question |
|---|---|---|
| `00-build.sh` | - | CUDA build with the right compute capability auto-detected |
| `01-download.sh` | - | fetch the model, and the MTP sidecar if the repo has one |
| `02-gguf-layout.py` | 0.1, 2.3, 6 | is it really 48 GDN + 16 attention layers? what does one layer cost? how many trailing layers fit in 5 GB? |
| `03-fit.sh` | 0.2, 2.4, 3.2 | what does the fitter decide, at what margin, and does it emit any tensor overrides for a dense model (I predict zero) |
| `04-residency.sh` | 1.2, 1.4, 3.2 | **the important one.** does `--repack` convert the CPU-side weights into swappable anonymous RAM? what does MTP cost in VRAM? |
| `05-ngl-sweep.sh` | 2.1, 2.2 | throughput vs. layers on GPU, and how much of it is KV placement |

## Reading `04-residency.sh`

The whole repack theory (`PLAN` section 1.2) lives or dies on two columns:

- **`rss_anon`** - anonymous heap memory. Swappable.
- **`rss_file`** - file-backed mmap pages. Clean; Linux evicts these rather than
  swapping them.

Prediction: in the `default` run `rss_anon` climbs to many GiB as the repack path
copies weights into owned buffers. In the `norepack` run that mass moves to
`rss_file` and `rss_anon` stays small.

If that is what you see, the plan's Phase 1a is confirmed and `-nr` becomes part of
the standing configuration - **but check the timings too**, because repack is a
real CPU matmul speedup and most of this model is on the CPU. It is a trade, not a
free win.

If `rss_anon` is small in *both* runs, my section 1.2 analysis is wrong and the
swap you predicted comes from somewhere else. Send me the CSVs either way.

## A note on the `UD-Q4_K_XL` quant

Unsloth's UD quants are mixed-precision - different tensors get different types.
That does not weaken the repack concern in `PLAN` section 1.2, because the CPU
repack path covers essentially the whole spread: `Q4_0`, `Q4_K`, `Q2_K`, `Q5_K`,
`Q6_K`, `IQ4_NL`, `MXFP4` and `Q8_0` all have repack traits
(`ggml/src/ggml-cpu/repack.cpp:4573-4699`), gated on `ne[1] % 8 == 0` for the AVX2
path. So expect most large 2D weights to be repack-eligible regardless of the mix.

## Known tool limitations found while writing these

Two things worth knowing, both of which are findings in their own right:

1. **`llama-fit-params` cannot model an MTP run at all.** `--mtp` is registered
   only for the download example (`common/arg.cpp:3009`) and `--spec-type` only for
   speculative/server/cli (`common/arg.cpp:4102`). Options outside the current
   example are never registered, so their `LLAMA_ARG_*` env vars are not consulted
   either (`common/arg.cpp:1385`). This makes `PLAN` section 3.2 worse than stated -
   the probe does not merely under-count the MTP context, the tool cannot be asked
   about it. Phase 3 should register `--spec-type` for `LLAMA_EXAMPLE_FIT_PARAMS`.
   Until then, MTP VRAM is measured empirically in `04-residency.sh`.

2. **`llama-bench` has no repack toggle.** It parses its own args and exposes
   `-lm`, `--no-host`, `-nkvo`, `-ot` but not `-nr`. So `05-ngl-sweep.sh` numbers
   are all repack-enabled. Any repack comparison has to go through
   `llama-cli`/`llama-completion`.

## What to send back

- `results/layout.json` and the `02` console output
- `results/03-fit/` in full (the verbose trace matters)
- `results/04-residency/*.summary.txt` and `*.csv`
- `results/05-ngl-sweep/*.md`

Those six things close every open item in `PLAN-qwen36-27b.md` section 6 except the
`NOTES.md` A.4 mmap-for-GPU-tensors question, which needs its own experiment.
