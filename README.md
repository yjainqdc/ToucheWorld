# ToucheWorld

Hand-conditioned egocentric video generation with LTX-2.5 and the SolarWM camera adapter.

This repository contains hand-conditioning modules, latent preprocessing, teacher and causal training, and inference entry points. Teacher-27 and Causal-27 adapters are released separately. **DMD weights: coming soon.**

## Environment

Use Linux with NVIDIA CUDA GPUs. The preprocessing code uses `fcntl`; this release does not support Windows. Full training and inference require the 22B base model and substantial GPU memory.

Install the compatible LTX-2.5 and SolarWM source environments first, then install this repository's direct dependencies:

```bash
python -m pip install -r requirements.txt
cp config.env.example config.env
```

Edit `config.env` to point to:

- The LTX-2.5 checkout containing `packages/ltx-core`, `packages/ltx-trainer`, and `packages/ltx-pipelines`.
- The compatible SolarWM source directory containing `solarwm/backends/ltx25`.
- The base transformer, video VAE, Gemma text encoder, and SolarWM camera adapter.
- The dataset root and latent cache directory.

These external source packages and weights are required and are not bundled. `requirements.txt` lists direct dependencies; it is not a validated environment lockfile. Match PyTorch, CUDA, and optional attention kernels to the upstream environments.

All launchers load `config.env` automatically. Select another file with `TOUCHEWORLD_CONFIG`. Explicitly supplied path settings and GPU lists take precedence over values from that file. Keep other stage-specific options commented out when supplying them inline.

## Adapter layout

Place the separately released weights under `models/ToucheWorld/`, or override `MODEL_DIR`, `RUN_DIR`, `HAND_RUN`, and `DISTILL_RUN`:

```text
models/ToucheWorld/
├── teacher27/
│   ├── hand-lora-ema-step27307.safetensors
│   └── hand-sidecar-step27307.safetensors
└── causal27/
    ├── hand-lora-ema-step27307.safetensors
    ├── hand-sidecar-step27307.safetensors
    └── distill-lora-ema-step10240.safetensors
```

The hand LoRA and sidecar are identical in both folders. The sidecar is a conditioning encoder, not a LoRA. These adapters require the LTX-2.5 base model and SolarWM camera weights.

## Prepare latent caches

Each JSONL row identifies a clip and paths to RGB, hand control, hand motion, and camera files. See `examples/manifest.jsonl` for a synthetic example; example media are not included. Relative media paths resolve under `DATA_ROOT`, independently of the manifest's directory.

The packed data convention is 153 frames at 24 fps, 768×512, with 182-dimensional hand motion and consistent `right,left` side order. Use compatible preprocessing and check the schema expected in `pipelines/ltx25_solar/hand_geometry.py`.

To preprocess without starting training:

```bash
SKIP_TRAIN=1 bash scripts/run_solar_hand_sft.sh
```

Cache files are named by replacing `:` with `_` in each `record_id`. For example, `sample:000001` becomes `sample_000001.safetensors`.

## Inference

Teacher-27, first-frame image-to-video:

```bash
CASE=sample:sample_000001.safetensors:0 \
  bash scripts/infer_teacher.sh
```

`CASE` follows `name:cache_filename:zero_hand`. Set the last field to `1` for a zero-hand-conditioning ablation. Additional `--case` arguments may be appended to the launcher.

Causal-27, block-autoregressive generation:

```bash
JSONL=/path/to/eval.jsonl \
  bash scripts/infer_causal.sh
```

Default causal settings are hand step 27307, distillation step 10240, block 4, window 8, and 30 denoising steps. The inference launcher uses `PREFIX_LATENTS=1` for first-frame conditioning. Set `PREFIX_LATENTS=8` only for the evaluation protocol requiring approximately two seconds of ground-truth context.

Outputs default to `outputs/teacher` and `outputs/causal`.

## Training

Teacher training keeps the camera adapter frozen and learns the hand LoRA and sidecar:

```bash
SKIP_PREENCODE=1 HAND_REGION_WEIGHT=3 \
OUTPUT_DIR=runs/teacher TRAIN_GPUS=0 \
  bash scripts/run_solar_hand_sft.sh
```

Causal training freezes the hand components and learns the causal distillation adapter:

```bash
HAND_RUN=/path/to/teacher-run HAND_STEP=27307 \
OUTPUT_DIR=runs/causal TRAIN_GPUS=0 \
  bash scripts/run_solar_causal_lora.sh
```

GPU lists are configurable; a single-GPU default is a configuration example, not a claim that a particular GPU has enough memory. Set step counts, learning rates, checkpoint intervals, and sequence parallelism for the target run. The launchers record the selected configuration in the output directory.

The DMD and earlier few-step training implementations remain available for research. `scripts/run_solar_hand_dmd.sh` accepts `HAND_RUN`, `HAND_STEP`, `CAUSAL_RUN`, and `CAUSAL_STEP` explicitly. Its defaults reference the released Teacher-27 / Causal-27 adapters; it is not a recipe claimed to reproduce a released DMD checkpoint.

## Code map

| Path | Purpose |
| --- | --- |
| `pipelines/ltx25_solar/hand_conditioning.py` | Spatial control projection and motion encoder |
| `pipelines/ltx25_solar/hand_inject.py` | Adapter and sidecar injection |
| `pipelines/ltx25_solar/hand_geometry.py` | Media, motion, and camera geometry helpers |
| `pipelines/ltx25_solar/preencode_hand.py` | Latent cache preprocessing |
| `pipelines/ltx25_solar/train_hand.py` | Teacher training and hand-region loss |
| `pipelines/ltx25_solar/train_causal.py` | Causal teacher-forcing training |
| `pipelines/ltx25_solar/block_rollout.py` | Block-autoregressive sampling |
| `pipelines/ltx25_solar/train_dmd.py` | Experimental DMD training |
| `eval/sample_solar_hand.py` | Cached-sample teacher inference |
| `eval/sample_bench_causal.py` | Manifest-driven causal inference |
| `pipelines/common/lora.py` | Shared EMA and checkpoint key helpers |
| `pipelines/common/paths.py` | Environment-based path defaults |

## License and acknowledgments

See `LICENSE-2_x` for the retained LTX-2.x Community License. This release depends on [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and SolarWM. Separately obtained dependencies, checkpoints, and datasets remain subject to their own licenses. Consult the applicable terms before use or redistribution.
