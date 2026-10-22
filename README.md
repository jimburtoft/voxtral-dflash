# voxtral-dflash

Training code for an openly licensed [DFlash](https://arxiv.org/abs/2602.06036) speculative-decoding
draft for [`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507),
trained on AWS Trainium2. The released draft (weights + model card) is on Hugging Face; see
`MODEL_CARD.md` for results and usage.

All code is Apache-2.0 (`LICENSE`, `NOTICE`). Only public sources and openly licensed data are
used; datasets, licenses, pinned revisions and filters are documented in `data/LICENSES.md`.

## Layout

| dir | contents |
|---|---|
| `model/` | `dflash.py` draft model, `config.py` release config, `release.py` save/load, `voxtral_target.py` static-shape Voxtral target (audio encoder + 30-layer decoder), `device.py` device selection / compile backend (Trainium, CUDA, CPU) |
| `train/` | `dflash_objective.py` (anchors, block mask, weighted loss, acceptance), `data.py` (label/audio loading, batch packing), queue trainer + feature producers, launch scripts, CPU reference trainer |
| `labels/` | target-generated labels: greedy transcripts of real audio (short and long-form) |
| `data/` | corpus build (`prep/`), license / overlap audits (`scripts/`), filters, slim manifests, `LICENSES.md` |
| `eval/` | offline acceptance evaluator, upstream vLLM loader / format checks |
| `tests/` | CPU unit tests |
| `tools/` | `extract_text_decoder.py` (Voxtral text decoder as a plain Llama checkpoint, for loader tests) |

## Environments

- **CPU** (model code, tests, offline checks): `pip install -r requirements-cpu.txt`, then
  `python -m pytest -q tests`.
- **Trainium2** (labels, feature extraction, training): a trn2 instance launched from the public
  **AWS Deep Learning AMI Neuron, SDK 2.33** (`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0`
  multi-framework, or `Deep Learning AMI Neuron torch-neuronx 2.14.0`). Use its torch-neuronx
  virtual environment as is and add the few extra packages on top:

  ```bash
  source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate   # torch 2.14 + torch-neuronx 2.14.3
  pip install -r requirements-neuron.txt                            # does not touch torch / torch-neuronx
  python -m pytest -q tests
  ```

  No container, no custom build and no torch-xla: the code uses plain PyTorch
  (`torch.device("neuron")`, `torch.compile(backend="neuron")`). We used one trn2.3xlarge
  (1 Trainium2 device, 4 logical cores at LNC=2, the default) for everything. Add swap before
  compiling (the DLAMI ships none; 64 GB is plenty).
- **NVIDIA GPU.** Nothing in the training path is Trainium-specific. Every script takes
  `--device auto|neuron|cuda|cpu` (`auto` = Trainium if torch-neuronx is installed, else CUDA, else
  CPU) and compiles with the matching `torch.compile` backend (`model/device.py`). The launch
  scripts pin one process per core / device with both `NEURON_RT_VISIBLE_CORES` and
  `CUDA_VISIBLE_DEVICES`; with fewer GPUs than processes set `NDEV` (e.g. `NDEV=1` puts the
  producer and the trainer on one GPU). Validated on an L40S (48 GB) with the AWS Deep Learning
  OSS Nvidia Driver AMI (PyTorch 2.12, CUDA 13): `source /opt/pytorch/bin/activate`, then
  `pip install -r requirements-cpu.txt` (does not change torch). Results below.

The launch scripts run from the repo root inside that environment. Paths come from
`$VXDF_CORPUS` (corpus, default `~/vxdf_corpus`) and `$VXDF_WORK` (labels, eval items, runs,
default `~/vxdf_work`); see `train/_env.sh`.

## Reproducing the release

Times are for one trn2.3xlarge.

### 1. Corpus (CPU, a few hours, ~90 GB disk)

```bash
cd data/prep
for n in ls_train ltts_train mls_train cv_train vp_train ps_train ami_train \
         ls_eval cv_eval vp_eval ami_eval fleurs_eval; do
  python build_corpus.py all $n --workers 8
done
python ami_longform.py --n 60            # eval: 30 s - 5 min AMI test windows
python manifest_and_sanity.py            # manifests + sanity checks (fails loudly)
python build_longform.py --workers 4     # long-form TRAIN set from the train shards only
python longform_sanity.py
```

This writes 175,000 train clips (464 h, 1-30 s), 4,301 long-form train clips (179 h, 31-300 s)
and 12,108 eval clips to `$VXDF_CORPUS/shards/`. Every filter (eval speakers / meetings, transcript
overlap with eval, People's Speech license allow-list) is applied here; see `data/LICENSES.md`.
The slim manifests in `data/corpus_manifest/` list the clip ids we used.

### 2. Labels (Trainium2, ~2 h short + ~6 h long-form)

Voxtral's own greedy transcripts of the real audio:

```bash
sh labels/run_labels_all.sh                  # short train + eval, one process per core
W=${VXDF_WORK:-~/vxdf_work}
for r in 0 1 2 3; do                          # long-form train (2-10 audio windows)
  NEURON_RT_VISIBLE_CORES=$r python labels/generate_labels_longform.py \
    --root $VXDF_CORPUS/shards/train_longform --out $W/labels/train_longform --rank $r --world 4 &
done; wait
python labels/generate_labels_longform.py --root $VXDF_CORPUS/shards/eval/ami_longform \
  --out $W/labels/eval_longform                # long-form eval labels
python labels/label_wer.py "$W/labels/train/*.labels.parquet"   # sanity: WER vs references
```

Target hidden states are not stored (~2 TB); they are recomputed on the fly during training.

### 3. Held-out evaluation items (Trainium2, ~30 min)

```bash
# short: 64 clips per held-out set, an anchor at every response position
for r in 0 1 2 3; do
  NEURON_RT_VISIBLE_CORES=$r python train/feature_producer.py --labels "$W/labels/eval/*.labels.parquet" \
    --audio-root $VXDF_CORPUS/shards --rank $r --world 4 --batch 4 \
    --eval-out $W/eval_batches/b4_per64 --eval-per-set 64 &
done; wait
# long-form: 24 AMI long-form windows
python train/feature_producer_longform.py --labels $W/labels/eval_longform \
  --audio-root $VXDF_CORPUS/shards/eval/ami_longform --queue /dev/shm/unused \
  --eval-out $W/eval_batches/longform_eval --eval-n 24 --n-blocks 64
```

### 4. Stage 1: short clips (Trainium2, ~17 h)

Cores 0-2 run frozen-target feature producers, core 3 trains the draft (the bf16 target and the
fp32 draft with AdamW state do not fit one 24 GB core together). Runs land in `$W/runs/<name>/`:

```bash
sh train/launch_queue_run.sh stage1 262500 \
  --compute bf16 --lr 6e-4 --warmup-frac 0.04 --eval-every 10000 --save-every 10000 --log-every 200
```

Batch = 4 clips x 16 random anchor blocks of 16 tokens, paper loss weights (gamma 7), AdamW
(0.9, 0.95), cosine schedule, clip 1.0, bf16 autocast with fp32 master weights. Held-out tau
plateaus around 180k-220k steps; the release used the 220k checkpoint (we stopped at 239k).

### 5. Stage 2: long-form adaptation (Trainium2, ~8 h)

One short-clip producer, two long-form producers, one long-form item every 2 steps:

```bash
N_SHORT=1 N_LONG=2 LONG_EVERY=2 sh train/launch_stage2.sh s2a 20000 <stage1>/step0220000/train_state.pt \
  --compute bf16 --lr 2e-4 --warmup-frac 0.02 --eval-every 2000 --save-every 2000
N_SHORT=1 N_LONG=2 LONG_EVERY=2 sh train/launch_stage2.sh s2b 16000 <s2a>/step0020000/train_state.pt \
  --compute bf16 --lr 1e-4 --warmup-frac 0.01 --eval-every 2000 --save-every 2000
```

The release is `s2b/step0014000/release/` (best mean of short and long-form held-out tau:
7.64 / 4.38). Without stage 2 the draft gets tau ~1.1 on multi-window (> 30 s) audio.

### 6. Checks

```bash
python eval/acceptance_eval.py --ckpt $W/runs/<run>/stepNNNNNNN/train_state.pt \
  --eval-dirs $W/eval_batches/b4_per64 $W/eval_batches/longform_eval
python eval/upstream_format_check.py --help              # upstream vLLM (CPU docker) loader check
python eval/vllm_neuron_serve_bench.py --help            # end-to-end serving (see "Serving benchmark")
```

`train/parity_draft_step.py --item <one eval item .pt>` checks the accelerator draft step against the
CPU fp32 reference (loss, gradients, AdamW update); `train/train_cpu_ref.py` is a small CPU reference trainer that works from
`labels/generate_labels_hf.py` output.

## Validation (public environments only)

The release draft was trained with a pre-release PyTorch-on-Neuron build. The pipeline was then
re-run with the steps in this README on two public environments:
- **Trainium2:** a fresh trn2.3xlarge, `Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0`
  (torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.28).
- **NVIDIA GPU:** a g6e.4xlarge (1x L40S), AWS Deep Learning OSS Nvidia Driver AMI (PyTorch 2.12.1,
  CUDA 13), `--device auto` -> `cuda`, with the producer and trainer on one GPU (`NDEV=1`).

| check | Trainium2, public SDK 2.33 | NVIDIA L40S | original (pre-release build) |
|---|---|---|---|
| install requirements into the image's venv | torch / torch-neuronx unchanged, `pip check` clean | torch unchanged | -- |
| unit tests (incl. z-lab reference) | 8 passed | 8 passed | 7 passed |
| step 3 eval items (encoder + 30-layer target) | 170 short / 244 long-form | 170 / 244 | 170 / 244 |
| held-out tau of the release checkpoint, short (6 sets) | **7.636** | **7.627** | 7.639 |
| held-out tau, AMI long-form | **4.388** | **4.387** | 4.381 |
| labels, FLEURS (647 clips), vs the original labels | 100 % finished, 97.4 % token-identical, char sim 0.99965 | 100 % finished, 96.6 % token-identical, char sim 0.99937 | -- |
| `train/parity_draft_step.py` vs CPU fp32 (5 AdamW steps), fp32 / bf16 autocast | grad cos 0.999994 / 0.999968, PARITY_OK | grad cos 1.000000 / 0.999964, PARITY_OK | 0.999994 / 0.999964 |
| stage 1 smoke run, 1,500 steps | loss 11.9 -> 4.6, tau 1.29, 0.52 s/step | loss 11.6 -> 4.7, tau 1.29, 0.29 s/step | same curve |

Token differences in the labels are bf16 numerics of different compilers / kernels (the
transcripts are 99.9+ % character-identical); the draft's acceptance is the same on all three.

## Serving benchmark (vllm-neuron, SDK 2.33)

`eval/vllm_neuron_serve_bench.py` starts `vllm serve` and sends real audio to
`/v1/audio/transcriptions`, one request at a time, with and without the draft; tau comes from the
server's spec-decode counters. Setup: trn2.3xlarge, TP=4, encoder TP=4, LNC=2, greedy, the
SDK 2.33 DLAMI vLLM environment (`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`,
vllm 0.29.0). The released vllm-neuron 0.29.0.1.2.0 in that environment does not include Voxtral
yet (`vllm serve` fails with `VoxtralForConditionalGeneration has no attribute from_configs`); the
numbers below use vllm-neuron 0.29.0.1.2.0 plus the open Voxtral pull request
([vllm-project/vllm-neuron#69](https://github.com/vllm-project/vllm-neuron/pull/69), commit
`75f45e4`) installed with `pip install --target <dir> --no-deps` and put first on `PYTHONPATH`.
That environment also lacks vLLM's `audio` extra; add `soundfile soxr av` the same way.

Short clips: 50 random held-out clips per set (<= 30 s). Long-form: 60 AMI test windows (30 s -
5 min), `--max-model-len 8192`, encoder buckets `[1, 4, 10]`, `VLLM_NEURON_RESERVED_HEADROOM_GIB=8`.

| set | tau | mean latency without / with draft (ms) | speedup | WER without / with draft |
|---|---|---|---|---|
| LibriSpeech test-clean | 8.47 | 249 / 93 | 2.67x | 2.29 / 2.29 % |
| LibriSpeech test-other | 7.08 | 222 / 92 | 2.40x | 4.45 / 4.45 % |
| VoxPopuli test | 8.50 | 247 / 92 | 2.68x | 8.22 / 8.22 % |
| FLEURS en_us test | 6.42 | 265 / 108 | 2.46x | 4.10 / 4.18 % |
| AMI ihm test | 6.02 | 140 / 76 | 1.84x | 16.84 / 17.91 % |
| Common Voice 17 test | 4.17 | 136 / 87 | 1.56x | 11.53 / 11.95 % |
| **all short (300)** | **6.84** | **210 / 91** (median 191 / 85) | **2.30x** | **6.55 / 6.71 %** |
| AMI long-form (60 windows) | 3.78 | 4,992 / 2,487 (median 4,382 / 2,103) | 2.01x | 34.59 / 34.55 % |

Greedy speculative decoding verifies every token with the target, so the output should equal the
no-draft output. In practice 290 / 300 short transcripts are identical; the other 10 differ because
the 16-token verify graph and the 1-token decode graph round bf16 differently, and one flipped
token changes the rest of a greedy decode. On those 10 clips the draft run matches the target's
independent PyTorch greedy label (step 2) more often (6) than the no-draft server (2). Most of the
short-clip WER difference is 2 AMI clips (out of 50, 1-27 reference words) where the verify
graph continued past a short utterance. Long-form transcripts are identical for 22 / 60 windows,
for the same reason over 30-300 s of output; WER is unchanged. AMI references are unpunctuated
Kaldi-style text, which inflates its absolute WER (simple lowercase / punctuation normaliser).

## Optional reference test

`tests/test_dflash_objective.py::test_matches_public_zlab_reference` compares our draft against
the public z-lab DFlash reference implementation (MIT). It is skipped unless the reference files
are present:

```bash
mkdir -p tmp/refs
curl -L -o tmp/refs/zlab_dflash.py https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/resolve/main/dflash.py
curl -L -o tmp/refs/utils.py       https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/resolve/main/utils.py
```

## Notes

- Serving Voxtral + this draft needs a server whose Voxtral implementation exposes the target's
  auxiliary hidden states at `target_layer_ids + 1`. vllm-neuron: the open pull request above.
  Upstream vLLM 0.32 has DFlash, but its `VoxtralForConditionalGeneration` does not implement the
  auxiliary hidden-state (EAGLE3) interface yet.
- Static-shape design choices (documented in the code): the audio encoder and the decoder are
  compiled as separate graphs, the target forward uses causal attention with right padding instead
  of a key-padding mask (exact for the valid positions), and multi-window prompts use chunked
  prefill so one graph serves every prompt length.
