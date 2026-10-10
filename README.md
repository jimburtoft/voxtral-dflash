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
| `model/` | `dflash.py` draft model, `config.py` release config, `release.py` save/load, `voxtral_target.py` static-shape Voxtral target (audio encoder + 30-layer decoder, CPU or Neuron) |
| `train/` | `dflash_objective.py` (anchors, block mask, weighted loss, acceptance), `data.py` (label/audio loading, batch packing), Trainium trainer + feature producers, CPU reference trainer |
| `labels/` | target-generated labels: greedy transcripts of real audio (short and long-form) |
| `data/` | corpus build (`prep/`), license / overlap audits (`scripts/`), filters, slim manifests, `LICENSES.md` |
| `eval/` | offline acceptance evaluator, upstream vLLM loader / format checks |
| `tests/` | CPU unit tests |
| `tools/` | `extract_text_decoder.py` (Voxtral text decoder as a plain Llama checkpoint, for loader tests) |

## Environments

- **CPU** (model code, tests, offline checks): `pip install -r requirements-cpu.txt`, then
  `python -m pytest -q tests`.
- **Trainium2** (labels, feature extraction, training): deploy a Neuron PyTorch Native
  (TorchNeuron) environment on a trn2 instance (the code uses `torch.device("neuron")` and
  `torch.compile(backend="neuron")`; no torch-xla), then `pip install -r requirements-neuron.txt`
  inside it without replacing its torch build. We used one trn2.3xlarge (1 Trainium2 device,
  4 logical cores at LNC=2) for everything.

The launch scripts assume the working tree is mounted in a container named `$CONTAINER`
(default `vxdf`) at `$HOST_ROOT` (default `/host`), with the corpus under `$HOST_ROOT/vxdf_corpus`;
adjust to your setup (or run the `python` commands they contain directly). Scripts that read the
corpus default to `$VXDF_CORPUS` (default `~/vxdf_corpus`).

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
for r in 0 1 2 3; do                          # long-form train (2-10 audio windows)
  NEURON_RT_VISIBLE_CORES=$r python labels/generate_labels_longform.py \
    --root $VXDF_CORPUS/shards/train_longform --out vxdf_labels/train_longform --rank $r --world 4 &
done; wait
python labels/generate_labels_longform.py --root $VXDF_CORPUS/shards/eval/ami_longform \
  --out vxdf_labels/eval_longform              # long-form eval labels
python labels/label_wer.py "vxdf_labels/train/*.labels.parquet"   # sanity: WER vs references
```

Target hidden states are not stored (~2 TB); they are recomputed on the fly during training.

### 3. Held-out evaluation items (Trainium2, ~30 min)

```bash
# short: 64 clips per held-out set, an anchor at every response position
for r in 0 1 2 3; do
  NEURON_RT_VISIBLE_CORES=$r python train/feature_producer.py --labels "vxdf_labels/eval/*.labels.parquet" \
    --audio-root $VXDF_CORPUS/shards --rank $r --world 4 --batch 4 \
    --eval-out eval_batches/b4_per64 --eval-per-set 64 &
done; wait
# long-form: 24 AMI long-form windows
python train/feature_producer_longform.py --labels vxdf_labels/eval_longform \
  --audio-root $VXDF_CORPUS/shards/eval/ami_longform --queue /dev/shm/unused \
  --eval-out eval_batches/longform_eval --eval-n 24 --n-blocks 64
```

### 4. Stage 1: short clips (Trainium2, ~17 h)

Cores 0-2 run frozen-target feature producers, core 3 trains the draft (the bf16 target and the
fp32 draft with AdamW state do not fit one core together):

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
python eval/acceptance_eval.py --ckpt <run>/stepNNNNNNN/train_state.pt \
  --eval-dirs eval_batches/b4_per64                     # Trainium2
python eval/upstream_format_check.py --help              # upstream vLLM (CPU docker) loader check
```

`train/parity_draft_step.py` checks the Trainium2 draft step against the CPU fp32 reference (loss,
gradients, AdamW update); `train/train_cpu_ref.py` is a small CPU reference trainer that works from
`labels/generate_labels_hf.py` output.

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
  auxiliary hidden states; vllm-neuron Voxtral support is pending upstream.
- Known compiler workarounds are commented in the code: the audio encoder and decoder prefill
  are compiled as separate graphs, the target forward uses causal attention with right padding
  instead of a key-padding mask, and multi-window prompts use chunked prefill.
