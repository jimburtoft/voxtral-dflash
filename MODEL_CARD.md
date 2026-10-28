---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-3B-2507
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - speculative-decoding
  - dflash
  - voxtral
  - draft-model
  - trainium
language:
  - en
datasets:
  - openslr/librispeech_asr
  - mythicinfinity/libritts_r
  - parler-tts/mls_eng
  - fsicoli/common_voice_17_0
  - facebook/voxpopuli
  - MLCommons/peoples_speech
  - edinburghcstr/ami
---

# Voxtral-Mini-3B-2507 DFlash draft

A [DFlash](https://arxiv.org/abs/2602.06036) block-diffusion speculative-decoding draft for
[`mistralai/Voxtral-Mini-3B-2507`](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507). It
proposes 15 tokens per step in one forward pass and is verified by Voxtral under greedy decoding.
The draft was trained on AWS Trainium2 with openly licensed speech data only.
Training code: [github.com/jimburtoft/voxtral-dflash](https://github.com/jimburtoft/voxtral-dflash).

- **Format:** the public DFlash checkpoint format (`DFlashDraftModel`, Qwen3 layers; the format
  used by vLLM's `qwen3_dflash.py` and the `z-lab/dflash` reference code).
- **Size:** 5 draft layers, hidden 3072, 582M parameters (BF16, 1.16 GB). The token embeddings
  and LM head are **not** shipped: the draft uses Voxtral's own, loaded from the target checkpoint.
- **Block size 16** (`num_speculative_tokens = 15`). Serve at block size 16: acceptance drops at
  smaller serving blocks for this draft (see results).
- **Target features:** outputs of Voxtral text-decoder layers `[1, 8, 14, 20, 27]`.
- **Mask token:** id 20 (a reserved, never-produced tekken special token).

## Results

All numbers use real audio, greedy decoding, batch size 1, and held-out sets that were never used
in training. tau = mean accepted tokens per draft step, including the target's bonus token.

### Acceptance (offline, reproducible with the released code)

Draft evaluated on 64 held-out clips per set with an anchor at **every** response position, using
the same greedy accept rule as vLLM DFlash verification:

| set | tau |
|---|---|
| LibriSpeech test-clean / test-other, Common Voice 17 en test, VoxPopuli en test, FLEURS en_us test, AMI ihm test (<= 30 s, all) | **7.64** |
| AMI long-form test windows (30 s - 5 min, 2-10 audio windows) | **4.38** |

### End-to-end serving (vllm-neuron, Neuron SDK 2.33, Trainium2)

Environment: trn2.3xlarge, the public Neuron SDK 2.33 DLAMI vLLM venv (vllm 0.29.0,
vllm-neuron 0.29.0.1.2.0) with the vllm-neuron Voxtral pull request
([vllm-project/vllm-neuron#69](https://github.com/vllm-project/vllm-neuron/pull/69), commit
`e0778ed`) on top, because the released vllm-neuron does not include Voxtral yet. That pull
request also adds the fixes these numbers depend on (DFlash draft mask, GQA decode kernel,
sequence-parallel audio encoder, LNC=1 kernel grids). Exact serving commands: the Voxtral recipe
in that pull request (`docs/model-recipes/voxtral.md`).

**One request at a time** (TP=4, LNC=2, greedy; in-process `LLM.generate`, 2 passes, measured
with the GQA decode kernel in place):

| set | without draft (mean) | with draft (mean) | speedup | tau |
|---|--:|--:|--:|--:|
| LibriSpeech test-other, 48 clips | 167.5 ms | 88.2 ms | 1.90x | 5.37 |
| LibriSpeech test-clean, 24 clips | -- | 85.3 ms | | 6.10 |

With the 384-token prefill bucket and the sequence-parallel encoder the same test-other clips take
86.0 ms over HTTP. The GQA decode kernel left the no-draft transcripts unchanged (48 / 48). With
the draft, 45 / 48 test-other and 24 / 24 test-clean transcripts are unchanged; the other 3 differ
by bf16 rounding in the multi-token verify step. WER with the draft is 4.91 % on test-clean; on
test-other it is 7.61-7.87 % with the in-process harness and 5.71-5.96 % with the HTTP harness
(simple lowercase / punctuation normaliser). The in-process harness uses a chat prompt ("Transcribe
this audio."), the HTTP harness the `/v1/audio/transcriptions` transcription prompt, which
transcribes better.

**Serving, closed loop** (48 LibriSpeech test-other clips, 2 passes, with the draft):

| concurrent requests | 1 | 2 | 4 | 8 | 16 |
|---|--:|--:|--:|--:|--:|
| TP=4, LNC=2 (`--max-num-seqs 16`): real-time factor | 68.8x | 80.1x | 94.9x | 105.5x | 114.6x |
| TP=4, LNC=2: p50 latency | 82 ms | 139 ms | 232 ms | 381 ms | 677 ms |
| TP=4 x DP=2, LNC=1 (`--max-num-seqs 8` per replica): real-time factor | 52.5x | 95.0x | 109.3x | 126.3x | 138.4x |
| TP=4 x DP=2, LNC=1: p50 latency | 108 ms | 118 ms | 207 ms | 331 ms | 580 ms |
| tau (both) | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 |

Without the draft the LNC=2 real-time factor is 30.8x at 1 and 76.0x at 16 concurrent requests.
LNC=2 gives the lowest latency for single requests; two LNC=1 replicas give 1.16-1.21x the
throughput with two or more requests in flight.

Earlier per-dataset results (2026-10-22, before the GQA kernel and encoder changes, one request
at a time, 50 held-out clips per set) for the acceptance on other data: tau 8.47 LibriSpeech
test-clean, 7.08 test-other, 8.50 VoxPopuli, 6.42 FLEURS, 6.02 AMI ihm, 4.17 Common Voice 17;
3.78 on 60 AMI long-form windows (30 s - 5 min). The speedup over no draft was 1.56x (Common Voice)
to 2.68x (VoxPopuli) per set, 2.30x over all 300 short clips and 2.01x on long-form audio. Greedy
speculative decoding keeps the target's output up to bf16 rounding in the multi-token verify graph
(290 / 300 short transcripts identical to the no-draft run).

Serving block-size sweep (short clips, same weights, earlier pre-release build): block 16:
tau 6.81; 14: 4.06; 12: 3.98; 8: 3.33. Serve at block 16.

## Usage

### vLLM (DFlash speculative decoding)

```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer-mode mistral \
  --speculative-config '{"method": "dflash", "model": "<this repo>", "num_speculative_tokens": 15}'
```

The draft needs the target's auxiliary hidden states (`target_layer_ids + 1` capture layers).
The loader format is verified against upstream vLLM 0.31 (CPU) with Voxtral's text decoder:
config, weights and draft tokens match the reference implementation. Voxtral's multimodal model
class in upstream vLLM (checked through 0.32) does not expose auxiliary hidden states yet, so
end-to-end audio serving there needs that support first. On Trainium, use vllm-neuron with the
Voxtral support from [vllm-project/vllm-neuron#69](https://github.com/vllm-project/vllm-neuron/pull/69)
(pending merge), as in the results above.

### Reference implementation (PyTorch)

The [training code repository](https://github.com/jimburtoft/voxtral-dflash) contains `model/dflash.py` (the draft), `model/release.py`
(load/save), and `eval/acceptance_eval.py` (offline acceptance). The checkpoint also loads with the
public `z-lab/dflash` reference modelling code (identical block outputs to 2e-5).

## Training

Full reproduction steps (corpus build, labels, both training stages, evaluation) are in the
[code repository](https://github.com/jimburtoft/voxtral-dflash) README.


- **Data:** 175,000 clips (464 h, <= 30 s) + 4,301 long-form clips (179 h, 30-300 s, built by
  concatenating consecutive training utterances of the same meeting / speaker / source) from
  LibriSpeech, LibriTTS-R, Multilingual LibriSpeech (en), Common Voice 17 (en), VoxPopuli (en),
  The People's Speech (clean, CC BY / public-domain sources only) and AMI. Eval speakers, eval
  meetings and transcripts overlapping eval sets were removed from training. FLEURS was used for
  evaluation only.
- **Labels:** Voxtral's own greedy transcripts of the real audio (target-generated labels), and
  Voxtral's hidden states at the five conditioning layers, computed on the fly during training.
- **Objective (paper recipe):** random anchors in the response, block of 16 = anchor + 15 masks,
  bidirectional attention within a block and to the target features before the anchor,
  exponentially decaying position weights (gamma 7).
- **Schedule:** stage 1 on short clips (239k steps of 4 clips x 16 blocks, AdamW 6e-4 cosine, bf16
  autocast with fp32 master weights); stage 2 long-form adaptation from stage 1 (36k steps mixing
  short and long-form batches, lr 2e-4 then 1e-4). Multi-window (long-form) adaptation raised
  long-form tau from 1.1 to 4.4 while keeping short-clip tau within 2 %.
- **Compute:** AWS Trainium2 (trn2.3xlarge) with PyTorch on Neuron; about 45 instance-hours for
  labelling, training and evaluation.
- **Environment:** to reproduce, use the public AWS Deep Learning AMI Neuron, SDK 2.33
  (`/opt/aws_neuronx_venv_torch_neuronx_2_14_0`, torch 2.14 + torch-neuronx 2.14.3) plus
  `requirements-neuron.txt`. The draft itself was trained with a pre-release build of that stack.
  On the public SDK 2.33 DLAMI, the released checkpoint gives the same held-out acceptance
  (7.636 short / 4.388 long-form vs 7.639 / 4.381), and the trainer passes the CPU parity check.
  The training code also runs on CUDA or CPU (`--device`).

## Limitations

- English only, transcription prompts only (the draft was trained on Voxtral's transcription
  template; chat / audio-understanding prompts were not trained).
- Acceptance is lower on short, accented crowd-sourced clips (Common Voice) and on long meetings.
- Trained for block size 16; smaller serving blocks lose more acceptance than for text DFlash drafts.

## License and attribution

The draft weights and the training code are released under Apache-2.0. The draft is derived from
the outputs and hidden states of Voxtral-Mini-3B-2507 (Apache-2.0, Mistral AI) and was trained on
the datasets listed above; see `NOTICE` for the required attributions and
`ATTRIBUTION_peoples_speech.csv` for per-source credits of The People's Speech.

Users must not attempt to identify speakers in Common Voice, LibriSpeech or Multilingual
LibriSpeech. No audio or speaker identifiers are distributed with this model.

## Citation

```bibtex
@article{chen2026dflash,
  title   = {DFlash: Block Diffusion for Flash Speculative Decoding},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```
