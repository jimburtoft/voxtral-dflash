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

trn2.3xlarge, tensor parallel 4, BS=1 (one request at a time), greedy, `/v1/audio/transcriptions`,
50 random held-out clips per set (<= 30 s) and 60 AMI long-form windows (30 s - 5 min). Environment:
the public Neuron SDK 2.33 DLAMI vLLM venv (vllm 0.29.0, vllm-neuron 0.29.0.1.2.0) with the open
vllm-neuron Voxtral pull request
([vllm-project/vllm-neuron#69](https://github.com/vllm-project/vllm-neuron/pull/69)) added on top,
because the released vllm-neuron does not include Voxtral yet. Benchmark script and exact steps:
code repository README, "Serving benchmark".

| set | tau | latency without draft (mean ms) | with draft (mean ms) | speedup | WER without / with draft |
|---|---|---|---|---|---|
| LibriSpeech test-clean | 8.47 | 249 | 93 | 2.67x | 2.29 / 2.29 % |
| LibriSpeech test-other | 7.08 | 222 | 92 | 2.40x | 4.45 / 4.45 % |
| VoxPopuli test | 8.50 | 247 | 92 | 2.68x | 8.22 / 8.22 % |
| FLEURS en_us test | 6.42 | 265 | 108 | 2.46x | 4.10 / 4.18 % |
| AMI ihm test | 6.02 | 140 | 76 | 1.84x | 16.84 / 17.91 % |
| Common Voice 17 test | 4.17 | 136 | 87 | 1.56x | 11.53 / 11.95 % |
| **all short (300 clips)** | **6.84** | **210** | **91** | **2.30x** | **6.55 / 6.71 %** |
| AMI long-form (60 windows, 30-300 s) | 3.78 | 4,992 | 2,487 | 2.01x | 34.59 / 34.55 % |

WER uses a simple lowercase / punctuation-stripping normaliser against the dataset references
(AMI references are unpunctuated, which inflates its absolute WER). Greedy speculative decoding
keeps the target's output; 290 of 300 short transcripts are identical to the no-draft run. The
other 10 differ through bf16 rounding in the multi-token verify graph (one flipped token changes
the rest of a greedy decode), and on those clips the draft run matches the target's independent
PyTorch greedy transcript more often (6) than the no-draft server (2). Most of the small short-clip
WER difference comes from 2 AMI clips of a few words.

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
