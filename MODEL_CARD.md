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

### End-to-end serving (pre-release vllm-neuron build, Trainium2)

trn2.3xlarge, tensor parallel 4, BS=1, `/v1/audio/transcriptions`, 50 clips per set (short) and 60
AMI long-form windows. These numbers come from a pre-release vllm-neuron build with Voxtral
support and a local fix for a DFlash attention-cache issue in that build; treat them as
indicative until vllm-neuron Voxtral support is released upstream.

| set | tau | latency without draft (mean ms) | with draft (mean ms) | speedup | WER without / with draft |
|---|---|---|---|---|---|
| LibriSpeech test-clean | 8.47 | 250 | 100 | 2.50x | 2.37 / 2.29 % |
| LibriSpeech test-other | 7.05 | 224 | 102 | 2.20x | 4.64 / 4.26 % |
| VoxPopuli test | 8.40 | 249 | 101 | 2.48x | 8.22 / 8.22 % |
| FLEURS en_us test | 6.44 | 267 | 118 | 2.27x | 4.10 / 4.10 % |
| AMI ihm test | 6.05 | 144 | 83 | 1.73x | 16.84 / 16.13 % |
| Common Voice 17 test | 4.09 | 139 | 95 | 1.46x | 11.74 / 11.74 % |
| **all short (300 clips)** | **6.81** | **212** | **100** | **2.13x** | **6.62 / 6.46 %** |
| AMI long-form (60 windows, 30-300 s) | 3.79 | 4,887 | 3,231 | 1.51x | 34.52 / 34.51 % |

WER uses a simple lowercase / punctuation-stripping normaliser against the dataset references
(AMI references are unpunctuated, which inflates its absolute WER). Speculative decoding does not
change the greedy output distribution; the small number of transcripts that differ from the
no-draft run come from bf16 numerics of the multi-token verify graph, and WER is unchanged.

Serving block-size sweep (short clips, same weights): block 16: tau 6.81 / 99.7 ms; 14: 4.06 /
120.7 ms; 12: 3.98 / 122.3 ms; 8: 3.33 / 131.2 ms.

## Usage

### vLLM (DFlash speculative decoding)

```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer-mode mistral \
  --speculative-config '{"method": "dflash", "model": "<this repo>", "num_speculative_tokens": 15}'
```

The draft needs the target's auxiliary hidden states (`target_layer_ids + 1` capture layers).
The loader format is verified against upstream vLLM 0.31 (CPU) with Voxtral's text decoder:
config, weights and draft tokens match the reference implementation. Voxtral's multimodal model
class in upstream vLLM 0.31 does not expose auxiliary hidden states yet, so end-to-end audio
serving there needs that support first. vllm-neuron (Trainium) support for Voxtral is pending
upstream.

### Reference implementation (PyTorch)

The training code repository contains `model/dflash.py` (the draft), `model/release.py`
(load/save), and `eval/acceptance_eval.py` (offline acceptance). The checkpoint also loads with the
public `z-lab/dflash` reference modelling code (identical block outputs to 2e-5).

## Training

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
