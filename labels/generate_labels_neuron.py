# SPDX-License-Identifier: Apache-2.0
"""Greedy transcript labels for the corpus on Neuron (PyTorch Native, static shapes).

One process per NeuronCore (``NEURON_RT_VISIBLE_CORES=<i>``); each takes every ``world``-th
shard. Per batch of B clips (all <= 30 s -> one audio window -> identical 383-token prompt):

    mel (CPU, HF processor) -> encode_audio -> prefill (B x 383) -> up to MAX_NEW decode steps

Rows that never emit EOS within MAX_NEW are kept but flagged ``finished=False`` (the trainer
drops them). Output: one parquet per input shard under ``--out``.

Usage (inside the PyTorch Native container):
    NEURON_RT_VISIBLE_CORES=0 python labels/generate_labels_neuron.py --rank 0 --world 4 \
        --manifest <corpus>/shards/train_manifest.parquet --root <corpus>/shards \
        --out <labels>/train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION  # noqa: E402
from model.voxtral_target import VoxtralTarget, load_hf_voxtral  # noqa: E402

EOS = 2


class Prep:
    def __init__(self):
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
        x = self.proc.apply_transcription_request(language="en", audio=np.zeros(16000, np.float32),
                                                  sampling_rate=16000, format=["wav"], model_id=TARGET_ID)
        self.prompt_ids = x["input_ids"][0].clone()

    def mel(self, pcm16: bytes, n_windows: int = 1):
        """Returns ([n_windows,128,3000] mel, prompt ids) and checks the window count."""
        a = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        x = self.proc.apply_transcription_request(language="en", audio=a, sampling_rate=16000,
                                                  format=["wav"], model_id=TARGET_ID)
        feats = x["input_features"]
        if n_windows == 1:
            assert torch.equal(x["input_ids"][0], self.prompt_ids), "prompt differs (clip > 30 s?)"
            return feats[0]
        assert feats.shape[0] == n_windows, (feats.shape, n_windows)
        return feats, x["input_ids"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    dev = torch.device("neuron")
    dt = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = pq.read_table(args.manifest, columns=["shard"])
    shards = sorted(set(man.column("shard").to_pylist()))[args.rank::args.world]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    prep = Prep()
    P = int(prep.prompt_ids.numel())
    hf = load_hf_voxtral(dt)
    tgt = VoxtralTarget(hf, max_pos=P + args.max_new + 8).eval()
    del hf
    tgt = tgt.to(dev)
    B, S = args.batch, P + args.max_new + 1
    tgt.alloc_cache(B, S, dt, dev)
    span = tgt.audio_span(prep.prompt_ids)
    prompt_dev = prep.prompt_ids[None].expand(B, P).contiguous().to(dev)

    def _prefill(audio):
        emb = tgt.build_embeds(prompt_dev, audio.reshape(B, -1, tgt.H), span)
        return tgt.prefill(emb)

    # Encoder and prefill are compiled as separate graphs: fused, neuronx-cc rejects the
    # graph with NCC_EVRF009 (it counts ~61 GB of I/O for a ~10 GB signature); separately both compile.
    if args.no_compile:
        encode, prefill = tgt.encode_audio, _prefill
    else:
        encode = torch.compile(tgt.encode_audio, backend="neuron", dynamic=False)
        prefill = torch.compile(_prefill, backend="neuron", dynamic=False)

    def enc_prefill(mel):
        return prefill(encode(mel))
    decode = tgt.decode if args.no_compile else torch.compile(tgt.decode, backend="neuron", dynamic=False)
    pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // max(1, args.world)))
    log = open(out_dir / f"rank{args.rank}.log", "a")
    t_start = time.time()
    n_done = 0
    for si, shard in enumerate(shards):
        dst = out_dir / (shard.replace("/", "__").replace(".parquet", "") + ".labels.parquet")
        if dst.exists():
            continue
        tbl = pq.read_table(Path(args.root) / shard)
        meta = tbl.drop(["audio_pcm16"]).to_pylist()
        audio = tbl.column("audio_pcm16").to_pylist()
        rows_out = []
        for b0 in range(0, len(meta), B):
            idx = list(range(b0, min(b0 + B, len(meta))))
            idx = [i for i in idx if meta[i]["duration_s"] <= args.max_dur]
            if not idx:
                continue
            mels = list(pool.map(prep.mel, [audio[i] for i in idx]))
            nreal = len(mels)
            while len(mels) < B:
                mels.append(mels[0])
            mel = torch.stack(mels).to(dt).to(dev)
            t0 = time.time()
            with torch.no_grad():
                tok = enc_prefill(mel)
                toks = [tok]
                for t in range(args.max_new - 1):
                    tok = decode(tok, torch.tensor([P + t], device=dev))
                    toks.append(tok)
                    if (t + 2) % args.check_every == 0:
                        seq = torch.stack(toks, 1)[:nreal].cpu()
                        if bool((seq == EOS).any(1).all()):
                            break
                seq = torch.stack(toks, 1)[:nreal].cpu()
            dt_b = time.time() - t0
            for j, i in enumerate(idx):
                s = seq[j].tolist()
                fin = EOS in s
                s = s[: s.index(EOS) + 1] if fin else s
                hyp = prep.proc.tokenizer.decode([x for x in s if x != EOS], skip_special_tokens=True)
                rows_out.append({**meta[i], "prompt_len": P, "response_ids": s, "n_resp": len(s),
                                 "finished": fin, "hyp_text": hyp})
            n_done += nreal
            el = time.time() - t_start
            print(json.dumps({"rank": args.rank, "shard": si, "of": len(shards), "batch_s": round(dt_b, 2),
                              "steps": len(toks), "done": n_done, "clips_per_s": round(n_done / el, 2)}),
                  file=log, flush=True)
        pt = pa.Table.from_pylist(rows_out)
        pq.write_table(pt, str(dst) + ".tmp", compression="zstd")
        os.replace(str(dst) + ".tmp", dst)
    print(json.dumps({"rank": args.rank, "finished": True, "done": n_done, "sec": time.time() - t_start}),
          file=log, flush=True)


if __name__ == "__main__":
    main()
