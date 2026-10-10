# SPDX-License-Identifier: Apache-2.0
"""Greedy labels for the long-form train corpus (W = 2..10 audio windows of 30 s) on Neuron.

Static-shape buckets keyed by W: prompt length P_W = 8 + 375 * W tokens (Voxtral transcription
template), decode cache sized P_W + max_new. Batch = 1 (4.3k clips), one process per core.
Output rows carry ``prompt_ids`` (the multi-window template) in addition to the usual columns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--check-every", type=int, default=32)
    args = ap.parse_args()
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
    dev = torch.device("neuron")
    dt = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(args.root).glob("*/shard-*.parquet"))[args.rank::args.world]
    hf = load_hf_voxtral(dt)
    tgt = VoxtralTarget(hf, max_pos=8 + 375 * 10 + args.max_new + 8).eval()
    del hf
    tgt = tgt.to(dev)
    enc = torch.compile(tgt.encode_audio, backend="neuron", dynamic=False)
    dec = torch.compile(tgt.decode, backend="neuron", dynamic=False)
    C = 512
    pchunk = torch.compile(tgt.prefill_chunk, backend="neuron", dynamic=False)
    S = 8 + 375 * 10 + C + args.max_new + 1           # one cache size for every W
    tgt.alloc_cache(1, S, dt, dev)
    log = open(out_dir / f"rank{args.rank}.log", "a")
    t0 = time.time()
    done = 0
    for sh in shards:
        dst = out_dir / (sh.parent.name + "__" + sh.stem + ".labels.parquet")
        if dst.exists():
            continue
        tbl = pq.read_table(sh)
        meta = tbl.drop(["audio_pcm16"]).to_pylist()
        audio = tbl.column("audio_pcm16").to_pylist()
        rows = []
        for i, m in enumerate(meta):
            a = np.frombuffer(audio[i], dtype="<i2").astype(np.float32) / 32768.0
            x = proc.apply_transcription_request(language="en", audio=a, sampling_rate=16000,
                                                 format=["wav"], model_id=TARGET_ID)
            ids = x["input_ids"]
            W = x["input_features"].shape[0]
            P = ids.shape[1]
            assert P == 8 + 375 * W, (P, W)
            span = tgt.audio_span(ids[0])
            with torch.no_grad():
                au = torch.cat([enc(x["input_features"][w:w + 1].to(dt).to(dev)) for w in range(W)], 0)
                emb = tgt.build_embeds(ids.to(dev), au.reshape(1, -1, tgt.H), span)
                n_ch = (P + C - 1) // C
                emb = torch.cat([emb, emb.new_zeros(1, n_ch * C - P, tgt.H)], 1)
                for c in range(n_ch):
                    am = pchunk(emb[:, c * C:(c + 1) * C].contiguous(), torch.arange(c * C, (c + 1) * C, device=dev))
                tok = am[:, (P - 1) - (n_ch - 1) * C].clone()  # fresh stride-1 tensor (a 1-elem slice keeps stride 512)
                toks = [tok]
                for t in range(args.max_new - 1):
                    tok = dec(tok, torch.tensor([P + t], device=dev))
                    toks.append(tok)
                    if (t + 2) % args.check_every == 0 and bool((torch.stack(toks).cpu() == EOS).any()):
                        break
                s = torch.stack(toks).flatten().cpu().tolist()
            fin = EOS in s
            s = s[: s.index(EOS) + 1] if fin else s
            hyp = proc.tokenizer.decode([v for v in s if v != EOS], skip_special_tokens=True)
            rows.append({**m, "prompt_len": P, "prompt_ids": ids[0].tolist(), "response_ids": s, "n_resp": len(s),
                         "finished": fin, "hyp_text": hyp})
            done += 1
            print(json.dumps({"rank": args.rank, "shard": sh.name, "i": i, "W": W, "n_resp": len(s), "fin": fin,
                              "done": done, "sec": round(time.time() - t0, 1)}), file=log, flush=True)
        pq.write_table(pa.Table.from_pylist(rows), str(dst) + ".tmp", compression="zstd")
        os.replace(str(dst) + ".tmp", dst)
    print(json.dumps({"rank": args.rank, "finished": True, "done": done}), file=log, flush=True)


if __name__ == "__main__":
    main()
