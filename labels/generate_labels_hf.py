# SPDX-License-Identifier: Apache-2.0
"""CPU / HF-transformers reference label generator.

For each clip: build the Voxtral transcription prompt from real audio, greedy-decode the
target's own transcript, then run one teacher-forced forward over prompt + response and
keep the residual-stream hidden states after the DFlash target layers.

Hidden-state convention (same as the public z-lab reference ``extract_context_feature``
and vLLM's aux-hidden-state capture): feature for target layer ``l`` is the output of
decoder layer ``l`` (HF ``hidden_states[l + 1]``), before the final norm.

This is the small-scale reference; generate_labels.py labels the full corpus on an accelerator.

Output: one ``.pt`` per shard with a list of records
``{clip_id, dataset, split, license, ids[int32 L], prompt_len, feats[bf16 L x n*H], ref_text, hyp_text}``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION, DraftConfig  # noqa: E402


def load_target(dtype=torch.bfloat16):
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    proc = AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
    model = VoxtralForConditionalGeneration.from_pretrained(TARGET_ID, revision=TARGET_REVISION, torch_dtype=dtype)
    model.eval()
    tc = model.config.text_config
    rope = getattr(tc, "rope_theta", None) or (getattr(tc, "rope_parameters", None) or {}).get("rope_theta")
    assert float(rope) == 1e8, f"unexpected target rope_theta {rope} (silent-config-drop check)"
    return proc, model


def build_inputs(proc, audio: np.ndarray, sr: int):
    inputs = proc.apply_transcription_request(language="en", audio=audio, sampling_rate=sr,
                                              format=["wav"], model_id=TARGET_ID)
    return inputs


@torch.no_grad()
def label_clip(proc, model, audio, sr, layer_ids, max_new_tokens=448):
    inputs = build_inputs(proc, audio, sr)
    inputs = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
    P = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    resp = out[0, P:]
    eos = model.config.text_config.eos_token_id if hasattr(model.config, "text_config") else 2
    eos = 2 if eos is None else eos
    # keep EOS as the final label (the draft should learn to predict end of transcript)
    hit = (resp == eos).nonzero()
    if hit.numel():
        resp = resp[: int(hit[0]) + 1]
    full = torch.cat([inputs["input_ids"][0], resp])[None]
    fwd = dict(inputs)
    fwd["input_ids"] = full
    fwd["attention_mask"] = torch.ones_like(full)
    o = model(**fwd, output_hidden_states=True, use_cache=False)
    hs = o.hidden_states
    feats = torch.cat([hs[l + 1][0] for l in layer_ids], dim=-1).to(torch.bfloat16)
    text = proc.tokenizer.decode(resp.tolist(), skip_special_tokens=True)
    return full[0].to(torch.int32), P, feats, text


def iter_librispeech_parquet(path: str, n: int, max_dur: float = 30.0):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["id", "audio", "text", "speaker_id"])
    rows = t.to_pylist()
    k = 0
    for r in rows:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if len(a) / sr > max_dur:
            continue
        yield r["id"], a, sr, r["text"]
        k += 1
        if k >= n:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--dataset", default="openslr/librispeech_asr")
    ap.add_argument("--split", default="train.clean.100")
    ap.add_argument("--license", default="CC-BY-4.0")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = DraftConfig()
    proc, model = load_target()
    recs = []
    t0 = time.time()
    for cid, a, sr, ref in iter_librispeech_parquet(args.parquet, args.n):
        ids, P, feats, hyp = label_clip(proc, model, a, sr, cfg.target_layer_ids)
        recs.append(dict(clip_id=cid, dataset=args.dataset, split=args.split, license=args.license,
                         duration_s=len(a) / sr, ids=ids, prompt_len=P, feats=feats,
                         ref_text=ref, hyp_text=hyp))
        print(f"[{len(recs)}/{args.n}] {cid} dur={len(a)/sr:.1f}s P={P} R={ids.numel()-P} "
              f"t={time.time()-t0:.0f}s | {hyp[:80]}", flush=True)
    torch.save(dict(records=recs, target_layer_ids=cfg.target_layer_ids, target=TARGET_ID,
                    target_revision=TARGET_REVISION), args.out)
    meta = [{k: v for k, v in r.items() if k not in ("ids", "feats")} | {"len": int(r["ids"].numel())} for r in recs]
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
