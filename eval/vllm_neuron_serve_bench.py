# SPDX-License-Identifier: Apache-2.0
"""End-to-end Voxtral (+ optional DFlash) on vllm-neuron through the server's
``/v1/audio/transcriptions`` endpoint, real audio, BS=1 (one request at a time).

Starts ``vllm serve`` with Voxtral serving settings for vllm-neuron (requires a vllm-neuron build
with Voxtral support; see the README), sends each clip as a WAV file,
records wall latency, the transcript, and the server's spec-decode counters (Prometheus
``/metrics``) before/after, so tau (accepted tokens per draft + 1) is measured by the server.

Usage (vllm-neuron venv; needs ``pyarrow`` and ``requests``):
    python eval/vllm_neuron_serve_bench.py --clips "$VXDF_CORPUS/shards/eval/*/*/shard-*.parquet" \
        --out run.json [--draft <draft repo or dir> --k 15] [--per-set 50] [--min-dur 0 --max-dur 30]
    # long-form: --min-dur 30 --max-dur 300 --max-model-len 8192 --enc-buckets 1,4,10
    #            (and VLLM_NEURON_RESERVED_HEADROOM_GIB=8)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import wave

import numpy as np
import pyarrow.parquet as pq
import requests

TARGET = "mistralai/Voxtral-Mini-3B-2507"
REV = "3060fe34b35ba5d44202ce9ff3c097642914f8f3"


def wav_bytes(pcm16: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16)
    return b.getvalue()


def load_clips(pattern, per_set, min_dur, max_dur):
    by = {}
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f, columns=["clip_id", "dataset", "split", "duration_s", "ref_text", "audio_pcm16"])
        for r in t.to_pylist():
            if min_dur <= r["duration_s"] <= max_dur:
                by.setdefault((r["dataset"], r["split"]), []).append(r)
    out = []
    for k in sorted(by):
        rs = sorted(by[k], key=lambda r: r["clip_id"])
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rs))[:per_set]
        out.extend(rs[i] for i in sorted(idx))
    return out


def metrics(base):
    txt = requests.get(base + "/metrics", timeout=30).text
    vals = {}
    for name in ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens",
                 "vllm:spec_decode_num_draft_tokens", "vllm:generation_tokens"):
        tot = 0.0
        for m in re.finditer(r"^" + re.escape(name) + r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", txt, re.M):
            tot += float(m.group(1))
        vals[name] = tot
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--per-set", type=int, default=50)
    ap.add_argument("--min-dur", type=float, default=0.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--draft", default="")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--enc-buckets", default="1,4")
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default="")
    args = ap.parse_args()

    enc_b = [int(x) for x in args.enc_buckets.split(",")]
    add = {"neuron_config": {"num_batched_tokens_buckets": [512, args.max_model_len], "num_seqs_buckets": [1],
                             "on_device_sampling_config": {"all_greedy": True}},
           "encoder_neuron_config": {"num_encoder_input_buckets": enc_b, "encoder_attention_block_size": 375,
                                     "tp_size": args.tp}}
    cmd = ["vllm", "serve", TARGET, "--revision", REV, "--tokenizer-mode", "mistral",
           "--tensor-parallel-size", str(args.tp), "--dtype", "bfloat16", "--max-model-len", str(args.max_model_len),
           "--max-num-seqs", "1", "--max-num-batched-tokens", str(args.max_model_len),
           "--limit-mm-per-prompt", json.dumps({"audio": 1}), "--no-enable-prefix-caching",
           "--mm-processor-cache-gb", "0", "--no-enable-log-requests", "--port", str(args.port),
           "--additional-config", json.dumps(add)]
    if args.draft:
        cmd += ["--speculative-config", json.dumps({"method": "dflash", "model": args.draft,
                                                     "num_speculative_tokens": args.k})]
    # trn2.3xlarge has no EFA; the plugin's EFA CPU-affinity probe is an optimisation only.
    env = dict(os.environ, VLLM_RPC_TIMEOUT="100000", NEURON_SKIP_EFA_AFFINITY="1")
    log = open(args.server_log or (args.out + ".server.log"), "w")
    print("CMD " + " ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{args.port}"
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"server exited with {srv.returncode}; see log")
            try:
                if requests.get(base + "/health", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > 7200:
                raise TimeoutError("server start")
            time.sleep(10)
        startup_s = time.time() - t0
        clips = load_clips(args.clips, args.per_set, args.min_dur, args.max_dur)

        def transcribe(c):
            files = {"file": ("a.wav", wav_bytes(c["audio_pcm16"]), "audio/wav")}
            data = {"model": TARGET, "language": "en", "temperature": "0.0", "response_format": "json"}
            t = time.perf_counter()
            r = requests.post(base + "/v1/audio/transcriptions", files=files, data=data, timeout=1800)
            dt = time.perf_counter() - t
            r.raise_for_status()
            return r.json().get("text", ""), dt

        for c in clips[: args.warmup]:
            transcribe(c)
        m0 = metrics(base)
        res = []
        for c in clips:
            mb = metrics(base)
            text, dt = transcribe(c)
            ma = metrics(base)
            res.append({"clip_id": c["clip_id"], "dataset": c["dataset"], "split": c["split"],
                        "duration_s": c["duration_s"], "latency_s": dt, "text": text, "ref_text": c["ref_text"],
                        "gen_tokens": ma["vllm:generation_tokens"] - mb["vllm:generation_tokens"],
                        "drafts": ma["vllm:spec_decode_num_drafts"] - mb["vllm:spec_decode_num_drafts"],
                        "accepted": ma["vllm:spec_decode_num_accepted_tokens"] - mb["vllm:spec_decode_num_accepted_tokens"]})
        m1 = metrics(base)
        lat = np.array([r["latency_s"] for r in res])
        summ = {"n": len(res), "draft": args.draft, "k": args.k, "startup_s": round(startup_s, 1),
                "mean_ms": round(1000 * lat.mean(), 1), "median_ms": round(1000 * float(np.median(lat)), 1),
                "p90_ms": round(1000 * float(np.percentile(lat, 90)), 1)}
        d = m1["vllm:spec_decode_num_drafts"] - m0["vllm:spec_decode_num_drafts"]
        if d > 0:
            a = m1["vllm:spec_decode_num_accepted_tokens"] - m0["vllm:spec_decode_num_accepted_tokens"]
            summ["accepted_per_draft"] = round(a / d, 4)
            summ["tau"] = round(1 + a / d, 4)
        print("SUMMARY " + json.dumps(summ), flush=True)
        json.dump({"summary": summ, "cmd": cmd, "clips": res}, open(args.out, "w"))
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(timeout=120)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    main()
