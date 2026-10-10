# SPDX-License-Identifier: Apache-2.0
"""Shared data helpers for DFlash training on Voxtral labels.

* ``LabelData``: labelled short clips (target greedy responses) joined with their audio.
* ``MelPrep``: Voxtral mel features + the fixed 1-window transcription prompt.
* ``build_cpu_batch``: pack a batch of clips into prompt+response ids and DFlash training blocks.
* ``eval_items``: exhaustive-anchor held-out evaluation items.
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import TARGET_ID, TARGET_REVISION, DraftConfig  # noqa: E402
from train.dflash_objective import pack_batch  # noqa: E402


def _frac_hash(clip_id: str) -> float:
    import hashlib
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16) / 2**32


class LabelData:
    """Labelled clips (finished, 2 <= n_resp <= r_max) with their audio held in RAM.

    Rank r owns the label files whose (sorted) index % world == r and loads only their audio
    (~1/world of the corpus, ~12 GB per rank for the full train set). Gradients are averaged over
    ranks every step, so each global batch mixes all ranks' shards. Steps per epoch are equalised
    across ranks with ``min_rows`` (all-reduced by the caller).
    """

    def __init__(self, label_glob: str, audio_root: str, r_max: int, rank: int, world: int, seed: int,
                 load_audio: bool = True, subset_frac: float = 1.0):
        files = sorted(glob.glob(label_glob))
        assert files, f"no label files match {label_glob}"
        files = files[rank::world]
        self.rows = []
        for f in files:
            t = pq.read_table(f)
            if t.num_rows == 0:
                continue
            t = t.select(["clip_id", "dataset", "split", "duration_s", "prompt_len", "response_ids",
                          "n_resp", "finished"]).to_pylist()
            shard = Path(f).name.replace(".labels.parquet", "").replace("__", "/") + ".parquet"
            keep = {r["clip_id"]: r for r in t if r["finished"] and 2 <= r["n_resp"] <= r_max
                    and (subset_frac >= 1.0 or _frac_hash(r["clip_id"]) < subset_frac)}
            if load_audio and keep:
                a = pq.read_table(Path(audio_root) / shard, columns=["clip_id", "audio_pcm16"])
                for cid, pcm in zip(a.column("clip_id").to_pylist(), a.column("audio_pcm16").to_pylist()):
                    if cid in keep:
                        keep[cid]["audio_pcm16"] = pcm
            self.rows.extend(keep.values())
        self.rank, self.world, self.seed = rank, world, seed

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def audio(r) -> bytes:
        return r["audio_pcm16"]

    def epoch_batches(self, epoch: int, batch: int, n_batches: int):
        idx = list(range(len(self.rows)))
        random.Random(self.seed * 7919 + 1000 * epoch + self.rank).shuffle(idx)
        for k in range(n_batches):
            yield [self.rows[i] for i in idx[k * batch:(k + 1) * batch]]


class MelPrep:
    def __init__(self):
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
        x = self.proc.apply_transcription_request(language="en", audio=np.zeros(16000, np.float32),
                                                  sampling_rate=16000, format=["wav"], model_id=TARGET_ID)
        self.prompt_ids = x["input_ids"][0].clone()

    def mel(self, pcm16: bytes) -> torch.Tensor:
        a = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        x = self.proc.apply_transcription_request(language="en", audio=a, sampling_rate=16000,
                                                  format=["wav"], model_id=TARGET_ID)
        assert torch.equal(x["input_ids"][0], self.prompt_ids)
        return x["input_features"][0]


# ----------------------------------------------------------------------------- batch build
def build_cpu_batch(rows, prompt_ids, r_max, n_blocks, cfg: DraftConfig, gen, anchors_override=None):
    P = int(prompt_ids.numel())
    Sc = P + r_max
    B = len(rows)
    ids = torch.full((B, Sc), cfg.pad_token_id, dtype=torch.long)
    valid = torch.zeros(B, Sc, dtype=torch.bool)
    seqs = []
    for b, r in enumerate(rows):
        resp = torch.tensor(r["response_ids"], dtype=torch.long)
        L = P + resp.numel()
        ids[b, :P] = prompt_ids
        ids[b, P:L] = resp
        valid[b, :L] = True
        seqs.append(ids[b, :L])
    pb = pack_batch(seqs, [P] * B, n_blocks, cfg.block_size, cfg.mask_token_id, generator=gen,
                    anchors_override=anchors_override)
    # pad context axis of the mask to the fixed Sc (pack_batch sizes it to the longest seq)
    sc_now = pb.mask.shape[-1] - pb.block_ids.shape[1]
    if sc_now < Sc:
        Sq = pb.block_ids.shape[1]
        m = torch.zeros(B, Sq, Sc + Sq, dtype=torch.bool)
        m[:, :, :sc_now] = pb.mask[:, :, :sc_now]
        m[:, :, Sc:] = pb.mask[:, :, sc_now:]
        pb.mask = m
        pb.ctx_pos = torch.arange(Sc).unsqueeze(0).expand(B, Sc).clone()
    return ids, valid, pb


def eval_items(rows, prompt_len, n_blocks, per_set: int, seed: int = 0):
    """(row, anchors) items: every response position is an anchor (exhaustive), chunked by n_blocks.
    At most ``per_set`` clips per (dataset, split)."""
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["split"]), []).append(r)
    items = []
    names = sorted(by)
    for si, k in enumerate(names):
        rs = by[k]
        random.Random(seed).shuffle(rs)
        for r in rs[:per_set]:
            L = prompt_len + r["n_resp"]
            a = torch.arange(prompt_len, L - 1)
            for c in range(0, a.numel(), n_blocks):
                items.append((si, r, a[c:c + n_blocks]))
    return names, items
