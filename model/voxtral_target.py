# SPDX-License-Identifier: Apache-2.0
"""Static-shape Voxtral-Mini-3B target (plain PyTorch: Trainium, CUDA or CPU).

Reuses the Hugging Face audio encoder + projector modules (frozen) and re-implements the
30-layer Llama text decoder with fixed-shape tensors so that ``torch.compile`` (``backend="neuron"``
on Trainium) sees one graph per bucket:

* ``encode_audio(mel[W,128,3000]) -> [W, 375, 3072]`` audio embeddings
* ``build_embeds(ids[B,P], audio[B, 375*n, H])`` -> prompt embeddings (audio spliced in at the
  contiguous ``[AUDIO]`` run, checked against the ids)
* ``forward_full(embeds, layer_ids)`` -> (final-norm hidden, stacked target-layer outputs) with
  causal attention -- used for online DFlash features and for teacher-forced checks
* ``prefill(embeds)`` / ``decode(tok, pos)`` -> static KV-cache greedy decoding for labels

Hidden-state convention: feature for layer ``l`` = output of decoder layer ``l`` (residual
stream, pre final norm) == HF ``hidden_states[l + 1]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

AUDIO_TOKEN_ID = 24
TOKENS_PER_WINDOW = 375


def _rope_tables(max_pos: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float64)
    f = torch.outer(t, inv)
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().float(), emb.sin().float()


def _rot_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class DecoderLayer(nn.Module):
    def __init__(self, H, I, nh, nkv, hd, eps):
        super().__init__()
        self.nh, self.nkv, self.hd, self.eps, self.H = nh, nkv, hd, eps, H
        self.ln1 = nn.Parameter(torch.ones(H), requires_grad=False)
        self.ln2 = nn.Parameter(torch.ones(H), requires_grad=False)
        self.wqkv = nn.Parameter(torch.empty((nh + 2 * nkv) * hd, H), requires_grad=False)
        self.wo = nn.Parameter(torch.empty(H, nh * hd), requires_grad=False)
        self.wgu = nn.Parameter(torch.empty(2 * I, H), requires_grad=False)
        self.wd = nn.Parameter(torch.empty(H, I), requires_grad=False)
        self.I = I

    def _qkv(self, x, cos, sin):
        B, S, _ = x.shape
        h = torch.rms_norm(x, (self.H,), self.ln1, self.eps)
        qkv = F.linear(h, self.wqkv)
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
        c = cos[:, None]
        s = sin[:, None]
        q = q * c + _rot_half(q) * s
        k = k * c + _rot_half(k) * s
        return q, k, v

    def _mlp(self, x):
        h = torch.rms_norm(x, (self.H,), self.ln2, self.eps)
        g, u = F.linear(h, self.wgu).chunk(2, dim=-1)
        return x + F.linear(F.silu(g) * u, self.wd)

    def _attn_out(self, x, q, k, v, mask):
        B, S = x.shape[:2]
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / math.sqrt(self.hd))
        o = o.transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return x + F.linear(o, self.wo)

    def forward_full(self, x, cos, sin, mask):
        q, k, v = self._qkv(x, cos, sin)
        x = self._attn_out(x, q, k, v, mask)
        return self._mlp(x), k, v

    def forward_cached(self, x, cos, sin, mask, kc, vc, pos_idx):
        """x [B,1,H]; kc/vc [B,nkv,S,hd] buffers updated in place at pos_idx (1-D, len 1)."""
        q, k, v = self._qkv(x, cos, sin)
        kc.index_copy_(2, pos_idx, k)
        vc.index_copy_(2, pos_idx, v)
        x = self._attn_out(x, q, kc, vc, mask)
        return self._mlp(x)


class VoxtralTarget(nn.Module):
    def __init__(self, hf_model, max_pos: int = 8192):
        super().__init__()
        tc = hf_model.config.text_config
        rope = getattr(tc, "rope_theta", None) or (getattr(tc, "rope_parameters", None) or {}).get("rope_theta")
        assert float(rope) == 1e8, f"target rope_theta {rope}"
        self.H, self.nl = tc.hidden_size, tc.num_hidden_layers
        self.nh, self.nkv = tc.num_attention_heads, tc.num_key_value_heads
        self.hd = getattr(tc, "head_dim", None) or self.H // self.nh
        self.eps = tc.rms_norm_eps
        # transformers 4.x: hf.{audio_tower, multi_modal_projector, language_model.{model,lm_head}}
        # transformers 5.x: hf.model.{audio_tower, multi_modal_projector, language_model(LlamaModel)}, hf.lm_head
        base = hf_model.model if hasattr(hf_model, "model") and hasattr(hf_model.model, "audio_tower") else hf_model
        self.audio_tower = base.audio_tower
        self.projector = base.multi_modal_projector
        self.audio_inter = hf_model.config.audio_config.intermediate_size
        lm = base.language_model
        dec = lm.model if hasattr(lm, "model") else lm
        lm_head = hf_model.lm_head if hasattr(hf_model, "lm_head") else lm.lm_head
        self.embed = nn.Parameter(dec.embed_tokens.weight.detach().clone(), requires_grad=False)
        self.norm = nn.Parameter(dec.norm.weight.detach().clone(), requires_grad=False)
        self.lm_head = nn.Parameter(lm_head.weight.detach().clone(), requires_grad=False)
        self.layers = nn.ModuleList()
        for L in dec.layers:
            a, m = L.self_attn, L.mlp
            d = DecoderLayer(self.H, m.gate_proj.weight.shape[0], self.nh, self.nkv, self.hd, self.eps)
            with torch.no_grad():
                d.ln1.copy_(L.input_layernorm.weight)
                d.ln2.copy_(L.post_attention_layernorm.weight)
                d.wqkv.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0))
                d.wo.copy_(a.o_proj.weight)
                d.wgu.copy_(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0))
                d.wd.copy_(m.down_proj.weight)
            self.layers.append(d)
        cos, sin = _rope_tables(max_pos, self.hd, float(rope))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(self.embed.dtype)

    # ---------------- audio ----------------
    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [N,128,3000] (N = total windows in the batch) -> [N, 375, H]"""
        h = self.audio_tower(mel).last_hidden_state                    # [W, 1500, 1280]
        W = h.shape[0]
        h = h.reshape(W * h.shape[1] * h.shape[2] // self.audio_inter, self.audio_inter)
        return self.projector(h).view(W, TOKENS_PER_WINDOW, self.H)

    @staticmethod
    def audio_span(ids_row: torch.Tensor) -> tuple[int, int]:
        pos = (ids_row == AUDIO_TOKEN_ID).nonzero().flatten()
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == pos.numel() and (e - s) % TOKENS_PER_WINDOW == 0, "audio tokens not contiguous"
        return s, e

    def build_embeds(self, ids: torch.Tensor, audio: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        """ids [B,L] (all rows share the audio span), audio [B, e-s, H]."""
        s, e = span
        emb = F.embedding(ids, self.embed)
        return torch.cat([emb[:, :s], audio.to(emb.dtype), emb[:, e:]], dim=1)

    # ---------------- full (teacher-forced) ----------------
    def _cos_sin(self, S, dtype):
        return self.cos[:S].to(dtype)[None], self.sin[:S].to(dtype)[None]

    def forward_full(self, x: torch.Tensor, layer_ids: list[int], key_valid: torch.Tensor | None = None,
                     want_logits: bool = False):
        """x [B,S,H] embeddings. Returns (feats [B,S,len(layer_ids)*H], logits or None).

        Causal attention; key_valid [B,S] bool masks right padding (pads never attended)."""
        B, S, _ = x.shape
        cos, sin = self._cos_sin(S, x.dtype)
        causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None]
        if key_valid is not None:
            mask = mask & key_valid[:, None, None, :]
            mask = mask | torch.eye(S, dtype=torch.bool, device=x.device)[None, None]
        want = set(layer_ids)
        last = max(layer_ids) if not want_logits else self.nl - 1
        outs = {}
        for i, layer in enumerate(self.layers):
            if i > last:
                break
            x, _, _ = layer.forward_full(x, cos, sin, mask)
            if i in want:
                outs[i] = x
        feats = torch.cat([outs[i] for i in layer_ids], dim=-1)
        logits = None
        if want_logits:
            logits = F.linear(torch.rms_norm(x, (self.H,), self.norm, self.eps), self.lm_head)
        return feats, logits

    # ---------------- static KV-cache greedy decoding ----------------
    def alloc_cache(self, B: int, S: int, dtype, device):
        self.kc = [torch.zeros(B, self.nkv, S, self.hd, dtype=dtype, device=device) for _ in self.layers]
        self.vc = [torch.zeros(B, self.nkv, S, self.hd, dtype=dtype, device=device) for _ in self.layers]
        self.S_cache = S

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,P,H] (no padding: all rows share P). Fills cache[:, :, :P]. Returns argmax [B]."""
        B, P, _ = x.shape
        cos, sin = self._cos_sin(P, x.dtype)
        mask = torch.ones(P, P, dtype=torch.bool, device=x.device).tril()[None, None]
        idx = torch.arange(P, device=x.device)
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(x, cos, sin)
            self.kc[i].index_copy_(2, idx, k)
            self.vc[i].index_copy_(2, idx, v)
            x = layer._attn_out(x, q, k, v, mask)
            x = layer._mlp(x)
        h = torch.rms_norm(x[:, -1:], (self.H,), self.norm, self.eps)
        return F.linear(h, self.lm_head)[:, 0].argmax(-1)

    def prefill_chunk(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Chunked prefill against the static cache (B rows share positions).

        x [B,C,H] embeddings for positions ``pos`` [C] (long, contiguous). Writes K/V at ``pos``,
        attends causally to everything cached at <= pos. One graph for every prompt length (no
        P x P mask for multi-window prompts of up to ~4k tokens). Returns argmax at every chunk position
        [B, C]; for a right-padded last chunk the caller reads the last real position. Padded
        positions write K/V beyond the prompt; decode overwrites them before they are attended to
        (decode at position p only attends to <= p, and writes p first).
        """
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        for i, layer in enumerate(self.layers):
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            h = layer._attn_out(h, q, self.kc[i], self.vc[i], mask)
            h = layer._mlp(h)
        hn = torch.rms_norm(h, (self.H,), self.norm, self.eps)
        return F.linear(hn, self.lm_head).argmax(-1)          # [B, C]

    def feats_chunk(self, x: torch.Tensor, pos: torch.Tensor, layer_ids: list[int]) -> torch.Tensor:
        """Teacher-forced features for one chunk against the static cache (long-form training).

        Same attention as ``prefill_chunk``; returns the outputs of decoder layers ``layer_ids``
        concatenated on the last dim [B, C, len(layer_ids) * H]. Layers after max(layer_ids) are
        skipped (their K/V are not needed by later chunks either: no later layer reads them).
        """
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar[None, :] <= pos[:, None])[None, None]
        h = x
        want = set(layer_ids)
        outs = {}
        for i, layer in enumerate(self.layers):
            if i > max(layer_ids):
                break
            q, k, v = layer._qkv(h, cos, sin)
            self.kc[i].index_copy_(2, pos, k)
            self.vc[i].index_copy_(2, pos, v)
            h = layer._attn_out(h, q, self.kc[i], self.vc[i], mask)
            h = layer._mlp(h)
            if i in want:
                outs[i] = h
        return torch.cat([outs[i] for i in layer_ids], dim=-1)

    def decode(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """tok [B] long, pos [1] long (same position for every row). Returns argmax [B]."""
        x = F.embedding(tok.view(-1, 1), self.embed)
        cos = F.embedding(pos, self.cos).to(x.dtype)[None]           # [1,1,hd]
        sin = F.embedding(pos, self.sin).to(x.dtype)[None]
        ar = torch.arange(self.S_cache, device=x.device)
        mask = (ar <= pos)[None, None, None, :]
        for i, layer in enumerate(self.layers):
            x = layer.forward_cached(x, cos, sin, mask, self.kc[i], self.vc[i], pos)
        h = torch.rms_norm(x, (self.H,), self.norm, self.eps)
        return F.linear(h, self.lm_head)[:, 0].argmax(-1)


def load_hf_voxtral(dtype=torch.bfloat16):
    from transformers import VoxtralForConditionalGeneration
    from .config import TARGET_ID, TARGET_REVISION
    m = VoxtralForConditionalGeneration.from_pretrained(TARGET_ID, revision=TARGET_REVISION, dtype=dtype,
                                                        attn_implementation="eager")
    return m.eval()
