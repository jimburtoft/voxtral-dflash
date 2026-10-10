# SPDX-License-Identifier: Apache-2.0
"""Format check of a release DFlash draft against upstream vLLM and the public z-lab reference.

This is a FORMAT / loader-equivalence check, not an acceptance measurement: the prompt is plain
text fed to Voxtral's text decoder (no audio). Acceptance numbers must come from real audio.

Two halves:

``vllm`` (run inside the upstream ``vllm/vllm-openai-cpu`` image, repo mounted at /work)::

    python3 /work/eval/upstream_format_check.py vllm --spec 0 --out /work/tmp/ufc/plain.pt
    python3 /work/eval/upstream_format_check.py vllm --spec 1 --out /work/tmp/ufc/spec.pt

  Target = Voxtral's 30-layer text decoder exported as LlamaForCausalLM
  (tools/extract_text_decoder.py), because upstream VoxtralForConditionalGeneration does not
  implement the aux-hidden-state interface DFlash needs. With --spec 1 the script records the
  draft config as resolved by vLLM (dflash_config, block_size, RoPE base actually used by the
  rotary layer, aux layers set on the target, mask id), checks weight loading (vLLM's
  DFlash ``load_weights`` returns None, which disables vLLM's own strict missing-weight check,
  so it is done here), checks embed / lm_head sharing with the target, and dumps the first
  proposal: target aux hidden states, anchor token, positions, draft logits and draft tokens.

``ref`` (run in the local CPU venv)::

    python3 /work/eval/upstream_format_check.py vllm --spec 1 --patch-knorm 1 --out /work/tmp/ufc/spec_knorm.pt
    python eval/upstream_format_check.py ref --plain tmp/ufc/plain.pt --spec tmp/ufc/spec.pt \
        --spec-patched tmp/ufc/spec_knorm.pt

  Recomputes the same first proposal with (a) model/dflash.py DFlashDraft (fp32 and bf16),
  target features from model/voxtral_target.py forward_full, and (b) the public z-lab
  DFlashDraftModel loaded with transformers ``from_pretrained`` from the release directory,
  and compares draft tokens / logits / block outputs.

Known upstream issue (vLLM 0.31.0, CPU platform): ``DFlashQwen3Model._normalize_context_k``
normalizes the context keys of all draft layers with one grouped ``ops.rms_norm`` call over a
stacked [L, head_dim] weight; the CPU kernel applies only layer 0's ``k_norm`` weight to every
layer (the XPU branch loops per layer). Context keys of layers 1..L-1 are then wrong, which
lowers acceptance but stays lossless. ``--patch-knorm 1`` replaces it with the per-layer loop
so the checkpoint can be compared against the intended computation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# "<s>The history of the city of Paris begins with a small Celtic settlement" (tekken v7 ids).
PROMPT_TEXT = "The history of the city of Paris begins with a small Celtic settlement"
PROMPT = [1, 1784, 6330, 1307, 1278, 5970, 1307, 6993, 19570, 1454, 1261, 3709, 55307, 27827]
MAX_TOKENS = 64


# --------------------------------------------------------------------------------------------
# vLLM half (container)
# --------------------------------------------------------------------------------------------
def run_vllm(args):
    os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "8")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine core in-process
    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import qwen3_dflash, utils as mutils

    rec: dict = {"propose_calls": 0, "patch_cpu_grouped_knorm": bool(args.patch_knorm)}

    if args.patch_knorm:
        # vLLM 0.31 CPU: DFlashQwen3Model._normalize_context_k runs ONE grouped ops.rms_norm over
        # [L, ctx, nkv, hd] with a stacked [L, hd] weight. The CPU kernel applies only row 0
        # (layer 0's k_norm) to every layer; only the XPU branch loops per layer. Use the
        # per-layer loop (same as the upstream XPU path) to get the intended computation.
        from vllm import _custom_ops as ops

        def _per_layer_knorm(self, all_k):
            out = torch.empty_like(all_k)
            for i in range(all_k.shape[0]):
                ops.rms_norm(out[i], all_k[i], self._k_norm_weights[i], self._rms_norm_eps)
            return out

        qwen3_dflash.DFlashQwen3Model._normalize_context_k = _per_layer_knorm

    # ---- weight-loading audit -------------------------------------------------------------
    orig_lw = qwen3_dflash.DFlashQwen3ForCausalLM.load_weights

    def lw(self, weights):
        names = []

        def tap(it):
            for n, w in it:
                names.append(n)
                yield n, w

        loaded_sets = []
        orig_awl = mutils.AutoWeightsLoader.load_weights

        def awl(loader, *a, **k):
            r = orig_awl(loader, *a, **k)
            loaded_sets.append(set(r))
            return r

        mutils.AutoWeightsLoader.load_weights = awl
        try:
            out = orig_lw(self, tap(weights))
        finally:
            mutils.AutoWeightsLoader.load_weights = orig_awl
        def norm(n):
            return n[len("model."):] if n.startswith("model.") else n

        loaded = {norm(n) for ls in loaded_sets for n in ls}
        params = {norm(n) for n, _ in self.named_parameters()}
        rec["load"] = {
            "checkpoint_tensors": len(names),
            "checkpoint_names": sorted(names),
            "returned_by_load_weights": None if out is None else "set",
            "params_total": len(params),
            # Expected: embed_tokens / lm_head (shared from the target after loading) and
            # mask_embedding (only filled when a mask_embedding.pt is shipped).
            "params_not_loaded": sorted(params - loaded),
            "loaded_not_params": sorted(loaded - params),
            "checkpoint_names_not_params": sorted(
                {n for n in names} - {k.replace("qkv_proj", p).replace("gate_up_proj", g)
                                      for k in params for p in ("q_proj", "k_proj", "v_proj")
                                      for g in ("gate_proj", "up_proj")}),
        }
        return out

    qwen3_dflash.DFlashQwen3ForCausalLM.load_weights = lw

    # ---- first-proposal capture -----------------------------------------------------------
    orig_cl = qwen3_dflash.DFlashQwen3ForCausalLM.compute_logits
    state = {"in_propose": False}

    def cl(self, hidden_states):
        logits = orig_cl(self, hidden_states)
        if state["in_propose"] and "draft_logits" not in rec:
            rec["draft_logits"] = logits.detach().float().cpu().clone()
        return logits

    qwen3_dflash.DFlashQwen3ForCausalLM.compute_logits = cl

    # The CPU platform runs the V2 model runner, whose DFlash path is DFlashSpeculator.
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as v2spec, utils as v2utils
    from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils
    import vllm.v1.worker.gpu.model_runner  # noqa: F401  (so its imported name can be patched)

    orig_set_aux = eagle3_utils.set_eagle3_aux_hidden_state_layers

    def set_aux(model, spec_config):
        orig_set_aux(model, spec_config)
        rec["target_aux_layers_from_config"] = list(eagle3_utils.get_eagle3_aux_layers_from_config(spec_config) or [])
        holder = model
        for _ in range(3):
            if hasattr(holder, "aux_hidden_state_layers"):
                break
            holder = getattr(holder, "model", holder)
        rec["target_aux_hidden_state_layers"] = list(getattr(holder, "aux_hidden_state_layers", ()))

    eagle3_utils.set_eagle3_aux_hidden_state_layers = set_aux
    for mod in list(sys.modules.values()):
        if getattr(mod, "set_eagle3_aux_hidden_state_layers", None) is orig_set_aux:
            mod.set_eagle3_aux_hidden_state_layers = set_aux

    orig_load = v2utils.load_dflash_model

    def load_dflash(target_model, vllm_config):
        dm = orig_load(target_model, vllm_config)
        tl = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
        ti = getattr(tl, "model", tl)
        rec["embed_shared"] = dm.model.embed_tokens.weight.data_ptr() == ti.embed_tokens.weight.data_ptr()
        rec["lm_head_shared"] = dm.lm_head.weight.data_ptr() == tl.lm_head.weight.data_ptr()
        return dm

    v2spec.load_dflash_model = load_dflash

    def describe(spk, hf, m):
        att = m.layers[0].self_attn
        return {
            "architectures": getattr(hf, "architectures", None),
            "dflash_config": getattr(hf, "dflash_config", None),
            "block_size": getattr(hf, "block_size", None),
            "rope_parameters": getattr(hf, "rope_parameters", None),
            "rope_theta_attr": getattr(hf, "rope_theta", None),
            "rotary_base_used": [float(l.self_attn.rotary_emb.base) for l in m.layers],
            "rotary_is_neox": bool(att.rotary_emb.is_neox_style),
            "draft_layer_causal": [bool(l.self_attn.causal) for l in m.layers],
            "requires_non_causal": bool(getattr(spk, "requires_non_causal", False)),
            "mask_token_id_model": m.mask_token_id,
            "parallel_drafting_token_id": spk.parallel_drafting_token_id,
            "has_separate_mask_embedding": bool(m.has_separate_mask_embedding),
            "use_aux_hidden_state": bool(m.use_aux_hidden_state),
            "fc_input_size": int(m.fc.input_size),
            "num_hidden_layers": len(m.layers),
            "num_speculative_tokens": int(spk.num_speculative_steps),
            "draft_vocab_size": getattr(hf, "draft_vocab_size", None),
            "has_d2t": spk.model.draft_id_to_target_id is not None,
        }

    orig_v2_propose = v2spec.DFlashSpeculator.propose

    def v2_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                   aux_hidden_states, num_sampled, num_rejected, last_sampled, *a, **k):
        real = not k.get("dummy_run", False) and not k.get("is_profile", False)
        first = real and rec["propose_calls"] == 0
        if real:
            rec["propose_calls"] += 1
        if first:
            n = input_batch.num_tokens
            rec["target_token_ids"] = input_batch.input_ids[:n].cpu().clone() \
                if hasattr(input_batch, "input_ids") else None
            rec["target_positions"] = input_batch.positions[:n].cpu().clone()
            rec["target_hidden_states"] = torch.cat([h[:n] for h in aux_hidden_states], -1).cpu().clone()
            rec["next_token_ids"] = last_sampled[input_batch.idx_mapping[:1].long()].cpu().clone()
        state["in_propose"] = first
        try:
            out = orig_v2_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                                  aux_hidden_states, num_sampled, num_rejected, last_sampled, *a, **k)
        finally:
            state["in_propose"] = False
        if first:
            nq = self.num_query_per_req
            n = input_batch.num_tokens
            rec["draft_tokens"] = out[:1].cpu().clone()
            rec["query_input_ids"] = self.input_buffers.input_ids[:nq].cpu().clone()
            rec["query_positions"] = self.input_buffers.positions[:nq].cpu().clone()
            rec["context_positions"] = self.context_positions[:n].cpu().clone()
            rec["resolved_config"] = describe(self, self.draft_model_config.hf_config, self.model.model)
            rec["runner"] = "v2"
        return out

    v2spec.DFlashSpeculator.propose = v2_propose

    kw = dict(model=args.target, tokenizer_mode="mistral", dtype="bfloat16", max_model_len=1024,
              enforce_eager=True, seed=0, distributed_executor_backend="uni",
              disable_log_stats=False)
    if args.spec:
        kw["speculative_config"] = {"method": "dflash", "model": args.draft,
                                    "num_speculative_tokens": args.k}
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    out = llm.generate([{"prompt_token_ids": PROMPT}], sp)
    rec["output_token_ids"] = list(out[0].outputs[0].token_ids)
    rec["spec"] = bool(args.spec)
    if args.spec:
        mets = {}
        try:
            for m in llm.get_metrics():
                if "spec_decode" in m.name:
                    v = getattr(m, "value", None)
                    if v is None and hasattr(m, "values"):
                        v = list(m.values)
                    mets[m.name] = v
        except Exception as e:  # noqa: BLE001
            mets["error"] = repr(e)
        rec["spec_metrics"] = mets
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, args.out)
    summary = {k: v for k, v in rec.items() if not hasattr(v, "shape")}
    summary.get("load", {}).pop("checkpoint_names", None)
    print("UFC_VLLM", json.dumps(summary, default=str))


# --------------------------------------------------------------------------------------------
# Reference half (local venv)
# --------------------------------------------------------------------------------------------
def _zlab_module(root: Path):
    import importlib
    import types
    ref_dir = root / "tmp" / "refs"
    pkg = types.ModuleType("zlabref")
    pkg.__path__ = [str(ref_dir / "zlabref")]
    (ref_dir / "zlabref").mkdir(exist_ok=True)
    (ref_dir / "zlabref" / "dflash.py").write_text((ref_dir / "zlab_dflash.py").read_text())
    (ref_dir / "zlabref" / "utils.py").write_text((ref_dir / "utils.py").read_text())
    sys.modules["zlabref"] = pkg
    return importlib.import_module("zlabref.dflash")


def _first_mismatch(a, b):
    a, b = list(a), list(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_ref(args):
    import torch
    import torch.nn.functional as F

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model.release import load_release
    from model.voxtral_target import VoxtralTarget, load_hf_voxtral

    torch.set_num_threads(args.threads)
    rel = Path(args.draft_local)
    plain = torch.load(args.plain, weights_only=False)
    spec = torch.load(args.spec, weights_only=False)
    res: dict = {"note": "format check on a text-only prompt; not an acceptance measurement"}

    # Prompt ids reproducible with the public tokenizer.
    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        tok = Tekkenizer.from_file(str(Path(args.target_local) / "tekken.json"))
        res["prompt_ids_match_tokenizer"] = [1] + tok.encode(PROMPT_TEXT, bos=False, eos=False) == PROMPT
    except Exception as e:  # noqa: BLE001
        res["prompt_ids_match_tokenizer"] = repr(e)

    # ---- vLLM side summary ----------------------------------------------------------------
    res["vllm_resolved_config"] = spec["resolved_config"]
    for k in ("target_aux_layers_from_config", "target_aux_hidden_state_layers", "embed_shared",
              "lm_head_shared", "runner"):
        res[f"vllm_{k}"] = spec.get(k)
    ld = dict(spec["load"])
    ld.pop("checkpoint_names", None)
    res["vllm_load"] = ld
    res["lossless"] = plain["output_token_ids"] == spec["output_token_ids"]
    res["plain_first_mismatch"] = _first_mismatch(plain["output_token_ids"], spec["output_token_ids"])
    n_gen = len(spec["output_token_ids"])
    res["vllm_propose_calls"] = spec["propose_calls"]
    res["vllm_mean_tokens_per_step"] = n_gen / max(1, spec["propose_calls"])
    res["vllm_spec_metrics"] = spec.get("spec_metrics")

    anchor = int(spec["next_token_ids"].reshape(-1)[0])
    P = len(PROMPT)
    bs = 1 + spec["resolved_config"]["num_speculative_tokens"]
    assert spec["target_token_ids"].tolist() == PROMPT, spec["target_token_ids"].tolist()
    assert spec["context_positions"].tolist() == list(range(P)), spec["context_positions"].tolist()
    res["vllm_query_positions"] = spec["query_positions"].tolist()
    res["vllm_query_input_ids"] = spec["query_input_ids"].tolist()
    vllm_tokens = spec["draft_tokens"].reshape(-1).tolist()
    vllm_logits = spec["draft_logits"]  # [15, V] fp32 copy of bf16 logits
    res["vllm_anchor"] = anchor
    res["vllm_draft_tokens"] = vllm_tokens

    # ---- reference target features --------------------------------------------------------
    cfg_json = json.loads((rel / "config.json").read_text())
    layer_ids = cfg_json["dflash_config"]["target_layer_ids"]
    mask_id = cfg_json["dflash_config"]["mask_token_id"]
    hf = load_hf_voxtral(dtype=torch.float32)
    tgt = VoxtralTarget(hf, max_pos=1024).eval()
    del hf
    ids = torch.tensor([PROMPT])
    block = torch.tensor([[anchor] + [mask_id] * (bs - 1)])
    q_pos = torch.arange(P, P + bs)[None]
    c_pos = torch.arange(P)[None]

    outs = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        tgt.to(dt)
        with torch.no_grad():
            x = F.embedding(ids, tgt.embed)
            feats, logits = tgt.forward_full(x, layer_ids, want_logits=True)
            noise = F.embedding(block, tgt.embed)
        outs[name] = dict(feats=feats, noise=noise, anchor=int(logits[0, -1].float().argmax()),
                          lm_head=tgt.lm_head.detach().clone())
    res["ref_anchor_fp32"] = outs["fp32"]["anchor"]
    res["ref_anchor_bf16"] = outs["bf16"]["anchor"]
    vfeats = spec["target_hidden_states"]  # [P, 5H] bf16 from vLLM
    res["target_feats_vllm_bf16_vs_ref_fp32_maxabs"] = float((vfeats.float() - outs["fp32"]["feats"][0]).abs().max())
    res["target_feats_rel_err"] = float((vfeats.float() - outs["fp32"]["feats"][0]).norm()
                                        / outs["fp32"]["feats"][0].norm())
    del tgt

    # ---- (a) our DFlashDraft --------------------------------------------------------------
    runs = {"vllm": spec}
    if args.spec_patched:
        sp2 = torch.load(args.spec_patched, weights_only=False)
        assert sp2["target_token_ids"].tolist() == PROMPT and int(sp2["next_token_ids"].reshape(-1)[0]) == anchor
        runs["vllm_knorm_patched"] = sp2
        res["lossless_knorm_patched"] = plain["output_token_ids"] == sp2["output_token_ids"]
        res["vllm_knorm_patched_spec_metrics"] = sp2.get("spec_metrics")
        res["vllm_knorm_patched_draft_tokens"] = sp2["draft_tokens"].reshape(-1).tolist()

    def ours(dt, feats, noise, lm_head):
        m = load_release(rel, dtype=dt).eval()
        with torch.no_grad():
            h = m(noise.to(dt), feats.to(dt), q_pos, c_pos, None)
            lg = F.linear(h, lm_head.to(dt)).float()
        return h.float(), lg[0, 1:]

    def cmp(tag, lg):
        toks = lg.argmax(-1).tolist()
        res[f"{tag}_tokens"] = toks
        for rn, r in runs.items():
            vt = r["draft_tokens"].reshape(-1).tolist()
            vl = r["draft_logits"]
            res[f"{tag}_vs_{rn}"] = {
                "tokens_equal": toks == vt,
                "first_mismatch": _first_mismatch(toks, vt),
                "logits_maxabs": round(float((lg - vl).abs().max()), 4),
                "logits_cos_min": round(float(F.cosine_similarity(lg, vl, -1).min()), 6),
            }

    h32, lg32 = ours(torch.float32, outs["fp32"]["feats"], outs["fp32"]["noise"], outs["fp32"]["lm_head"])
    cmp("ours_fp32", lg32)
    _, lg16 = ours(torch.bfloat16, outs["bf16"]["feats"], outs["bf16"]["noise"], outs["bf16"]["lm_head"])
    cmp("ours_bf16", lg16)
    res["ours_bf16_vs_ours_fp32_logits_cos_min"] = round(float(F.cosine_similarity(lg16, lg32, -1).min()), 6)
    # Same draft fed with vLLM's own target features: isolates the draft loader from target noise.
    vf = vfeats[None]
    _, lgv32 = ours(torch.float32, vf.float(), outs["fp32"]["noise"], outs["fp32"]["lm_head"])
    cmp("ours_fp32_vllmfeats", lgv32)
    _, lgv16 = ours(torch.bfloat16, vf, outs["bf16"]["noise"], outs["bf16"]["lm_head"])
    cmp("ours_bf16_vllmfeats", lgv16)

    # ---- (b) public z-lab reference, loaded with transformers from_pretrained --------------
    zl = _zlab_module(root)
    zm, info = zl.DFlashDraftModel.from_pretrained(str(rel), torch_dtype=torch.float32,
                                                   attn_implementation="eager",
                                                   output_loading_info=True)
    zm.eval()
    res["zlab_loading_info"] = {k: sorted(v) if isinstance(v, (list, set)) else str(v)
                                for k, v in info.items()}
    res["zlab_rope_theta"] = float(zm.rotary_emb.inv_freq[1] ** (-zm.config.head_dim / 2)) \
        if hasattr(zm.rotary_emb, "inv_freq") else None
    res["zlab_target_layer_ids"] = list(zm.target_layer_ids)
    res["zlab_mask_token_id"] = zm.mask_token_id
    res["zlab_block_size"] = zm.block_size
    with torch.no_grad():
        hz = zm(position_ids=torch.arange(P + bs)[None], noise_embedding=outs["fp32"]["noise"],
                target_hidden=outs["fp32"]["feats"], attention_mask=None, is_causal=False).float()
    res["zlab_vs_ours_fp32_block_maxabs"] = float((hz - h32).abs().max())
    res["zlab_vs_ours_fp32_allclose_1e-4"] = bool(torch.allclose(hz, h32, atol=1e-4, rtol=1e-4))
    lgz = F.linear(hz, outs["fp32"]["lm_head"])[0, 1:]
    res["zlab_tokens_eq_ours_fp32"] = lgz.argmax(-1).tolist() == res["ours_fp32_tokens"]
    res["zlab_vs_ours_fp32_logits_maxabs"] = float((lgz - lg32).abs().max())

    Path(args.json).write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps(res, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("vllm")
    v.add_argument("--target", default="/work/tmp/voxtral_text_llama")
    v.add_argument("--draft", default="/work/release")
    v.add_argument("--spec", type=int, default=1)
    v.add_argument("--k", type=int, default=15)
    v.add_argument("--out", required=True)
    v.add_argument("--patch-knorm", type=int, default=0,
                   help="1 = per-layer context K-norm (works around the CPU grouped rms_norm issue)")
    r = sub.add_parser("ref")
    r.add_argument("--plain", required=True)
    r.add_argument("--spec", required=True)
    r.add_argument("--spec-patched", default=None, help="vllm --spec 1 --patch-knorm 1 dump")
    r.add_argument("--draft-local", default="release")
    r.add_argument("--target-local", default="tmp/voxtral_text_llama")
    r.add_argument("--json", default="tmp/ufc/result.json")
    r.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    run_vllm(a) if a.mode == "vllm" else run_ref(a)


if __name__ == "__main__":
    main()
