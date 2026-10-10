# SPDX-License-Identifier: Apache-2.0
"""Device smoke test: draft forward + loss + backward on Neuron vs CPU at one batch.

Run inside the PyTorch Native container:
    python train/neuron_smoke.py --data <batch>.pt --layers 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.dflash import DFlashDraft, FrozenTargetHead  # noqa: E402
from train.dflash_objective import dflash_loss, pack_batch  # noqa: E402
from train.train_cpu_ref import collate, load_records, load_target_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    cfg = DraftConfig(num_hidden_layers=args.layers)
    recs = load_records([args.data])[: args.batch]
    head_cpu = load_target_head(torch.float32)
    pb = pack_batch([r["ids"].long() for r in recs], [int(r["prompt_len"]) for r in recs], args.n_blocks,
                    cfg.block_size, cfg.mask_token_id, generator=torch.Generator().manual_seed(0))
    feats = collate(recs, cfg.num_target_features * cfg.target_hidden_size, torch.float32)
    noise = head_cpu.embed_tokens(pb.block_ids)

    ref = DFlashDraft(cfg)
    sd = {k: v.clone() for k, v in ref.state_dict().items()}

    def step(model, head, dev, dtype, compiled=False):
        model = model.to(dev).to(dtype)
        fwd = torch.compile(model, backend="neuron") if compiled else model
        t = time.time()
        h = fwd(noise.to(dev, dtype), feats.to(dev, dtype), pb.q_pos.to(dev), pb.ctx_pos.to(dev), pb.mask.to(dev))
        logits = head.logits(h)
        pbd = type(pb)(**{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in pb.__dict__.items()})
        loss = dflash_loss(logits, pbd)
        loss.backward()
        g = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters()}
        return loss.item(), g, h.detach().float().cpu(), time.time() - t

    l_cpu, g_cpu, h_cpu, _ = step(ref, head_cpu, "cpu", torch.float32)
    dev = torch.device("neuron")
    m = DFlashDraft(cfg)
    m.load_state_dict(sd)
    head_dev = FrozenTargetHead(head_cpu.embed.to(dev, dt), head_cpu.lm_head.to(dev, dt))
    l_dev, g_dev, h_dev, t1 = step(m, head_dev, dev, dt, args.compile)
    print(f"loss cpu={l_cpu:.5f} neuron={l_dev:.5f} (first call {t1:.1f}s)")
    rel_h = ((h_dev - h_cpu).norm() / h_cpu.norm()).item()
    print(f"hidden rel err {rel_h:.4e}")
    worst = []
    for n in g_cpu:
        a, b = g_cpu[n], g_dev[n]
        rel = ((a - b).norm() / a.norm().clamp(min=1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        worst.append((rel, cos, n))
    worst.sort(reverse=True)
    for rel, cos, n in worst[:6]:
        print(f"grad {n}: rel {rel:.3e} cos {cos:.6f}")
    print("SMOKE_OK" if abs(l_cpu - l_dev) / l_cpu < 2e-2 and worst[0][1] > 0.99 else "SMOKE_MISMATCH")


if __name__ == "__main__":
    main()
