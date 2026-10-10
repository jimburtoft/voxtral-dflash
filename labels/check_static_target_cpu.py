# SPDX-License-Identifier: Apache-2.0
"""CPU check: static VoxtralTarget == HF Voxtral (features at target layers, greedy tokens)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DraftConfig  # noqa: E402
from model.voxtral_target import VoxtralTarget, load_hf_voxtral  # noqa: E402


def main(label_path: str, n: int = 3):
    torch.set_num_threads(16)
    from transformers import AutoProcessor
    from model.config import TARGET_ID, TARGET_REVISION
    proc = AutoProcessor.from_pretrained(TARGET_ID, revision=TARGET_REVISION)
    hf = load_hf_voxtral(torch.bfloat16)
    tgt = VoxtralTarget(hf).eval()
    cfg = DraftConfig()
    recs = torch.load(label_path, weights_only=False)["records"][:n]
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    D = Path.home() / ".cache/huggingface/hub/datasets--openslr--librispeech_asr/snapshots/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/all/train.clean.100/0000.parquet"
    rows = {r["id"]: r for r in pq.read_table(D, columns=["id", "audio"]).to_pylist()}
    for r in recs:
        a, sr = sf.read(io.BytesIO(rows[r["clip_id"]]["audio"]["bytes"]), dtype="float32")
        x = proc.apply_transcription_request(language="en", audio=a, sampling_rate=sr, format=["wav"], model_id=TARGET_ID)
        with torch.no_grad():
            audio = tgt.encode_audio(x["input_features"].to(torch.bfloat16))
            ids = r["ids"].long()[None]
            span = tgt.audio_span(ids[0])
            emb = tgt.build_embeds(ids, audio.reshape(1, -1, tgt.H), span)
            feats, logits = tgt.forward_full(emb, cfg.target_layer_ids, want_logits=True)
            ref = r["feats"].float()
            rel = ((feats[0].float() - ref).norm() / ref.norm()).item()
            P = r["prompt_len"]
            tf_pred = logits[0, P - 1:-1].argmax(-1)
            tf_match = (tf_pred == ids[0, P:]).float().mean().item()
            # static-cache greedy
            tgt.alloc_cache(1, 512, torch.bfloat16, "cpu")
            tok = tgt.prefill(emb[:, :P])
            out = [int(tok)]
            for t in range(P, P + ids.shape[1] - P - 1):
                tok = tgt.decode(tok, torch.tensor([t]))
                out.append(int(tok))
            g_match = out == ids[0, P:].tolist()
        print(f"{r['clip_id']}: feat rel err vs HF {rel:.3e} | teacher-forced argmax==HF greedy {tf_match:.3f} | static greedy identical {g_match}")


if __name__ == "__main__":
    main(sys.argv[1])
