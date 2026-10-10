# SPDX-License-Identifier: Apache-2.0
"""Extract Voxtral's text decoder as a plain LlamaForCausalLM checkpoint (dev tool).

Used to exercise DFlash loaders (vLLM, vllm-neuron) against the exact 30-layer target
decoder when the multimodal wrapper does not expose aux hidden states. Writes a local copy only;
the extracted weights are Voxtral weights (Apache-2.0, Mistral AI) and are not part of the draft release.
"""

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import TARGET_ID, TARGET_REVISION  # noqa: E402


def main(out: str):
    src = Path(snapshot_download(TARGET_ID, revision=TARGET_REVISION,
                                 allow_patterns=["*.json", "model-*.safetensors"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["text_config"])
    tc.update({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "torch_dtype": "bfloat16",
               "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False})
    (out / "config.json").write_text(json.dumps(tc, indent=2))
    shutil.copy(src / "generation_config.json", out / "generation_config.json")
    shutil.copy(src / "tekken.json", out / "tekken.json")
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for k, f in wm.items():
        if not k.startswith("language_model."):
            continue
        with safe_open(src / f, "pt") as fh:
            sd[k[len("language_model."):]] = fh.get_tensor(k)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(out, len(sd))


if __name__ == "__main__":
    main(sys.argv[1])
