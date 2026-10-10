# SPDX-License-Identifier: Apache-2.0
"""Load a release-format draft in upstream vLLM's DFlash path (CPU container) and run it.

Target = Voxtral's 30-layer text decoder (LlamaForCausalLM, see tools/extract_text_decoder.py),
because upstream VoxtralForConditionalGeneration does not implement the aux-hidden-state
(EAGLE3) interface DFlash needs. Checks: config parsing, weight loading, the resolved
target layers / mask id / rope, greedy losslessness (spec output == plain output).

Run inside vllm/vllm-openai-cpu with this repo mounted at /work.
"""

import json
import os
import sys

TARGET = sys.argv[1]
DRAFT = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 15

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "8")

from vllm import LLM, SamplingParams  # noqa: E402

PROMPTS = [
    [1, 3, 1784, 1877, 1278, 9196, 1307, 1278, 6220, 1044, 2],
]


def run(spec):
    kw = dict(model=TARGET, tokenizer_mode="mistral", dtype="bfloat16", max_model_len=1024,
              enforce_eager=True, seed=0)
    if spec:
        kw["speculative_config"] = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": K}
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    prompts = [{"prompt_token_ids": p[:-1]} for p in PROMPTS]
    outs = llm.generate(prompts, sp)
    res = [list(o.outputs[0].token_ids) for o in outs]
    info = {}
    if spec:
        try:
            core = llm.llm_engine
            info["metrics"] = [str(m) for m in llm.get_metrics() if "spec" in m.name][:8]
        except Exception as e:  # metrics API differs across versions
            info["metrics_error"] = repr(e)
        try:
            dm = llm.llm_engine.vllm_config.speculative_config.draft_model_config.hf_config
            info["draft_cfg"] = {"dflash_config": dm.dflash_config, "block_size": getattr(dm, "block_size", None),
                                 "rope_parameters": getattr(dm, "rope_parameters", None),
                                 "num_hidden_layers": dm.num_hidden_layers}
        except Exception as e:
            info["cfg_error"] = repr(e)
    del llm
    return res, info


if __name__ == "__main__":
    plain, _ = run(False)
    spec, info = run(True)
    print("RESULT", json.dumps({"plain": plain, "spec": spec, "lossless": plain == spec, **info}, default=str))
