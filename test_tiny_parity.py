# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Numerical conversion smoke test; random tiny weights, NOT trained-model quality."""

import json
import os
from pathlib import Path
import sys
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
import mlx.core as mx
import numpy as np
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from d3_mlx_local import ROOT, build_text_model, convert_text_weight, encode
from d3_runtime import PromptEncoder


class TinyParity(unittest.TestCase):
    def test_explicit_layer_types_and_numerical_parity(self):
        cfg = json.loads((ROOT / "upstream/d3-flash-mlx/config.json").read_text())["text_config"]
        cfg.update(hidden_size=256, intermediate_size=512, vocab_size=128,
                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                   head_dim=128, linear_num_key_heads=1, linear_num_value_heads=2,
                   max_position_embeddings=1024, eos_token_id=127,
                   layer_types=["linear_attention", "full_attention"])
        cfg["rope_parameters"]["mrope_section"] = [6, 5, 5]
        torch.manual_seed(20261012)
        torch.set_num_threads(4)
        reference_cfg = Qwen3_5TextConfig(**cfg)
        reference_cfg._attn_implementation = "sdpa"
        device = os.environ.get("TINY_REFERENCE_DEVICE", "cpu")
        reference = Qwen3_5TextModel(reference_cfg).to(torch.bfloat16).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if name == "norm.weight" or name.endswith((
                    ".input_layernorm.weight", ".post_attention_layernorm.weight",
                    ".q_norm.weight", ".k_norm.weight",
                )):
                    parameter.copy_(torch.randn_like(parameter) * 0.1)
        local = build_text_model(cfg)
        weights = [(name, convert_text_weight(name, mx.array(value.float().numpy()).astype(mx.bfloat16)))
                   for name, value in reference.state_dict().items()]
        local.load_weights(weights, strict=True)
        local.eval()
        mx.eval(local.parameters())
        reference = reference.to(device)
        self.assertFalse(local.layers[1].is_linear)
        results = []
        for length in (1, 17, 65):
            ids = [i % 127 for i in range(length)]
            inputs = torch.tensor([ids], device=device)
            with torch.inference_mode():
                expected = PromptEncoder(reference)(reference.embed_tokens(inputs), torch.ones_like(inputs))
            actual = encode(local, ids)
            mx.eval(actual)
            difference = np.abs(np.array(actual.astype(mx.float32)) - expected.float().cpu().numpy())
            row = {"device": device, "tokens": length, "max_abs_hidden_error": float(difference.max()),
                   "mean_abs_hidden_error": float(difference.mean())}
            results.append(row)
            print(json.dumps(row), flush=True)
            self.assertTrue(np.isfinite(difference).all())
            self.assertLess(float(difference.mean()), 0.02)
            self.assertLess(float(difference.max()), 0.15)
        (ROOT / f"tiny-parity-{device}.json").write_text(json.dumps(results, indent=2) + "\n")
        before = encode(local, [1, 2, 3])[0, 0]
        after = encode(local, [1, 2, 4])[0, 0]
        self.assertGreater(float(mx.max(mx.abs(before - after)).item()), 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
