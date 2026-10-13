# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Experimental text-only d3 runtime using original weights and MLX.

Uses the upstream Apache-2.0 d3 prompt contract unchanged. Unlike the inspected
community port, this loads only the text backbone, validates all weight keys,
uses explicit no-cache positions, and scores option positions, not token IDs.
This is not a harness integration or a complete System One server.
"""

import json
import math
from pathlib import Path
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5 import language as L
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream/d3-flash"))
from d3_format import MAX_OPTIONS, answer_codes, options, render, to_answer


class BF16RMSNorm(nn.RMSNorm):
    """Keep the FP32 shifted scale without promoting residuals to FP32."""

    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


def build_text_model(config):
    args = TextConfig.from_dict(config)
    lm = L.Qwen3_5Model(args)
    kinds = config["layer_types"]
    if len(kinds) != len(lm.layers):
        raise ValueError("Layer count mismatch")
    for layer, kind in zip(lm.layers, kinds):
        if kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"Unknown layer type: {kind}")
        linear = kind == "linear_attention"
        if linear != layer.is_linear:
            if linear:
                del layer.self_attn
                layer.linear_attn = L.Qwen3_5GatedDeltaNet(args)
            else:
                del layer.linear_attn
                layer.self_attn = L.Qwen3_5Attention(args)
        layer.is_linear = linear
        layer.input_layernorm = BF16RMSNorm(args.hidden_size, args.rms_norm_eps)
        layer.post_attention_layernorm = BF16RMSNorm(args.hidden_size, args.rms_norm_eps)
        if not linear:
            layer.self_attn.q_norm = BF16RMSNorm(args.head_dim, args.rms_norm_eps)
            layer.self_attn.k_norm = BF16RMSNorm(args.head_dim, args.rms_norm_eps)
    lm.norm = BF16RMSNorm(args.hidden_size, args.rms_norm_eps)
    return lm


def convert_text_weight(name, value):
    if name.endswith("conv1d.weight"):
        value = value.transpose(0, 2, 1)
    # PyTorch Qwen RMSNorm stores an offset from 1. Gated norm does not.
    if name == "norm.weight" or name.endswith((
        ".input_layernorm.weight", ".post_attention_layernorm.weight",
        ".q_norm.weight", ".k_norm.weight",
    )):
        value = value.astype(mx.float32) + 1.0
    return value


def encode(lm, ids):
    h = lm.embed_tokens(mx.array([ids], dtype=mx.int32))
    positions = mx.broadcast_to(mx.arange(len(ids))[None, None, :], (3, 1, len(ids)))
    for layer in lm.layers:
        h = layer(h, mask=None, cache=None, position_ids=positions)
        mx.eval(h)
    return lm.norm(h)


def option_probabilities(logits, count, temperature=1.0):
    if not isinstance(count, int) or not 1 <= count <= MAX_OPTIONS:
        raise ValueError("Option count must be 1..255")
    if logits.shape != (MAX_OPTIONS,):
        raise ValueError("Expected the 255-option readout, not vocabulary logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    # Slicing before softmax is equivalent to masking unused options with -inf.
    probabilities = mx.softmax(logits[:count].astype(mx.float32) / temperature)
    mx.eval(probabilities)
    values = probabilities.tolist()
    if not all(math.isfinite(p) for p in values):
        raise ValueError("Non-finite decision probabilities")
    return values


def validate_question(question):
    if not isinstance(question, dict):
        raise ValueError("Question must be an object")
    if question.get("type") not in ("choice", "noul"):
        raise ValueError("This experimental runtime supports choice/noul text only")
    criteria = question.get("criteria")
    if question["type"] == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError("Choice criteria must contain 1..255 options")
        if not all(isinstance(key, str) and key for key in criteria):
            raise ValueError("Choice keys must be nonempty strings")
    elif criteria is not None:
        if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
            raise ValueError("noul criteria may only name false/true")


class D3MLX:
    def __init__(self, model_dir, max_length=4096):
        started = time.perf_counter()
        root = Path(model_dir)
        cfg = json.loads((root / "config.json").read_text())
        self.config = json.loads((root / "decision_config.json").read_text())
        expected = {"format_id": "d3-code-readout-v1", "format_version": 1,
                    "attention_mode": "noncausal_full_attention", "prompt": "d3",
                    "pooling": "last", "readout_dtype": "float32"}
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f"Unsupported decision configuration: {key}")
        self.temperature = float(self.config["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid checkpoint temperature")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        limits = [max_length, cfg["text_config"]["max_position_embeddings"]]
        if self.config.get("max_length"):
            limits.append(self.config["max_length"])
        self.max_length = min(limits)
        self.tok = AutoTokenizer.from_pretrained(root, local_files_only=True,
                                                trust_remote_code=False)
        self.codes, token_ids = answer_codes(self.tok)
        if self.codes != self.config["codes"] or token_ids != self.config["token_ids"]:
            raise ValueError("Checkpoint answer codes differ from tokenizer")

        self.lm = build_text_model(cfg["text_config"])

        index = json.loads((root / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        weights = {}
        excluded = []
        for shard in sorted(set(weight_map.values())):
            values = mx.load(str(root / shard))
            for key, value in values.items():
                if not key.startswith("language_model."):
                    if not key.startswith("visual."):
                        raise ValueError(f"Unrecognized checkpoint weight: {key}")
                    excluded.append(key)
                    continue
                name = key.removeprefix("language_model.")
                if name in weights:
                    raise ValueError(f"Duplicate weight: {name}")
                weights[name] = convert_text_weight(name, value)
            del values
        expected_keys = dict(tree_flatten(self.lm.parameters()))
        missing = set(expected_keys) - set(weights)
        extra = set(weights) - set(expected_keys)
        if missing or extra:
            raise ValueError(f"Weight key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        self.lm.load_weights(list(weights.items()), strict=True)
        self.lm.eval()
        self.readout = mx.load(str(root / "readout.safetensors"))["weight"]
        if self.readout.shape != (MAX_OPTIONS, cfg["text_config"]["hidden_size"]):
            raise ValueError("Readout shape mismatch")
        self.readout = self.readout.astype(mx.float32)
        mx.eval(self.lm.parameters(), self.readout)
        self.loaded_seconds = time.perf_counter() - started
        self.weight_report = {"text_weights_loaded": len(weights),
                              "vision_weights_excluded": len(excluded),
                              "missing": [], "unexpected": []}

    def prepare(self, state, question):
        validate_question(question)
        text = render(self.tok, state, question, self.codes)
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError(f"Prompt has {len(ids)} tokens; limit {self.max_length}; not truncated")
        return ids

    def logits(self, ids):
        if not ids or len(ids) > self.max_length:
            raise ValueError("Empty or oversized token sequence")
        hidden = encode(self.lm, ids)[0, -1].astype(mx.float32)
        logits = self.readout @ hidden
        mx.eval(logits)
        return logits

    def system_one(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        # Validate the whole request before running any model computation.
        prepared = {name: self.prepare(state, q) for name, q in questions.items()}
        answers = {}
        for name, ids in prepared.items():
            q = questions[name]
            probs = option_probabilities(self.logits(ids), len(options(q)[0]), self.temperature)
            answers[name] = to_answer(q, probs)
        return {"answers": answers}
