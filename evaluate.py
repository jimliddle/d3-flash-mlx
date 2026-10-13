# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Separate-process reference/MLX evaluation on public synthetic fixtures only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from eval_cases import cases

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream/d3-flash"))


def main():
    startup = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["mlx", "reference"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/d3-flash")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeats and limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = cases()[:args.limit]
    from d3_format import options, to_answer

    if args.engine == "mlx":
        import mlx.core as mx
        from d3_mlx_local import D3MLX, option_probabilities
        mx.set_memory_limit(48 * 1024**3)
        mx.set_cache_limit(2 * 1024**3)
        model = D3MLX(args.model_dir, max_length=9000)
        loaded = model.loaded_seconds
        weights = model.weight_report
    else:
        import torch
        from d3_runtime import D3
        torch.set_num_threads(8)
        if args.device == "mps":
            torch.mps.set_per_process_memory_fraction(0.5)
        model = D3.from_pretrained(args.model_dir, device=args.device,
                                   verify="fast", batch_size=1, max_length=9000)
        loaded = model.loaded_seconds
        weights = {"reference_manifest_verified": bool(model.manifest)}
    report = {"engine": args.engine, "device": "metal" if args.engine == "mlx" else args.device,
              "loaded_seconds": loaded, "ready_seconds": time.perf_counter() - startup,
              "load_caveat": "New process; OS filesystem cache was not flushed",
              "weights": weights, "cases": []}
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    for fixture in fixtures:
        q, state = fixture["question"], fixture["state"]
        count = len(options(q)[0])
        for repeat in range(args.repeats):
            started = time.perf_counter()
            if args.engine == "mlx":
                ids = model.prepare(state, q)
                logits_array = model.logits(ids)
                probs = option_probabilities(logits_array, count, model.temperature)
                logits = logits_array[:count].tolist()
            else:
                prepared = model.prepare(state, {"q": q})
                if prepared.errors:
                    raise ValueError(prepared.errors)
                ids = prepared.sequences["q"]
                logits_array = model.logits([ids], [count])
                probs = torch.softmax(logits_array / model.temperature, dim=-1)[0, :count].tolist()
                logits = logits_array[0, :count].tolist()
            seconds = time.perf_counter() - started
            answer = to_answer(q, probs)
            decision = answer.get("choice")
            if q["type"] == "noul":
                decision = "true" if answer["noul"] >= 0.5 else "false"
            row = {"id": fixture["id"], "repeat": repeat, "tokens": len(ids),
                   "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   "seconds": seconds, "logits": logits, "probabilities": probs,
                   "decision": decision, "expected": fixture["expected"],
                   "correct": decision == fixture["expected"]}
            report["cases"].append(row)
            report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if args.engine == "mlx":
                report["peak_mlx_bytes"] = mx.get_peak_memory()
            (args.output_dir / f"results-{args.engine}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print("Finished; model unloads when this process exits.", flush=True)


if __name__ == "__main__":
    main()
