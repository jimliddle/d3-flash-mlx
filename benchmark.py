# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Bounded text-only latency/memory probe, separate from compaction quality tests."""

import argparse
import json
import os
from pathlib import Path
import resource
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mlx.core as mx
from d3_mlx_local import D3MLX, option_probabilities, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/d3-flash")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(48 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)
    model = D3MLX(args.model_dir, max_length=9000)
    q = {"type": "choice", "instructions": "Should the candidate result be retained?",
         "criteria": {"keep": "Unique evidence needed to finish the task",
                      "drop": "Redundant information safe to remove"}}
    filler = "Completed housekeeping event: temporary status report already archived.\n"
    ending = ("\nCURRENT TASK: Diagnose the unresolved migration failure. "
              "CANDIDATE: the only error log, which has not been saved elsewhere. "
              "It contains the failed command and exact database error.\n")
    filler_ids = model.tok.encode(filler, add_special_tokens=False)
    report = {"load_seconds": model.loaded_seconds, "cases": [],
              "load_caveat": "New process; OS filesystem cache was not flushed",
              "scope": "Synthetic text only; no live sessions or compaction hook"}
    for target in (512, 2048, 4096, 8192):
        overhead = len(model.prepare(ending, q))
        repetitions = max(1, (target - overhead) // len(filler_ids))
        state = filler * repetitions + ending
        for repeat in range(2):
            started = time.perf_counter()
            ids = model.prepare(state, q)
            probs = option_probabilities(model.logits(ids), 2, model.temperature)
            row = {"target_tokens": target, "actual_tokens": len(ids), "repeat": repeat,
                   "seconds": time.perf_counter() - started,
                   "keep_probability": probs[0],
                   "peak_mlx_bytes": mx.get_peak_memory(),
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            report["cases"].append(row)
            (args.output_dir / "benchmark-results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print("Benchmark finished; model unloads on exit.", flush=True)


if __name__ == "__main__":
    main()
