# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Fail the compatibility gate on missing cases, changed prompts or score drift."""

import argparse
import json
from pathlib import Path

from eval_cases import cases

ROOT = Path(__file__).resolve().parent


def compare(reference, local, expected_ids):
    # Only first run per fixture is compared; repeat timings are reported separately.
    reference = {row["id"]: row for row in reference["cases"] if row["repeat"] == 0}
    local = {row["id"]: row for row in local["cases"] if row["repeat"] == 0}
    if set(reference) != set(expected_ids) or set(local) != set(expected_ids):
        raise ValueError("Incomplete evaluation: expected every fixture on both engines")
    rows = []
    for key in expected_ids:
        a, b = reference[key], local[key]
        if a["token_sha256"] != b["token_sha256"]:
            raise ValueError(f"Different token inputs: {key}")
        pa, pb = a["probabilities"], b["probabilities"]
        if len(pa) != len(pb):
            raise ValueError(f"Different option counts: {key}")
        delta = max(abs(x - y) for x, y in zip(pa, pb))
        agreement = a["decision"] == b["decision"]
        rows.append({"id": key, "same_decision": agreement,
                     "max_probability_delta": delta,
                     "passed": agreement and delta <= 0.03})
    return {"gate": "identical tokens, identical winning decision, max probability delta <= 0.03",
            "passed": all(row["passed"] for row in rows), "cases": rows,
            "caveat": "Small synthetic smoke test, not a compaction safety or calibration benchmark"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    reference = json.loads((args.results_dir / "results-reference.json").read_text())
    local = json.loads((args.results_dir / "results-mlx.json").read_text())
    ids = [row["id"] for row in cases()]
    report = compare(reference, local, ids)
    (args.results_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
