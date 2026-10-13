# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Deterministic public fixtures, including option-head and longer-input checks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cases():
    rows = json.loads((ROOT / "fixtures.json").read_text())
    rows.append({
        "id": "27-options", "state": "The requested item is item 26.",
        "question": {"type": "choice", "instructions": "Select the requested item.",
                     "criteria": {f"item {i}": None for i in range(27)}},
        "expected": "item 26",
    })
    rows.append({
        "id": "mixed-sentiment", "state": "The device works well, but support took days. My feelings are mixed.",
        "question": {"type": "choice", "instructions": "Classify the overall expressed sentiment.",
                     "criteria": {"positive": "Positive only", "negative": "Negative only",
                                  "mixed": "Both positive and negative", "unknown": "No sentiment expressed"}},
        "expected": "mixed",
    })
    for repeats in (160, 660):
        rows.append({
            "id": f"long-retention-{repeats}",
            "state": ("Completed housekeeping event: temporary status report already archived.\n" * repeats
                      + "\nCURRENT TASK: Diagnose the unresolved migration failure. "
                      "CANDIDATE: the only error log, which has not been saved elsewhere. "
                      "It contains the failed command and exact database error.\n"),
            "question": {"type": "choice", "instructions": "Should the candidate result be retained?",
                         "criteria": {"keep": "Unique evidence needed to finish the task",
                                      "drop": "Redundant information safe to remove"}},
            "expected": "keep",
        })
    return rows
