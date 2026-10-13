# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""Download one pinned reference checkpoint; no credentials or remote code needed."""

import hashlib
import json
import os
from pathlib import Path
import time

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
REVISION = "f6075bff3c0f7d5704e353d0fb2ebf94888d1691"


def main():
    started = time.perf_counter()
    destination = ROOT / "models/d3-flash"
    snapshot_download(
        "vllm-sr/d3-flash", revision=REVISION, local_dir=destination,
        token=False, max_workers=1,
    )
    files = json.loads((ROOT / "upstream/reference-tree.json").read_text())
    verified = []
    for entry in files:
        if entry["type"] != "file":
            continue
        path = destination / entry["path"]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Size mismatch: {path.name}")
        if "lfs" in entry:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["lfs"]["oid"]:
                raise ValueError(f"SHA256 mismatch: {path.name}")
            verified.append({"file": path.name, "sha256": digest})
    report = {"revision": REVISION, "seconds": time.perf_counter() - started,
              "verified_lfs": verified,
              "bytes": sum(e["size"] for e in files if e["type"] == "file")}
    (ROOT / "download-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
