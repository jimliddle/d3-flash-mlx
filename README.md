# d3-flash-mlx

A text-only runtime for [d3-flash](https://huggingface.co/vllm-sr/d3-flash) on Apple Silicon. It uses MLX and Apple's Metal GPU backend, loads the original checkpoint, and returns scores for the options in a typed question.

This is an unofficial port. It is not affiliated with the vLLM Semantic Router team, Qwen, or Apple. There is already a community MLX port; this repository takes a separate approach and includes a regression test for a readout bug found in a pinned version of that port.

## Status

- Supports text `choice` and `noul` (yes/no) questions, with up to 255 options.
- Keeps the upstream prompt format and tokenizer.
- Loads the text weights with strict key and shape checks. Vision weights are not used.
- Rejects oversized inputs instead of truncating them.
- Runs locally after setup. No inference API, hosted fallback, or telemetry is added by this runtime.

It does not support images, video, the `score` question type, batching, or an HTTP server. It is not a compaction plugin. The tests do not establish that its scores are calibrated or that it can safely discard conversation history.

## Setup

Tested with Python 3.12 on an Apple M5 Max with 128 GiB RAM. Other Apple Silicon machines have not been tested here. The longest test used about 17.8 GiB of MLX memory; leave room for macOS and other applications. This is not a claim that it fits on a 24 GB machine.

The model download is about 16.8 GB, plus the Python environment. Weights are not included in this repository. The download uses the original BF16 checkpoint, not a 4-bit conversion.

```sh
git clone https://github.com/jimliddle/d3-flash-mlx.git
cd d3-flash-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_reference.py
```

The downloader pins the model revision, checks root-file sizes and large-file SHA-256 hashes, and downloads the upstream verification assets. It does not use a Hugging Face token. MLX-VLM is pinned to a public Git revision because the version used here was not available on PyPI. PyTorch and Accelerate are included for the reference tests.

## Use

Run this from the repository directory:

```python
from d3_mlx_local import D3MLX

model = D3MLX("models/d3-flash")
result = model.system_one(
    "The backup failed with a permission error. No backup was created.",
    {
        "backup_ok": {
            "type": "noul",
            "instructions": "Did the backup complete successfully?",
        },
        "route": {
            "type": "choice",
            "instructions": "Who should investigate this failure?",
            "criteria": {
                "operator": "A person who can check file permissions",
                "none": "No investigation is needed",
            },
        },
    },
)
print(result)
```

The default limit is 4,096 tokens, including the formatted question. Set `max_length` when constructing the model to change it. Trained-checkpoint comparisons here cover inputs up to 8,058 tokens; longer inputs have not been validated. Questions run sequentially. Keep the model object alive to avoid loading it for each request; exiting the process releases it.

Treat the returned scores as model outputs, not measured probabilities of correctness. Keep permission checks and irreversible actions outside the model.

## Tests

Unit tests, including a tiny randomly initialised reference model, do not need the checkpoint:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
TINY_REFERENCE_DEVICE=mps python -m unittest test_tiny_parity -v
```

After downloading the checkpoint, compare the MLX runtime with the unchanged upstream PyTorch runtime:

```sh
python evaluate.py mlx --repeats 2
python evaluate.py reference --device mps
python compare_results.py
python benchmark.py
```

`evaluate.py` and `benchmark.py` accept `--model-dir` and `--output-dir`. Run them one at a time so the two models do not compete for memory. The reference can use `--device cpu`, but it is much slower. Generated results are ignored by Git.

The recorded ten-case test matched the reference's winning decision on every case. The largest probability difference was 0.0088, or 0.88 percentage points. These are synthetic compatibility checks, not a general accuracy benchmark. See [results](RESULTS.md) and the [readout bug note](docs/READOUT.md).

## Licence and sources

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [third-party sources](THIRD_PARTY.md). Upstream reference files and notices are retained unchanged. No model weights are redistributed.
