# Sources and licences

## d3-flash

- Author: vLLM Semantic Router Team.
- Source: [vllm-sr/d3-flash](https://huggingface.co/vllm-sr/d3-flash/tree/f6075bff3c0f7d5704e353d0fb2ebf94888d1691).
- Revision: `f6075bff3c0f7d5704e353d0fb2ebf94888d1691`.
- Licence: Apache-2.0, retained in `upstream/d3-flash/LICENSE` and the root `LICENSE`.
- Original notice: `upstream/d3-flash/NOTICE`.

`upstream/d3-flash/` contains unchanged copies of `d3_fast.py`, `d3_format.py`, `d3_runtime.py`, and `decision_config.json`, with their licence and notice. The runtime imports the upstream prompt formatter. The evaluation uses the upstream PyTorch implementation as its reference.

`upstream/reference-tree.json` records the pinned checkpoint's root-file metadata. The downloader uses it to verify file sizes and LFS hashes. The weights and example assets are fetched from Hugging Face, not redistributed here. The upstream model is based on Qwen/Qwen3.5-9B, also under Apache-2.0.

## Community MLX port

- Source: [ahmedandaloes/d3-flash-mlx](https://huggingface.co/ahmedandaloes/d3-flash-mlx/tree/77adb3bea9abde92d1f32158743a0cbbb0bf7aa3).
- Revision: `77adb3bea9abde92d1f32158743a0cbbb0bf7aa3`.
- Licence and notice: retained in `upstream/d3-flash-mlx/`.

That directory contains unchanged copies of `d3_mlx.py`, `config.json`, and `decision_config.json`. They are an audit snapshot, not the runtime used for inference here. The readout regression test extracts the affected expression from that snapshot. The tiny numerical test uses its architecture configuration. Findings apply to this revision, not necessarily to later releases.

## Changes in this repository

The root Python files, fixtures, tests, measurements, and documentation are additions, published under Apache-2.0. `d3_mlx_local.py` implements the d3 text inference contract using MLX-VLM's Qwen layers. It loads the original checkpoint instead of the community conversion, checks all text weight keys, handles Qwen normalisation offsets, supplies explicit token positions, and reads option rows rather than vocabulary token IDs. See [the readout note](docs/READOUT.md).

No vendored upstream file has been modified. This is a new Git repository containing an attributed port, not a GitHub-native fork of either Hugging Face repository.

## Dependencies

MLX, MLX-VLM, PyTorch, Transformers, Hugging Face Hub, Safetensors, and Accelerate are installed separately; their source is not vendored here. Their own licences apply. `requirements.txt` records the tested versions.

The pinned [MLX-VLM revision](https://github.com/Blaizzy/mlx-vlm/tree/4f4634bb813c0298cb1467bed2e957526c71d0b4) is MIT-licensed. Its Qwen implementation supplies the model layers used by this port. This project does not implement a new Metal backend.
