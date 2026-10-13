# Test results

Recorded on 12 October 2026 on an Apple M5 Max with 128 GiB RAM, using Python 3.12.13 and the versions in `requirements.txt`. Raw results are in `measurements/2026-10-12/`.

## Compatibility

Ten synthetic cases cover yes/no decisions, three-way routing, retention of unique evidence, removal of a duplicate, an instruction embedded in a log, 27 options, mixed sentiment, and two longer retention inputs. The longest has 8,058 tokens.

- All ten inputs had identical token hashes in MLX and the upstream PyTorch runtime.
- Both runtimes chose the same answer on all ten and matched the hand-written expected labels.
- The largest probability difference was 0.0087988 (0.88 percentage points), on the longest input.
- Two MLX runs per case returned identical scores.

The comparison gate requires every fixture, identical input hashes, the same winning decision, and no probability difference above 0.03. This is a compatibility tolerance, not a calibration threshold.

Ten unit tests cover the pinned readout bug, option indexing, temperature, non-finite scores, question validation, normalisation dtype, and tiny-model numerical parity. The tiny randomly initialised model was compared with PyTorch on CPU and MPS at lengths 1, 17, and 65. Those tests check implementation details, not learned task quality.

On 13 October, the standalone repository was installed into a new virtual environment from the public dependencies. All ten unit tests, the separate tiny MPS test, and the full trained-checkpoint comparison passed again. The maximum probability difference was unchanged. The existing checkpoint was supplied with `--model-dir`; no private code or configuration was needed.

## Time and memory

Model construction took about 0.6 seconds; imports and setup brought readiness to about 2.1 seconds in the recorded MLX evaluation. The operating-system file cache was warm. These are not cold-boot load times. A first measurement before kernel caches were warm took about 1.03 seconds for a short inference; later short requests took about 67-79 ms.

The separate length probe recorded:

| Input tokens | First request | Second request |
| --- | ---: | ---: |
| 510 | 0.201 s | 0.152 s |
| 2,046 | 0.750 s | 0.559 s |
| 4,086 | 1.166 s | 1.193 s |
| 8,190 | 3.149 s | 3.340 s |

Peak MLX memory in that probe was 19.11 GB (17.80 GiB). Process RSS and MLX memory overlap and must not be added together. The benchmark sets a 48 GiB MLX memory limit; that is a ceiling, not a requirement.

The upstream PyTorch MPS reference took about 14.5 seconds for the 8,058-token case. That reference used fallback kernels. It is not a comparison with an optimised serving stack or with the performance quoted on the model card.

## What this does not show

These tests do not establish general decision accuracy, safe context compaction, resistance to prompt injection, or calibrated confidence. A single synthetic untrusted-instruction case is not a security evaluation. There are no tests on private conversations or production workflows, no multimodal tests, and no quantisation results.

For practical use, test the decisions you need, include ambiguous cases, and measure the cost of wrong answers. Agreement with a reference implementation is useful, but the reference can also be wrong.
