# Readout indexing

The d3 checkpoint has a 255-row decision head. Each row represents an option position. It is not a vocabulary-sized language-model head.

In the community MLX port at revision `77adb3bea9abde92d1f32158743a0cbbb0bf7aa3`, `d3_mlx.py` computes head logits, applies softmax, then selects entries using answer-token IDs:

```python
picked = [float(probs[i]) for i in self.code_ids[:len(keys)]]
```

The first two token IDs are 32 and 33. Those select decision rows 32 and 33, not rows 0 and 1. The 27th code has token ID 5840, outside the 255-row head.

`test_readout_audit.py` reads the expression from the unchanged snapshot and evaluates it with synthetic scores. One test makes the correct yes score 0.1 while the token-indexed result is 0.9. A second demonstrates the out-of-range index with a Python list. It does not claim that MLX itself raises the same exception for that index.

This runtime instead selects the first N logits, divides by the checkpoint temperature, and applies softmax over those N options. Unused rows cannot affect the result.

The regression tests establish the indexing defect at that source revision. They do not measure its effect on a representative dataset. The trained-model comparison separately checks agreement with the upstream PyTorch runtime on ten synthetic cases.
