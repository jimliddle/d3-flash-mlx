# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

import unittest

import mlx.core as mx
from d3_mlx_local import BF16RMSNorm, option_probabilities, validate_question


class RuntimeTests(unittest.TestCase):
    def test_norm_preserves_activation_dtype(self):
        norm = BF16RMSNorm(4)
        norm.weight = mx.ones((4,), dtype=mx.float32)
        self.assertEqual(norm(mx.ones((1, 2, 4), dtype=mx.bfloat16)).dtype, mx.bfloat16)

    def test_binary_uses_option_positions(self):
        logits = [0.0] * 255
        logits[:2] = [0.0, 2.0]
        logits[32:34] = [2.0, 0.0]
        self.assertGreater(option_probabilities(mx.array(logits), 2)[1], 0.8)

    def test_unused_logits_do_not_underflow_active_options(self):
        logits = mx.array([0.0, 0.0] + [1e6] * 253)
        self.assertEqual(option_probabilities(logits, 2), [0.5, 0.5])

    def test_27_and_255_options(self):
        for count in (27, 255):
            result = option_probabilities(mx.zeros((255,)), count)
            self.assertEqual(len(result), count)
            self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_temperature(self):
        logits = mx.arange(255).astype(mx.float32)
        cold = option_probabilities(logits, 2, 0.5)
        warm = option_probabilities(logits, 2, 2.0)
        self.assertGreater(cold[1], warm[1])

    def test_invalid_probabilities(self):
        for count in (0, 256):
            with self.assertRaises(ValueError):
                option_probabilities(mx.zeros((255,)), count)
        for temp in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                option_probabilities(mx.zeros((255,)), 2, temp)
        with self.assertRaises(ValueError):
            option_probabilities(mx.full((255,), float("nan")), 2)

    def test_invalid_contracts(self):
        for question in ({"type": "score"}, {"type": "choice", "criteria": {}},
                         {"type": "noul", "criteria": {"maybe": "x"}}):
            with self.assertRaises(ValueError):
                validate_question(question)


if __name__ == "__main__":
    unittest.main(verbosity=2)
