# Copyright 2026 jimliddle
# SPDX-License-Identifier: Apache-2.0
# Local implementation/test addition; upstream snapshots are unchanged.

"""No-weights reproduction of the pinned community MLX readout-indexing bug.

These tests confirm a defect in the inspected source, not model inference parity.
No downloaded module is imported or executed; only the inspected selection
expression is extracted, checked against its expected AST, and evaluated.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parent


class ReadoutAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (ROOT / "upstream/d3-flash/decision_config.json").read_text()
        )
        cls.code_ids = config["token_ids"]
        cls.width = len(config["codes"])
        tree = ast.parse((ROOT / "upstream/d3-flash-mlx/d3_mlx.py").read_text())
        expressions = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "picked" for t in node.targets)
        ]
        if len(expressions) != 1:
            raise AssertionError("Pinned source selection changed")
        expected = ast.parse(
            "[float(probs[i]) for i in self.code_ids[:len(keys)]]", mode="eval"
        )
        if ast.dump(expressions[0]) != ast.dump(expected.body):
            raise AssertionError("Pinned source selection changed")
        cls.selection = compile(ast.Expression(expressions[0]), "selection", "eval")

    def select(self, probabilities, count):
        return eval(self.selection, {
            "__builtins__": {}, "float": float, "len": len,
            "probs": probabilities, "keys": list(range(count)),
            "self": SimpleNamespace(code_ids=self.code_ids),
        })

    def test_binary_decision_can_flip(self):
        self.assertEqual(self.width, 255)
        self.assertEqual(self.code_ids[:2], [32, 33])
        probabilities = [0.0] * self.width
        probabilities[0:2] = [0.45, 0.05]
        probabilities[32:34] = [0.05, 0.45]
        reference = probabilities[:2]
        port = self.select(probabilities, 2)
        reference_yes = reference[1] / sum(reference)
        port_yes = port[1] / sum(port)
        self.assertAlmostEqual(reference_yes, 0.1)
        self.assertAlmostEqual(port_yes, 0.9)

    def test_vocabulary_index_exceeds_option_head(self):
        # Python demonstrates the invalid index; MLX's runtime behavior is untested.
        self.assertGreaterEqual(self.code_ids[26], self.width)
        with self.assertRaises(IndexError):
            self.select([1 / self.width] * self.width, 27)


if __name__ == "__main__":
    unittest.main(verbosity=2)
