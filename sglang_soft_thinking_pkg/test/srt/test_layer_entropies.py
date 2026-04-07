import ast
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]


def _load_functions(relative_path, names, extra_globals=None):
    source = (ROOT / relative_path).read_text()
    module = ast.parse(source)
    namespace = {"__builtins__": __builtins__}
    if extra_globals:
        namespace.update(extra_globals)

    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(
        compile(ast.Module(body=selected_nodes, type_ignores=[]), str(ROOT / relative_path), "exec"),
        namespace,
    )
    return [namespace[name] for name in names]


class TestLayerEntropyHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (get_qwen_layers_to_keep,) = _load_functions(
            Path("sglang_soft_thinking_pkg/python/sglang/srt/models/qwen2.py"),
            ["_get_qwen_layers_to_keep"],
            extra_globals={
                "os": os,
                "get_bool_env_var": lambda name, default="false": os.getenv(name, default).lower() in {"true", "1"},
                "_parse_layer_weights": lambda: [
                    float(w.strip())
                    for w in os.getenv("LAYER_WEIGHTS", "1.0").split(",")
                    if w.strip()
                ] or [1.0],
            },
        )
        (
            compute_entropy_from_probs,
            compute_layer_entropies,
            select_logits_for_aggregation,
        ) = _load_functions(
            Path("sglang_soft_thinking_pkg/python/sglang/srt/layers/sampler.py"),
            [
                "_compute_entropy_from_probs",
                "_compute_layer_entropies",
                "_select_logits_for_aggregation",
            ],
            extra_globals={"torch": torch},
        )
        cls.get_qwen_layers_to_keep = staticmethod(get_qwen_layers_to_keep)
        cls.compute_entropy_from_probs = staticmethod(compute_entropy_from_probs)
        cls.compute_layer_entropies = staticmethod(compute_layer_entropies)
        cls.select_logits_for_aggregation = staticmethod(select_logits_for_aggregation)

    def test_qwen_keeps_all_layers_when_layer_entropies_enabled(self):
        with mock.patch.dict(
            os.environ,
            {"LAYER_ENTROPIES": "1", "LAYER_WEIGHTS": "0.1,0.2"},
            clear=False,
        ):
            self.assertEqual(self.get_qwen_layers_to_keep(28), 28)

    def test_aggregation_uses_weighted_suffix_while_entropies_cover_all_layers(self):
        weights = torch.tensor([0.25, 0.75])
        logits = [
            torch.tensor([[4.0, -4.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[2.0, -2.0]], dtype=torch.float32),
        ]

        selected_logits, selected_weights = self.select_logits_for_aggregation(
            logits, weights
        )
        self.assertEqual(len(selected_logits), 2)
        self.assertTrue(torch.equal(selected_weights, weights))
        self.assertTrue(torch.equal(selected_logits[0], logits[1]))
        self.assertTrue(torch.equal(selected_logits[1], logits[2]))

        layer_entropies = self.compute_layer_entropies(logits)
        self.assertEqual(layer_entropies.shape, (1, 3))
        self.assertLess(layer_entropies[0, 0].item(), 0.01)
        self.assertAlmostEqual(
            layer_entropies[0, 1].item(),
            math.log(2.0),
            places=5,
        )
        self.assertLess(layer_entropies[0, 2].item(), layer_entropies[0, 1].item())

        weighted_logits = sum(
            logit * weight
            for logit, weight in zip(selected_logits, selected_weights.tolist())
        )
        probs = torch.softmax(weighted_logits, dim=-1)
        entropy = self.compute_entropy_from_probs(probs)
        self.assertEqual(entropy.shape, (1,))
        self.assertLess(entropy.item(), layer_entropies[0, 1].item())


if __name__ == "__main__":
    unittest.main()
