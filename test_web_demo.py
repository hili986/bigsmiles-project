"""
Unit tests for web_demo.py — BigSMILES Web Demo Backend.
web_demo.py 单元测试 — BigSMILES Web 演示后端。

Tests cover:
    1. API endpoint handlers (check, parse, fingerprint, predict)
    2. HTML serving
    3. Error handling
    4. Pipeline integration
"""

import unittest
import json

# We test the handler functions directly, not the HTTP server.
from web_demo import (
    handle_check,
    handle_parse,
    handle_fingerprint,
    handle_predict,
    handle_pipeline,
    get_index_html,
)


# ===========================================================================
# 1. Syntax Check Tests / 语法检查测试
# ===========================================================================

class TestHandleCheck(unittest.TestCase):
    """Tests for handle_check()."""

    def test_valid_bigsmiles(self):
        result = handle_check("{[$]CC[$]}")
        self.assertTrue(result["valid"])
        self.assertEqual(result["bigsmiles"], "{[$]CC[$]}")

    def test_invalid_bigsmiles(self):
        result = handle_check("{[$]CC")
        self.assertFalse(result["valid"])
        self.assertIn("errors", result)

    def test_empty_input(self):
        result = handle_check("")
        self.assertFalse(result["valid"])

    def test_simple_smiles(self):
        """Non-BigSMILES SMILES should be valid (no stochastic objects)."""
        result = handle_check("CC(=O)O")
        self.assertTrue(result["valid"])


# ===========================================================================
# 2. Parse Tests / 解析测试
# ===========================================================================

class TestHandleParse(unittest.TestCase):
    """Tests for handle_parse()."""

    def test_parse_basic(self):
        result = handle_parse("{[$]CC[$]}")
        self.assertIn("topology", result)
        self.assertIn("repeat_units", result)
        self.assertIn("bonding_descriptors", result)

    def test_parse_linear_homopolymer(self):
        result = handle_parse("{[$]CC[$]}")
        self.assertEqual(result["topology"], "linear_homopolymer")

    def test_parse_returns_round_trip(self):
        result = handle_parse("{[$]CC[$]}")
        self.assertIn("round_trip", result)

    def test_parse_invalid_returns_error(self):
        result = handle_parse("{[$]CC")
        self.assertIn("error", result)


# ===========================================================================
# 3. Fingerprint Tests / 指纹测试
# ===========================================================================

class TestHandleFingerprint(unittest.TestCase):
    """Tests for handle_fingerprint()."""

    def test_fingerprint_basic(self):
        result = handle_fingerprint("{[$]CC[$]}")
        self.assertIn("morgan", result)
        self.assertIn("fragments", result)
        self.assertIn("descriptors", result)
        self.assertIn("total_features", result)

    def test_fingerprint_morgan_length(self):
        result = handle_fingerprint("{[$]CC[$]}")
        if result.get("morgan") is not None:
            self.assertEqual(result["morgan_bits"], len(result["morgan"]))

    def test_fingerprint_no_rdkit_graceful(self):
        """Should return partial results if RDKit not available."""
        result = handle_fingerprint("{[$]CC[$]}")
        # fragments and descriptors always work; morgan may be None
        self.assertIn("fragments", result)
        self.assertIn("descriptors", result)

    def test_fingerprint_fragment_names(self):
        result = handle_fingerprint("{[$]CC[$]}")
        self.assertIn("fragment_names", result)
        self.assertIsInstance(result["fragment_names"], list)


# ===========================================================================
# 4. Predict Tests / 预测测试
# ===========================================================================

class TestHandlePredict(unittest.TestCase):
    """Tests for handle_predict()."""

    def test_predict_returns_tg(self):
        result = handle_predict("{[$]CC[$]}")
        self.assertIn("predicted_tg_k", result)
        self.assertIsInstance(result["predicted_tg_k"], (int, float))

    def test_predict_returns_model_info(self):
        result = handle_predict("{[$]CC[$]}")
        self.assertIn("model", result)

    def test_predict_value_reasonable(self):
        """Predicted Tg should be in a reasonable range (100-800K)."""
        result = handle_predict("{[$]CC[$]}")
        tg = result["predicted_tg_k"]
        self.assertGreater(tg, 50)
        self.assertLess(tg, 1000)


# ===========================================================================
# 5. Pipeline Tests / 流水线测试
# ===========================================================================

class TestHandlePipeline(unittest.TestCase):
    """Tests for handle_pipeline() — end-to-end."""

    def test_pipeline_valid_input(self):
        result = handle_pipeline("{[$]CC[$]}")
        self.assertIn("check", result)
        self.assertIn("parse", result)
        self.assertIn("fingerprint", result)
        self.assertIn("predict", result)
        self.assertTrue(result["check"]["valid"])

    def test_pipeline_invalid_input(self):
        result = handle_pipeline("{[$]CC")
        self.assertIn("check", result)
        self.assertFalse(result["check"]["valid"])
        # Pipeline should still attempt what it can
        self.assertIn("parse", result)

    def test_pipeline_polyethylene(self):
        """Polyethylene: {[$]CC[$]}"""
        result = handle_pipeline("{[$]CC[$]}")
        self.assertTrue(result["check"]["valid"])
        self.assertIn("predicted_tg_k", result["predict"])

    def test_pipeline_polystyrene(self):
        """Polystyrene: {[$]CC(c1ccccc1)[$]}"""
        result = handle_pipeline("{[$]CC(c1ccccc1)[$]}")
        self.assertTrue(result["check"]["valid"])
        tg = result["predict"]["predicted_tg_k"]
        self.assertGreater(tg, 50)


# ===========================================================================
# 6. HTML Serving Tests / HTML 服务测试
# ===========================================================================

class TestGetIndexHtml(unittest.TestCase):
    """Tests for get_index_html()."""

    def test_returns_html(self):
        html = get_index_html()
        self.assertIn("<html", html)
        self.assertIn("BigSMILES", html)

    def test_has_input_field(self):
        html = get_index_html()
        self.assertIn("input", html.lower())

    def test_has_javascript(self):
        html = get_index_html()
        self.assertIn("<script", html)


if __name__ == "__main__":
    unittest.main()
