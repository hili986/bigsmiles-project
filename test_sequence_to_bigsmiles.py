#!/usr/bin/env python3
"""
sequence_to_bigsmiles.py 测试套件
Test suite for sequence_to_bigsmiles.py

4 个测试类、31 个测试用例 / 4 test classes, 31 test cases:
  - TestSequenceValidator  — 序列验证
  - TestFullSMILESBuilder  — Full SMILES 构建
  - TestBigSMILESGenerator — BigSMILES 生成
  - TestImageGeneration    — 图像生成
"""

import os
import sys
import unittest
import tempfile
import shutil

from sequence_to_bigsmiles import (
    validate_sequence,
    build_full_smiles,
    build_fragment_smiles,
    generate_bigsmiles,
    generate_images,
    sequence_to_representations,
    DNA_INTERNAL, DNA_TERMINAL,
    RNA_INTERNAL, RNA_TERMINAL,
    BIGSMILES_DNA, BIGSMILES_RNA,
)


# PD-L1 适配体序列 / PD-L1 Aptamer sequence (45 nt)
PDL1_APTAMER = "ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT"


# ===========================================================================
# TestSequenceValidator — 序列验证器测试
# ===========================================================================

class TestSequenceValidator(unittest.TestCase):
    """测试序列验证和清洗功能。/ Tests for sequence validation and cleaning."""

    def test_valid_dna_sequence(self):
        """有效 DNA 序列应原样返回（大写）。"""
        self.assertEqual(validate_sequence("ACGT", "DNA"), "ACGT")

    def test_valid_rna_sequence(self):
        """有效 RNA 序列应原样返回。"""
        self.assertEqual(validate_sequence("AUGC", "RNA"), "AUGC")

    def test_lowercase_converted(self):
        """小写输入应转为大写。"""
        self.assertEqual(validate_sequence("acgt", "DNA"), "ACGT")
        self.assertEqual(validate_sequence("augc", "RNA"), "AUGC")

    def test_strip_5prime_marker(self):
        """去除 5' 方向标记。"""
        self.assertEqual(validate_sequence("5'-ACGT", "DNA"), "ACGT")
        self.assertEqual(validate_sequence("5' ACGT", "DNA"), "ACGT")

    def test_strip_3prime_marker(self):
        """去除 3' 方向标记。"""
        self.assertEqual(validate_sequence("ACGT-3'", "DNA"), "ACGT")

    def test_strip_both_markers(self):
        """去除两端方向标记。"""
        self.assertEqual(validate_sequence("5'-ACGT-3'", "DNA"), "ACGT")

    def test_invalid_dna_base_u(self):
        """DNA 中不允许 U。"""
        with self.assertRaises(ValueError):
            validate_sequence("ACGU", "DNA")

    def test_invalid_rna_base_t(self):
        """RNA 中不允许 T。"""
        with self.assertRaises(ValueError):
            validate_sequence("ACGT", "RNA")

    def test_invalid_character(self):
        """非碱基字符应报错。"""
        with self.assertRaises(ValueError):
            validate_sequence("ACGX", "DNA")

    def test_empty_sequence(self):
        """空序列应报错。"""
        with self.assertRaises(ValueError):
            validate_sequence("", "DNA")


# ===========================================================================
# TestFullSMILESBuilder — Full SMILES 构建器测试
# ===========================================================================

class TestFullSMILESBuilder(unittest.TestCase):
    """测试完整 SMILES 构建功能。/ Tests for full SMILES construction."""

    def test_single_nucleotide_dna_a(self):
        """单核苷酸 A：O + terminal_A。"""
        smiles = build_full_smiles("A", "DNA")
        self.assertEqual(smiles, "O" + DNA_TERMINAL["A"])

    def test_dinucleotide_ac(self):
        """二核苷酸 AC：O + internal_A + terminal_C。"""
        smiles = build_full_smiles("AC", "DNA")
        expected = "O" + DNA_INTERNAL["A"] + DNA_TERMINAL["C"]
        self.assertEqual(smiles, expected)

    def test_trinucleotide_acg(self):
        """三核苷酸 ACG。"""
        smiles = build_full_smiles("ACG", "DNA")
        expected = "O" + DNA_INTERNAL["A"] + DNA_INTERNAL["C"] + DNA_TERMINAL["G"]
        self.assertEqual(smiles, expected)

    def test_tetranucleotide_acgt(self):
        """四核苷酸 ACGT（覆盖全部 4 种碱基）。"""
        smiles = build_full_smiles("ACGT", "DNA")
        expected = ("O" + DNA_INTERNAL["A"] + DNA_INTERNAL["C"]
                    + DNA_INTERNAL["G"] + DNA_TERMINAL["T"])
        self.assertEqual(smiles, expected)

    def test_pdl1_aptamer_45nt(self):
        """45-nt PD-L1 适配体序列应生成合法 SMILES。"""
        smiles = build_full_smiles(PDL1_APTAMER, "DNA")
        self.assertTrue(smiles.startswith("O"))
        self.assertTrue(len(smiles) > 1000)

    def test_rdkit_valid_short(self):
        """短序列 ACGT 的 SMILES 应通过 RDKit 解析。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        smiles = build_full_smiles("ACGT", "DNA")
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol, f"RDKit failed to parse: {smiles[:80]}...")

    def test_rdkit_valid_aptamer(self):
        """45-nt 适配体的 SMILES 应通过 RDKit 解析（关键验证）。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        smiles = build_full_smiles(PDL1_APTAMER, "DNA")
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol, "RDKit failed to parse 45-nt aptamer SMILES")

    def test_direction_3to5_reverses(self):
        """3'→5' 方向应反转序列。"""
        smiles_3to5 = build_full_smiles("AC", "DNA", "3to5")
        smiles_ca_5to3 = build_full_smiles("CA", "DNA", "5to3")
        self.assertEqual(smiles_3to5, smiles_ca_5to3)

    def test_rna_augc(self):
        """RNA AUGC 序列应使用 RNA 片段。"""
        smiles = build_full_smiles("AUGC", "RNA")
        expected = ("O" + RNA_INTERNAL["A"] + RNA_INTERNAL["U"]
                    + RNA_INTERNAL["G"] + RNA_TERMINAL["C"])
        self.assertEqual(smiles, expected)

    def test_rna_rdkit_valid(self):
        """RNA AUGC 的 SMILES 应通过 RDKit 解析。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        smiles = build_full_smiles("AUGC", "RNA")
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol, f"RDKit failed to parse RNA SMILES: {smiles[:80]}...")

    def test_dna_vs_rna_different(self):
        """同序列 ACG 的 DNA 与 RNA SMILES 应不同（2'-OH 差异）。"""
        dna = build_full_smiles("ACG", "DNA")
        rna = build_full_smiles("ACG", "RNA")
        self.assertNotEqual(dna, rna)

    def test_fragment_smiles(self):
        """片段 SMILES 应等于子序列的完整 SMILES。"""
        full_acg = build_full_smiles("ACG", "DNA")
        frag = build_fragment_smiles("ACGTACGT", "DNA", 0, 3)
        self.assertEqual(frag, full_acg)


# ===========================================================================
# TestBigSMILESGenerator — BigSMILES 生成器测试
# ===========================================================================

class TestBigSMILESGenerator(unittest.TestCase):
    """测试 BigSMILES 生成功能。/ Tests for BigSMILES generation."""

    def test_dna_bigsmiles_matches_example_13_1(self):
        """DNA BigSMILES 应与示例库 13.1 完全一致。"""
        from bigsmiles_examples import EXAMPLES
        example_13_1 = [e for e in EXAMPLES if e["id"] == "13.1"][0]
        self.assertEqual(generate_bigsmiles("DNA"), example_13_1["bigsmiles"])

    def test_rna_bigsmiles_matches_example_13_2(self):
        """RNA BigSMILES 应与示例库 13.2 完全一致。"""
        from bigsmiles_examples import EXAMPLES
        example_13_2 = [e for e in EXAMPLES if e["id"] == "13.2"][0]
        self.assertEqual(generate_bigsmiles("RNA"), example_13_2["bigsmiles"])

    def test_dna_bigsmiles_checker_pass(self):
        """DNA BigSMILES 应通过语法检查器。"""
        from bigsmiles_checker import check_bigsmiles
        self.assertTrue(check_bigsmiles(generate_bigsmiles("DNA"), verbose=False))

    def test_rna_bigsmiles_checker_pass(self):
        """RNA BigSMILES 应通过语法检查器。"""
        from bigsmiles_checker import check_bigsmiles
        self.assertTrue(check_bigsmiles(generate_bigsmiles("RNA"), verbose=False))

    def test_bigsmiles_sequence_independent(self):
        """BigSMILES 与输入序列无关——不同序列产生相同结果。"""
        self.assertEqual(generate_bigsmiles("DNA"), BIGSMILES_DNA)
        self.assertEqual(generate_bigsmiles("RNA"), BIGSMILES_RNA)


# ===========================================================================
# TestImageGeneration — 图像生成测试
# ===========================================================================

class TestImageGeneration(unittest.TestCase):
    """测试图像生成功能。/ Tests for image generation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_short_sequence_image(self):
        """短序列（<=6 nt）应生成完整分子图。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        smiles = build_full_smiles("ACG", "DNA")
        images = generate_images(smiles, "ACG", "DNA", self.tmpdir)
        self.assertTrue(len(images) > 0)
        for img in images:
            self.assertTrue(os.path.exists(img))

    def test_medium_sequence_image(self):
        """中等序列（7-20 nt）应生成完整+片段图。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        seq = "ACGTACGTACGT"  # 12 nt
        smiles = build_full_smiles(seq, "DNA")
        images = generate_images(smiles, seq, "DNA", self.tmpdir)
        self.assertTrue(len(images) > 0)

    def test_long_sequence_image(self):
        """长序列（>20 nt）应至少生成片段图。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        smiles = build_full_smiles(PDL1_APTAMER, "DNA")
        images = generate_images(smiles, PDL1_APTAMER, "DNA", self.tmpdir)
        self.assertTrue(len(images) > 0)

    def test_end_to_end_orchestration(self):
        """端到端测试 sequence_to_representations。"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not available")
        result = sequence_to_representations(
            "ACGT", "DNA", "5to3",
            generate_imgs=True, output_dir=self.tmpdir,
        )
        self.assertEqual(result["clean_sequence"], "ACGT")
        self.assertEqual(result["sequence_type"], "DNA")
        self.assertEqual(result["length"], 4)
        self.assertIn("bigsmiles", result)
        self.assertIn("full_smiles", result)
        self.assertTrue(result["rdkit_valid"])
        self.assertTrue(result["bigsmiles_valid"])
        self.assertTrue(len(result["images"]) > 0)


if __name__ == "__main__":
    unittest.main()
