#!/usr/bin/env python3
"""
HELM → 3D 转换工具测试
Tests for helm_to_3d.py
"""

import unittest
import os
import sys
import json
import tempfile
import shutil

from helm_to_3d import (
    HELMMonomer, HELMChain, HELMParseResult, HELMParseError,
    parse_helm, _parse_single_polymer, _parse_monomer_group,
    _validate_chain, sequence_to_helm,
    helm_chain_to_smiles, chain_to_helm_string,
    generate_3d_sdf, helm_to_3d,
)


# ===========================================================================
# Section 1: HELM 解析器测试 / Parser Tests
# ===========================================================================

class TestHELMMonomerParsing(unittest.TestCase):
    """单体解析测试 / Monomer parsing tests"""

    def test_rna_internal_uppercase(self):
        """R(A)P — RNA 内部腺嘌呤"""
        m = _parse_monomer_group("R(A)P", 1, "RNA")
        self.assertEqual(m.sugar, "R")
        self.assertEqual(m.base, "A")
        self.assertTrue(m.has_phosphate)
        self.assertEqual(m.position, 1)

    def test_rna_terminal_uppercase(self):
        """R(U) — RNA 末端尿嘧啶"""
        m = _parse_monomer_group("R(U)", 4, "RNA")
        self.assertEqual(m.sugar, "R")
        self.assertEqual(m.base, "U")
        self.assertFalse(m.has_phosphate)

    def test_rna_lowercase(self):
        """r(a)p — 小写变体"""
        m = _parse_monomer_group("r(a)p", 1, "RNA")
        self.assertEqual(m.sugar, "R")
        self.assertEqual(m.base, "A")
        self.assertTrue(m.has_phosphate)

    def test_dna_bracket_dR(self):
        """[dR](T)P — 方括号脱氧核糖"""
        m = _parse_monomer_group("[dR](T)P", 1, "DNA")
        self.assertEqual(m.sugar, "dR")
        self.assertEqual(m.base, "T")
        self.assertTrue(m.has_phosphate)

    def test_dna_bare_dR(self):
        """dR(G)P — 无方括号脱氧核糖"""
        m = _parse_monomer_group("dR(G)P", 2, "DNA")
        self.assertEqual(m.sugar, "dR")
        self.assertEqual(m.base, "G")
        self.assertTrue(m.has_phosphate)

    def test_dna_terminal(self):
        """[dR](C) — DNA 末端"""
        m = _parse_monomer_group("[dR](C)", 3, "DNA")
        self.assertEqual(m.sugar, "dR")
        self.assertEqual(m.base, "C")
        self.assertFalse(m.has_phosphate)

    def test_all_bases(self):
        """所有碱基都能解析"""
        for base in "ACGTU":
            m = _parse_monomer_group(f"R({base})P", 1, "RNA")
            self.assertEqual(m.base, base)

    def test_invalid_monomer(self):
        """无效单体格式应报错"""
        with self.assertRaises(HELMParseError):
            _parse_monomer_group("X(A)P", 1, "RNA")

    def test_invalid_base(self):
        """无效碱基应报错"""
        with self.assertRaises(HELMParseError):
            _parse_monomer_group("R(X)P", 1, "RNA")

    def test_r_defaults_to_dR_for_dna(self):
        """R(A)P 在 DNA 上下文中应推断为 dR"""
        m = _parse_monomer_group("R(A)P", 1, "DNA")
        self.assertEqual(m.sugar, "dR")


class TestHELMChainParsing(unittest.TestCase):
    """链解析测试 / Chain parsing tests"""

    def test_simple_rna(self):
        """RNA1{R(A)P.R(C)P.R(G)P.R(U)}"""
        chain = _parse_single_polymer("RNA1{R(A)P.R(C)P.R(G)P.R(U)}")
        self.assertEqual(chain.polymer_id, "RNA1")
        self.assertEqual(chain.chain_type, "RNA")
        self.assertEqual(chain.length, 4)
        self.assertEqual(chain.sequence, "ACGU")

    def test_simple_dna(self):
        """DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}"""
        chain = _parse_single_polymer("DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}")
        self.assertEqual(chain.chain_type, "DNA")
        self.assertEqual(chain.sequence, "ACGT")

    def test_dna_bare_dR(self):
        """DNA1{dR(A)P.dR(T)} — 无方括号 dR"""
        chain = _parse_single_polymer("DNA1{dR(A)P.dR(T)}")
        self.assertEqual(chain.chain_type, "DNA")
        self.assertEqual(chain.sequence, "AT")

    def test_single_nucleotide(self):
        """单核苷酸链"""
        chain = _parse_single_polymer("RNA1{R(A)}")
        self.assertEqual(chain.length, 1)
        self.assertEqual(chain.sequence, "A")
        self.assertFalse(chain.monomers[0].has_phosphate)

    def test_long_chain(self):
        """12 个核苷酸的链"""
        monomers = ".".join(
            f"R({b})P" if i < 11 else f"R({b})"
            for i, b in enumerate("ACGUACGUACGU")
        )
        chain = _parse_single_polymer(f"RNA1{{{monomers}}}")
        self.assertEqual(chain.length, 12)
        self.assertEqual(chain.sequence, "ACGUACGUACGU")

    def test_terminal_phosphate_auto_corrected(self):
        """末端磷酸自动移除"""
        chain = _parse_single_polymer("RNA1{R(A)P.R(C)P}")
        self.assertFalse(chain.monomers[-1].has_phosphate)

    def test_invalid_format(self):
        """无效格式应报错"""
        with self.assertRaises(HELMParseError):
            _parse_single_polymer("INVALID_FORMAT")


class TestHELMFullParsing(unittest.TestCase):
    """完整 HELM 字符串解析测试"""

    def test_standard_rna(self):
        """标准 RNA HELM"""
        result = parse_helm("RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$")
        self.assertEqual(len(result.chains), 1)
        self.assertEqual(result.chains[0].sequence, "ACGU")

    def test_standard_dna(self):
        """标准 DNA HELM"""
        result = parse_helm("DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}$$$$")
        self.assertEqual(result.chains[0].chain_type, "DNA")
        self.assertEqual(result.chains[0].sequence, "ACGT")

    def test_helm_v2(self):
        """HELM V2.0 版本标记"""
        result = parse_helm("RNA1{R(A)P.R(U)}$$$$V2.0")
        self.assertEqual(result.version, "V2.0")
        self.assertEqual(result.chains[0].sequence, "AU")

    def test_missing_delimiters(self):
        """缺少分隔符应报错"""
        with self.assertRaises(HELMParseError):
            parse_helm("RNA1{R(A)P.R(U)}")

    def test_empty_polymer(self):
        """空聚合物段应报错"""
        with self.assertRaises(HELMParseError):
            parse_helm("$$$$")


class TestChainValidation(unittest.TestCase):
    """链验证测试"""

    def test_dna_with_uracil_rejected(self):
        """DNA 链含 U 应报错"""
        chain = HELMChain(
            polymer_id="DNA1", chain_type="DNA",
            monomers=[HELMMonomer("dR", "U", False, 1)],
        )
        with self.assertRaises(HELMParseError):
            _validate_chain(chain)

    def test_rna_with_thymine_rejected(self):
        """RNA 链含 T 应报错"""
        chain = HELMChain(
            polymer_id="RNA1", chain_type="RNA",
            monomers=[HELMMonomer("R", "T", False, 1)],
        )
        with self.assertRaises(HELMParseError):
            _validate_chain(chain)

    def test_empty_chain_rejected(self):
        """空链应报错"""
        chain = HELMChain(polymer_id="RNA1", chain_type="RNA")
        with self.assertRaises(HELMParseError):
            _validate_chain(chain)


# ===========================================================================
# Section 2: 序列 → HELM 转换测试 / Sequence to HELM Tests
# ===========================================================================

class TestSequenceToHelm(unittest.TestCase):
    """简化序列 → HELM 转换测试"""

    def test_dna_sequence(self):
        """ACGT → DNA HELM"""
        helm = sequence_to_helm("ACGT", "DNA")
        self.assertIn("DNA1{", helm)
        self.assertIn("[dR](A)P", helm)
        self.assertIn("[dR](T)", helm)  # 末端无 P
        self.assertTrue(helm.endswith("$$$$"))

    def test_rna_sequence(self):
        """ACGU → RNA HELM"""
        helm = sequence_to_helm("ACGU", "RNA")
        self.assertIn("RNA1{", helm)
        self.assertIn("R(A)P", helm)
        self.assertIn("R(U)", helm)

    def test_single_base(self):
        """单碱基序列"""
        helm = sequence_to_helm("A", "DNA")
        self.assertIn("[dR](A)", helm)
        self.assertNotIn("P", helm.split("{")[1].split("}")[0])

    def test_lowercase_input(self):
        """小写输入自动转大写"""
        helm = sequence_to_helm("acgt", "DNA")
        self.assertIn("[dR](A)P", helm)

    def test_direction_markers_stripped(self):
        """去除方向标记"""
        helm = sequence_to_helm("5'-ACGT-3'", "DNA")
        result = parse_helm(helm)
        self.assertEqual(result.chains[0].sequence, "ACGT")

    def test_invalid_base_rejected(self):
        """无效碱基应报错"""
        with self.assertRaises(ValueError):
            sequence_to_helm("ACGX", "DNA")

    def test_empty_sequence_rejected(self):
        """空序列应报错"""
        with self.assertRaises(ValueError):
            sequence_to_helm("", "DNA")

    def test_dna_u_rejected(self):
        """DNA 中的 U 应报错"""
        with self.assertRaises(ValueError):
            sequence_to_helm("ACGU", "DNA")

    def test_rna_t_rejected(self):
        """RNA 中的 T 应报错"""
        with self.assertRaises(ValueError):
            sequence_to_helm("ACGT", "RNA")


# ===========================================================================
# Section 3: SMILES 构建测试 / SMILES Builder Tests
# ===========================================================================

class TestSMILESBuilder(unittest.TestCase):
    """SMILES 构建测试"""

    def _make_chain(self, seq, chain_type):
        """辅助：从序列构建 HELMChain"""
        sugar = "R" if chain_type == "RNA" else "dR"
        pid = "RNA1" if chain_type == "RNA" else "DNA1"
        monomers = [
            HELMMonomer(sugar, b, i < len(seq) - 1, i + 1)
            for i, b in enumerate(seq)
        ]
        return HELMChain(polymer_id=pid, chain_type=chain_type, monomers=monomers)

    def test_dna_starts_with_O(self):
        """SMILES 以 5'-OH 'O' 开头"""
        chain = self._make_chain("ACGT", "DNA")
        smiles = helm_chain_to_smiles(chain)
        self.assertTrue(smiles.startswith("O"))

    def test_rna_starts_with_O(self):
        """RNA SMILES 以 'O' 开头"""
        chain = self._make_chain("ACGU", "RNA")
        smiles = helm_chain_to_smiles(chain)
        self.assertTrue(smiles.startswith("O"))

    def test_dna_contains_phosphate(self):
        """DNA 内部含磷酸基"""
        chain = self._make_chain("AC", "DNA")
        smiles = helm_chain_to_smiles(chain)
        self.assertIn("P(=O)(O)O", smiles)

    def test_rna_contains_2prime_oh(self):
        """RNA 含 2'-OH"""
        chain = self._make_chain("AU", "RNA")
        smiles = helm_chain_to_smiles(chain)
        self.assertIn("C(O)C", smiles)

    def test_single_nucleotide_no_phosphate(self):
        """单核苷酸无磷酸"""
        chain = self._make_chain("A", "DNA")
        smiles = helm_chain_to_smiles(chain)
        self.assertNotIn("P(=O)", smiles)

    def test_smiles_length_grows_with_sequence(self):
        """SMILES 长度随序列增长"""
        chain2 = self._make_chain("AC", "DNA")
        chain4 = self._make_chain("ACGT", "DNA")
        self.assertGreater(len(helm_chain_to_smiles(chain4)),
                           len(helm_chain_to_smiles(chain2)))

    def test_rdkit_validates_dna_smiles(self):
        """RDKit 验证 DNA SMILES 有效性"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not installed")
        chain = self._make_chain("ACGT", "DNA")
        smiles = helm_chain_to_smiles(chain)
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)

    def test_rdkit_validates_rna_smiles(self):
        """RDKit 验证 RNA SMILES 有效性"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not installed")
        chain = self._make_chain("ACGU", "RNA")
        smiles = helm_chain_to_smiles(chain)
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol)


# ===========================================================================
# Section 4: HELM 字符串重建测试 / HELM Reconstruction Tests
# ===========================================================================

class TestHELMReconstruction(unittest.TestCase):
    """HELM 字符串重建测试"""

    def test_rna_round_trip(self):
        """RNA HELM 解析后重建"""
        original = "RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$"
        result = parse_helm(original)
        rebuilt = chain_to_helm_string(result.chains[0])
        self.assertEqual(rebuilt, original)

    def test_dna_round_trip(self):
        """DNA HELM 解析后重建"""
        original = "DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}$$$$"
        result = parse_helm(original)
        rebuilt = chain_to_helm_string(result.chains[0])
        self.assertEqual(rebuilt, original)

    def test_sequence_helm_round_trip(self):
        """序列 → HELM → 解析 → 序列 round trip"""
        helm = sequence_to_helm("ACGT", "DNA")
        result = parse_helm(helm)
        self.assertEqual(result.chains[0].sequence, "ACGT")


# ===========================================================================
# Section 5: 端到端测试 / End-to-End Tests
# ===========================================================================

class TestEndToEnd(unittest.TestCase):
    """端到端集成测试"""

    def test_helm_to_3d_with_helm_input(self):
        """HELM 输入 → 结果字典"""
        result = helm_to_3d(
            "RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$",
            output_path=os.path.join(tempfile.gettempdir(), "test_helm.sdf"),
        )
        self.assertEqual(result["chain_type"], "RNA")
        self.assertEqual(result["sequence"], "ACGU")
        self.assertEqual(result["length"], 4)
        self.assertIn("smiles", result)
        # sdf_path 可能为 None（如果 RDKit 未安装）
        if result["rdkit_valid"]:
            self.assertIsNotNone(result["sdf_path"])

    def test_helm_to_3d_with_sequence_input(self):
        """简化序列输入 → 结果字典"""
        result = helm_to_3d(
            "ACGT", seq_type="DNA",
            output_path=os.path.join(tempfile.gettempdir(), "test_seq.sdf"),
        )
        self.assertEqual(result["chain_type"], "DNA")
        self.assertEqual(result["sequence"], "ACGT")

    def test_helm_to_3d_auto_detect_rna(self):
        """含 U 的序列自动推断为 RNA"""
        result = helm_to_3d(
            "AUGC",
            output_path=os.path.join(tempfile.gettempdir(), "test_auto.sdf"),
        )
        self.assertEqual(result["chain_type"], "RNA")

    def test_sdf_file_created(self):
        """SDF 文件实际生成"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not installed")

        sdf_path = os.path.join(tempfile.gettempdir(), "test_3d_output.sdf")
        result = helm_to_3d("ACGT", seq_type="DNA", output_path=sdf_path)
        self.assertTrue(os.path.exists(sdf_path))
        self.assertGreater(os.path.getsize(sdf_path), 0)

        # 验证 SDF 内容
        with open(sdf_path, "r") as f:
            content = f.read()
        self.assertIn("$$$$", content)
        self.assertIn("<Sequence>", content)
        self.assertIn("ACGT", content)

        # 清理
        os.remove(sdf_path)
        json_path = sdf_path.replace(".sdf", ".json")
        if os.path.exists(json_path):
            os.remove(json_path)

    def test_json_record_created(self):
        """JSON 记录文件同步生成"""
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit not installed")

        sdf_path = os.path.join(tempfile.gettempdir(), "test_json_rec.sdf")
        json_path = sdf_path.replace(".sdf", ".json")
        result = helm_to_3d("ACG", seq_type="RNA", output_path=sdf_path)

        if result.get("sdf_path"):
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r") as f:
                record = json.load(f)
            self.assertEqual(record["sequence"], "ACG")
            self.assertIn("timestamp", record)

            os.remove(sdf_path)
            os.remove(json_path)


# ===========================================================================
# Section 6: 边界与错误测试 / Edge Cases & Error Tests
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """边界与错误处理测试"""

    def test_whitespace_handling(self):
        """输入含前后空白"""
        result = parse_helm("  RNA1{R(A)P.R(U)}$$$$  ")
        self.assertEqual(result.chains[0].sequence, "AU")

    def test_monomer_whitespace(self):
        """单体间含空白"""
        m = _parse_monomer_group("  R(A)P  ", 1, "RNA")
        self.assertEqual(m.base, "A")

    def test_case_insensitive_base(self):
        """小写碱基可解析"""
        m = _parse_monomer_group("r(a)p", 1, "RNA")
        self.assertEqual(m.base, "A")

    def test_helm_error_is_valueerror(self):
        """HELMParseError 是 ValueError 的子类"""
        self.assertTrue(issubclass(HELMParseError, ValueError))

    def test_chain_properties(self):
        """HELMChain 属性正确"""
        chain = HELMChain(
            polymer_id="RNA1", chain_type="RNA",
            monomers=[
                HELMMonomer("R", "A", True, 1),
                HELMMonomer("R", "C", False, 2),
            ],
        )
        self.assertEqual(chain.sequence, "AC")
        self.assertEqual(chain.length, 2)


if __name__ == "__main__":
    unittest.main()
