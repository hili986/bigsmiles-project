#!/usr/bin/env python3
"""
HELM 核酸序列 → 3D 分子模型转换工具
HELM Nucleic Acid Sequence → 3D Molecular Model Converter

HELM (Hierarchical Editing Language for Macromolecules) 是一种用于表示
复杂生物大分子结构的层次化标记语言。本脚本解析 HELM 核酸表示法，
将其转换为 SMILES，并生成 3D 分子结构文件 (.sdf)。

支持两种输入格式:
  1. 标准 HELM: RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$
  2. 简化序列: ACGU --type RNA

用法 / Usage:
  python helm_to_3d.py "RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$"
  python helm_to_3d.py "DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}$$$$"
  python helm_to_3d.py ACGT --type DNA
  python helm_to_3d.py AUGC --type RNA
  python helm_to_3d.py ACGT --type DNA -o my_dna.sdf
"""

import os
import re
import sys
import json
import argparse
from dataclasses import dataclass, field
from datetime import datetime


# ===========================================================================
# Section 1: 数据结构 / Data Structures
# ===========================================================================

@dataclass
class HELMMonomer:
    """HELM 单体 / HELM Monomer"""
    sugar: str            # "R" (ribose) or "dR" (deoxyribose)
    base: str             # "A", "C", "G", "T", "U"
    has_phosphate: bool   # True for internal, False for 3' terminal
    position: int         # 1-indexed position in chain


@dataclass
class HELMChain:
    """HELM 核酸链 / HELM Nucleic Acid Chain"""
    polymer_id: str              # e.g., "RNA1", "DNA1"
    chain_type: str              # "RNA" or "DNA"
    monomers: list = field(default_factory=list)

    @property
    def sequence(self):
        """返回碱基序列字符串。"""
        return "".join(m.base for m in self.monomers)

    @property
    def length(self):
        """返回链长度（核苷酸数）。"""
        return len(self.monomers)


@dataclass
class HELMParseResult:
    """HELM 解析结果 / HELM Parse Result"""
    raw_input: str
    chains: list = field(default_factory=list)
    connections: str = ""
    groups: str = ""
    annotations: str = ""
    version: str = ""


# ===========================================================================
# Section 2: HELM 解析器 / HELM Parser
# ===========================================================================

class HELMParseError(ValueError):
    """HELM 解析错误 / HELM Parse Error"""
    pass


def parse_helm(helm_string):
    """
    解析 HELM 字符串为结构化数据。
    Parse a HELM string into structured data.

    支持格式 / Supported formats:
      - HELM 2.0: RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$V2.0
      - HELM 1.0: RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$
      - 小写变体: RNA1{r(A)p.r(C)p.r(G)p.r(U)}$$$$

    Parameters:
        helm_string: HELM 格式字符串
    Returns:
        HELMParseResult 对象
    Raises:
        HELMParseError: 格式不合法
    """
    helm_string = helm_string.strip()
    result = HELMParseResult(raw_input=helm_string)

    # 检查版本号 / Check version
    if helm_string.upper().endswith("V2.0"):
        result.version = "V2.0"
        helm_string = helm_string[:-4]

    # 按 $ 分割四个段 / Split into sections by $
    sections = helm_string.split("$")
    if len(sections) < 5:
        raise HELMParseError(
            f"HELM 格式错误：需要至少 4 个 '$' 分隔符，实际得到 {len(sections) - 1} 个\n"
            f"HELM format error: expected at least 4 '$' delimiters, got {len(sections) - 1}\n"
            f"正确格式 / Correct format: POLYMER{{monomers}}$$$$"
        )

    polymer_section = sections[0]
    result.connections = sections[1]
    result.groups = sections[2]
    result.annotations = sections[3]

    # 解析聚合物列表（多链用 | 分隔）/ Parse polymer list
    polymer_parts = polymer_section.split("|")
    for part in polymer_parts:
        part = part.strip()
        if not part:
            continue
        chain = _parse_single_polymer(part)
        result.chains.append(chain)

    if not result.chains:
        raise HELMParseError(
            "未找到有效的核酸链 / No valid nucleic acid chain found"
        )

    return result


def _parse_single_polymer(polymer_str):
    """
    解析单个聚合物段。
    Parse a single polymer section, e.g. RNA1{R(A)P.R(C)P.R(U)}

    Parameters:
        polymer_str: 聚合物字符串
    Returns:
        HELMChain 对象
    """
    match = re.match(r'^([A-Za-z]+\d*)\{(.+)\}$', polymer_str)
    if not match:
        raise HELMParseError(
            f"聚合物格式错误 / Invalid polymer format: '{polymer_str}'\n"
            f"期望格式 / Expected: POLYMER_ID{{monomers}}"
        )

    polymer_id = match.group(1)
    monomer_str = match.group(2)

    # 推断链类型 / Infer chain type from ID
    id_upper = polymer_id.upper()
    if id_upper.startswith("DNA"):
        chain_type = "DNA"
    else:
        chain_type = "RNA"

    chain = HELMChain(polymer_id=polymer_id, chain_type=chain_type)

    # 按 . 分割单体组 / Split monomer groups by .
    groups = monomer_str.split(".")
    for i, group in enumerate(groups):
        monomer = _parse_monomer_group(group, i + 1, chain_type)
        chain.monomers.append(monomer)

    # 从单体糖类型修正链类型 / Correct chain type from sugar types
    has_deoxy = any(m.sugar == "dR" for m in chain.monomers)
    has_ribose = any(m.sugar == "R" for m in chain.monomers)
    if has_deoxy and not has_ribose:
        chain.chain_type = "DNA"
    elif has_ribose and not has_deoxy:
        chain.chain_type = "RNA"

    _validate_chain(chain)
    return chain


def _parse_monomer_group(group, position, default_type):
    """
    解析单个单体组。
    Parse a single monomer group, e.g. R(A)P, [dR](T)P, dR(G), r(c)p

    Parameters:
        group: 单体组字符串
        position: 在链中的位置（1-indexed）
        default_type: 默认链类型 ("RNA" / "DNA")
    Returns:
        HELMMonomer 对象
    """
    group = group.strip()
    g = group.upper()

    # 模式 1: [DR](X)P 或 [DR](X) — 方括号脱氧核糖
    m = re.match(r'^\[DR\]\(([ACGTU])\)(P?)$', g)
    if m:
        return HELMMonomer(
            sugar="dR", base=m.group(1),
            has_phosphate=bool(m.group(2)), position=position,
        )

    # 模式 2: DR(X)P 或 DR(X) — 脱氧核糖（无方括号）
    m = re.match(r'^DR\(([ACGTU])\)(P?)$', g)
    if m:
        return HELMMonomer(
            sugar="dR", base=m.group(1),
            has_phosphate=bool(m.group(2)), position=position,
        )

    # 模式 3: R(X)P 或 R(X) — 核糖（RNA）或根据 default_type 推断
    m = re.match(r'^R\(([ACGTU])\)(P?)$', g)
    if m:
        sugar = "R" if default_type == "RNA" else "dR"
        return HELMMonomer(
            sugar=sugar, base=m.group(1),
            has_phosphate=bool(m.group(2)), position=position,
        )

    raise HELMParseError(
        f"无法解析单体 / Cannot parse monomer: '{group}' (位置 / position {position})\n"
        f"支持格式 / Supported: R(A)P, R(A), dR(T)P, [dR](G)P"
    )


def _validate_chain(chain):
    """
    验证核酸链的化学合理性。
    Validate chemical correctness of a nucleic acid chain.
    """
    if not chain.monomers:
        raise HELMParseError("核酸链为空 / Empty nucleic acid chain")

    # 末端单体不应有磷酸（自动修正）/ Terminal should not have phosphate
    if chain.monomers[-1].has_phosphate:
        chain.monomers[-1].has_phosphate = False

    # 碱基与链类型匹配 / Base-type compatibility
    for m in chain.monomers:
        if chain.chain_type == "DNA" and m.base == "U":
            raise HELMParseError(
                f"DNA 链含有 U（尿嘧啶），应使用 T / "
                f"DNA chain contains U (uracil), use T instead"
            )
        if chain.chain_type == "RNA" and m.base == "T":
            raise HELMParseError(
                f"RNA 链含有 T（胸腺嘧啶），应使用 U / "
                f"RNA chain contains T (thymine), use U instead"
            )


def sequence_to_helm(seq, seq_type="DNA"):
    """
    将简化碱基序列转换为 HELM 表示法。
    Convert a plain base sequence to HELM notation.

    Parameters:
        seq: 碱基序列，如 "ACGT"
        seq_type: "DNA" 或 "RNA"
    Returns:
        HELM 字符串
    """
    seq = seq.strip().upper()
    seq = re.sub(r"^[35]'[- ]?", "", seq)
    seq = re.sub(r"[- ]?[35]'$", "", seq)

    valid_bases = set("ATGC") if seq_type == "DNA" else set("AUGC")
    for i, base in enumerate(seq):
        if base not in valid_bases:
            raise ValueError(
                f"无效碱基 / Invalid base '{base}' at position {i + 1} "
                f"for {seq_type}. Valid: {', '.join(sorted(valid_bases))}"
            )
    if not seq:
        raise ValueError("序列为空 / Empty sequence")

    sugar = "[dR]" if seq_type == "DNA" else "R"
    polymer_id = "DNA1" if seq_type == "DNA" else "RNA1"

    groups = []
    for i, base in enumerate(seq):
        suffix = "P" if i < len(seq) - 1 else ""
        groups.append(f"{sugar}({base}){suffix}")

    return f"{polymer_id}{{{'.'.join(groups)}}}$$$$"


# ===========================================================================
# Section 3: 核苷酸 SMILES 片段库 / Nucleotide SMILES Fragment Library
# ===========================================================================

# DNA 脱氧核苷酸 — 内部（含 3',5'-磷酸二酯连接）
# DNA deoxyribonucleotides — internal (with phosphodiester linkage)
_DNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O",
}

# DNA 脱氧核苷酸 — 3' 末端（无磷酸，3'-OH）
# DNA deoxyribonucleotides — 3' terminal (free 3'-OH)
_DNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3O",
}

# RNA 核糖核苷酸 — 内部（含 2'-OH 和磷酸二酯）
# RNA ribonucleotides — internal (with 2'-OH and phosphodiester)
_RNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3OP(=O)(O)O",
}

# RNA 核糖核苷酸 — 3' 末端
# RNA ribonucleotides — 3' terminal
_RNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3O",
}


def helm_chain_to_smiles(chain):
    """
    将 HELM 核酸链转换为完整原子级 SMILES。
    Convert a HELM nucleic acid chain to full atom-level SMILES.

    拼接逻辑 / Concatenation logic:
      smiles = "O" (5'-OH)
             + internal[base] x (N-1)
             + terminal[base] x 1

    Parameters:
        chain: HELMChain 对象
    Returns:
        完整 SMILES 字符串
    """
    if chain.chain_type == "DNA":
        internal, terminal = _DNA_INTERNAL, _DNA_TERMINAL
    else:
        internal, terminal = _RNA_INTERNAL, _RNA_TERMINAL

    smiles = "O"  # 5'-OH
    for i, monomer in enumerate(chain.monomers):
        if i < len(chain.monomers) - 1:
            smiles += internal[monomer.base]
        else:
            smiles += terminal[monomer.base]

    return smiles


def chain_to_helm_string(chain):
    """
    从 HELMChain 对象重建 HELM 字符串。
    Reconstruct HELM string from a HELMChain object.
    """
    sugar = "R" if chain.chain_type == "RNA" else "[dR]"
    groups = []
    for i, m in enumerate(chain.monomers):
        suffix = "P" if i < len(chain.monomers) - 1 else ""
        groups.append(f"{sugar}({m.base}){suffix}")
    return f"{chain.polymer_id}{{{'.'.join(groups)}}}$$$$"


# ===========================================================================
# Section 4: 3D 结构生成器 / 3D Structure Generator
# ===========================================================================

def _format_time(seconds):
    """格式化时间估计。/ Format time estimate."""
    if seconds < 60:
        return f"~{max(1, int(seconds))} 秒 / sec"
    elif seconds < 3600:
        return f"~{seconds / 60:.1f} 分钟 / min"
    return f"~{seconds / 3600:.1f} 小时 / hr"


def generate_3d_sdf(smiles, chain, output_path):
    """
    使用 RDKit ETKDG 算法生成 3D 构象，导出为 SDF 文件。
    Generate 3D conformation via RDKit ETKDG, export as SDF.

    流程 / Pipeline:
      SMILES → Mol → AddHs → ETKDG embed → UFF/MMFF optimize → RemoveHs → SDF

    Parameters:
        smiles: 完整 SMILES 字符串
        chain: HELMChain 对象
        output_path: 输出 SDF 文件路径
    Returns:
        成功返回文件路径，失败返回 None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("[ERROR] RDKit 未安装 / RDKit not installed")
        print("  安装方法 / Install: conda install -c conda-forge rdkit")
        return None

    # SMILES → Mol 对象 / Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("[ERROR] SMILES 解析失败 / SMILES parsing failed")
        return None

    mol = Chem.AddHs(mol)
    n = chain.length

    # 预估时间 / Estimate time
    t_est = 0.008 * n ** 3
    print(f"[INFO] 预估耗时 / ETA: {_format_time(t_est)}")

    # ETKDG 3D 嵌入 / 3D embedding via ETKDG
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if n > 4:
        params.useRandomCoords = True
    params.maxIterations = max(5000, n * 200)

    print("[...] 3D 嵌入中 / 3D embedding ...", end="", flush=True)
    result = AllChem.EmbedMolecule(mol, params)

    if result == -1:
        # 回退：增大迭代次数 / Fallback with more iterations
        params.useRandomCoords = True
        params.maxIterations = max(10000, n * 500)
        result = AllChem.EmbedMolecule(mol, params)

    if result == -1:
        print(" 失败 / FAILED")
        print("[ERROR] 3D 嵌入失败，序列可能过长 / "
              "3D embedding failed, sequence may be too long")
        return None
    print(" 完成 / done")

    # 力场优化 / Force-field optimization (UFF → MMFF fallback)
    print("[...] 力场优化中 / Optimizing ...", end="", flush=True)
    max_iters = max(2000, n * 100)
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception:
            print(" (跳过 / skipped)", end="")
    print(" 完成 / done")

    mol = Chem.RemoveHs(mol)

    # 设置分子属性 / Set molecule properties
    helm_str = chain_to_helm_string(chain)
    mol.SetProp("HELM", helm_str)
    mol.SetProp("Sequence", chain.sequence)
    mol.SetProp("Type", chain.chain_type)
    mol.SetProp("Length", str(n))

    # 写入 SDF 文件 / Write SDF file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mol_block = Chem.MolToMolBlock(mol)
    props = (
        f">  <HELM>\n{helm_str}\n\n"
        f">  <Sequence>\n{chain.sequence}\n\n"
        f">  <Type>\n{chain.chain_type}\n\n"
        f">  <Length>\n{n}\n\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mol_block)
        f.write(props)
        f.write("$$$$\n")

    print(f"[OK] 3D 结构已保存 / 3D structure saved: {output_path}")
    print(f"     用 PyMOL / ChimeraX / 3Dmol.js 打开查看空间结构")
    return output_path


# ===========================================================================
# Section 5: 主编排函数 / Main Orchestration
# ===========================================================================

def helm_to_3d(input_str, seq_type=None, output_path=None):
    """
    主编排函数：HELM/序列 → 解析 → SMILES → 3D SDF。
    Main orchestration: HELM/sequence → parse → SMILES → 3D SDF.

    Parameters:
        input_str: HELM 字符串或简化碱基序列
        seq_type: 序列类型 ("DNA" / "RNA")，仅简化输入时需要
        output_path: 输出 SDF 文件路径（可选，自动生成）
    Returns:
        结果字典
    """
    # 判断输入格式 / Detect input format
    is_helm = "$" in input_str and "{" in input_str

    if is_helm:
        print("[INFO] 检测到 HELM 输入 / HELM input detected")
        helm_result = parse_helm(input_str)
    else:
        # 简化序列 → 自动转为 HELM / Sequence → auto-convert to HELM
        if seq_type is None:
            seq_type = "RNA" if "U" in input_str.upper() else "DNA"
        print(f"[INFO] 简化序列输入，转为 HELM / Converting sequence to HELM")
        helm_string = sequence_to_helm(input_str, seq_type)
        print(f"[INFO] HELM: {helm_string}")
        helm_result = parse_helm(helm_string)

    chain = helm_result.chains[0]

    # 信息输出 / Print info
    print(f"\n{'=' * 60}")
    print(f"  HELM 核酸 -> 3D 转换 / HELM Nucleic Acid -> 3D Conversion")
    print(f"{'=' * 60}")
    print(f"  链 ID / Chain ID:  {chain.polymer_id}")
    print(f"  链类型 / Type:     {chain.chain_type}")
    print(f"  序列 / Sequence:   {chain.sequence}")
    print(f"  长度 / Length:     {chain.length} nt")
    print(f"  HELM:              {chain_to_helm_string(chain)}")
    print(f"{'=' * 60}\n")

    # 构建 SMILES / Build SMILES
    smiles = helm_chain_to_smiles(chain)
    print(f"[INFO] Full SMILES 已构建 / built ({len(smiles)} chars)")

    # RDKit 验证 / Validate with RDKit
    rdkit_valid = None
    atom_count = None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        rdkit_valid = mol is not None
        if mol:
            atom_count = mol.GetNumAtoms()
        print(f"[INFO] RDKit 验证 / validation: "
              f"{'PASS' if rdkit_valid else 'FAIL'}"
              f"{f', {atom_count} atoms' if atom_count else ''}")
    except ImportError:
        print("[WARN] RDKit 未安装，跳过验证 / RDKit not installed, skipping validation")

    # 确定输出路径 / Determine output path
    if output_path is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "helm_3d"
        )
        os.makedirs(out_dir, exist_ok=True)
        seq = chain.sequence
        if chain.length <= 20:
            seq_tag = seq
        else:
            seq_tag = f"{seq[:6]}...{seq[-3:]}_({chain.length}nt)"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            out_dir, f"{chain.chain_type}_{seq_tag}_{timestamp}.sdf"
        )

    # 生成 3D 结构 / Generate 3D structure
    sdf_path = generate_3d_sdf(smiles, chain, output_path)

    result = {
        "helm": chain_to_helm_string(chain),
        "chain_type": chain.chain_type,
        "sequence": chain.sequence,
        "length": chain.length,
        "smiles": smiles,
        "rdkit_valid": rdkit_valid,
        "atom_count": atom_count,
        "sdf_path": sdf_path,
    }

    # 保存 JSON 记录 / Save JSON record
    if sdf_path:
        json_path = sdf_path.replace(".sdf", ".json")
        record = {**result, "timestamp": datetime.now().isoformat()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[OK] 记录已保存 / Record saved: {json_path}")

    return result


# ===========================================================================
# Section 6: CLI 接口 / CLI Interface
# ===========================================================================

def main():
    """命令行入口。/ CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "HELM 核酸序列 -> 3D 分子模型转换器\n"
            "HELM Nucleic Acid -> 3D Molecular Model Converter"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例 / Examples:\n"
            '  python helm_to_3d.py "RNA1{R(A)P.R(C)P.R(G)P.R(U)}$$$$"\n'
            '  python helm_to_3d.py "DNA1{[dR](A)P.[dR](C)P.[dR](G)P.[dR](T)}$$$$"\n'
            "  python helm_to_3d.py ACGT --type DNA\n"
            "  python helm_to_3d.py AUGC --type RNA\n"
            '  python helm_to_3d.py ACGT --type DNA -o my_dna.sdf\n'
        ),
    )
    parser.add_argument(
        "input",
        help='HELM 字符串或核酸序列 / HELM string or nucleic acid sequence',
    )
    parser.add_argument(
        "--type", choices=["DNA", "RNA"], default=None,
        help="序列类型（简化输入时使用）/ Sequence type for simplified input",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出 SDF 文件路径 / Output SDF file path",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 输出结果摘要 / Output result summary as JSON",
    )

    args = parser.parse_args()

    try:
        result = helm_to_3d(args.input, seq_type=args.type, output_path=args.output)
    except (HELMParseError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'=' * 60}")
        if result.get("sdf_path"):
            print(f"  3D 模型已生成 / 3D model generated successfully")
            print(f"  路径 / Path: {result['sdf_path']}")
            print(f"  打开方式 / Open with: PyMOL, ChimeraX, 3Dmol.js")
        else:
            print(f"  3D 生成失败 / 3D generation failed")
            print(f"  请确认 RDKit 已安装 / Ensure RDKit is installed")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
