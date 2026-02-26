#!/usr/bin/env python3
"""
核酸序列 → BigSMILES / Full SMILES 转换工具
Nucleic Acid Sequence → BigSMILES / Full SMILES Conversion Tool

双表示混合策略 / Dual-Representation Hybrid Strategy:
  1. BigSMILES — 描述聚合物类别（DNA/RNA），与序列无关
  2. Full SMILES — 逐核苷酸拼接完整原子级 SMILES，编码精确序列

用法 / Usage:
  python sequence_to_bigsmiles.py ACGT
  python sequence_to_bigsmiles.py ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT
  python sequence_to_bigsmiles.py AUGC --type RNA
  python sequence_to_bigsmiles.py ACGT --no-images --json
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime


# ===========================================================================
# Section 1: 常量 & 核苷酸字典 / Constants & Nucleotide Dictionaries
# ===========================================================================

DNA_BASES = set("ATGC")
RNA_BASES = set("AUGC")

# --- DNA 脱氧核苷酸 SMILES 片段 / DNA Deoxyribonucleotide SMILES Fragments ---
# 内部核苷酸：含 3',5'-磷酸二酯桥连 / Internal: with phosphodiester linkage
DNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O",
}

# 3' 末端核苷酸：无尾部磷酸 / Terminal: free 3'-OH, no trailing phosphate
DNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3O",
}

# --- RNA 核糖核苷酸 SMILES 片段 / RNA Ribonucleotide SMILES Fragments ---
# 差异: CC → C(O)C（2'-OH）, T → U（无5-甲基）
RNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3OP(=O)(O)O",
}

RNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3O",
}

# --- BigSMILES 表示（复用示例库 13.1 / 13.2）/ BigSMILES (from examples 13.1/13.2) ---
BIGSMILES_DNA = (
    "O{[>]CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O[<],"
    "[>]CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3OP(=O)(O)O[<],"
    "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3OP(=O)(O)O[<],"
    "[>]CC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O[<];"
    "[>]CC3OC(n1cnc2c(N)ncnc12)CC3O,"
    "[>]CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3O,"
    "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3O,"
    "[>]CC3OC(n1ccc(N)nc1=O)CC3O}"
)

BIGSMILES_RNA = (
    "O{[>]CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O[<],"
    "[>]CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3OP(=O)(O)O[<],"
    "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3OP(=O)(O)O[<],"
    "[>]CC3OC(n1ccc(N)nc1=O)C(O)C3OP(=O)(O)O[<];"
    "[>]CC3OC(n1cnc2c(N)ncnc12)C(O)C3O,"
    "[>]CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3O,"
    "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3O,"
    "[>]CC3OC(n1ccc(N)nc1=O)C(O)C3O}"
)


# ===========================================================================
# Section 2: 序列解析器 & 验证器 / Sequence Parser & Validator
# ===========================================================================

def validate_sequence(seq, seq_type="DNA"):
    """
    验证并清洗核酸序列。/ Validate and clean a nucleic acid sequence.

    - 去除首尾空白、转大写
    - 去除方向标记（如 5'-, -3'）
    - 逐碱基检查有效性

    Parameters:
        seq: 输入序列字符串
        seq_type: "DNA" 或 "RNA"
    Returns:
        清洗后的大写序列字符串
    Raises:
        ValueError: 包含无效碱基字符
    """
    seq = seq.strip().upper()
    # 去除方向标记 / Strip direction markers
    seq = re.sub(r"^[35]'[- ]?", "", seq)
    seq = re.sub(r"[- ]?[35]'$", "", seq)

    valid_bases = DNA_BASES if seq_type == "DNA" else RNA_BASES
    for i, base in enumerate(seq):
        if base not in valid_bases:
            raise ValueError(
                f"无效碱基 / Invalid base '{base}' at position {i + 1} "
                f"for {seq_type}. 有效碱基 / Valid bases: {', '.join(sorted(valid_bases))}"
            )
    if len(seq) == 0:
        raise ValueError("序列为空 / Empty sequence")
    return seq


# ===========================================================================
# Section 3: 完整 SMILES 构建器 / Full SMILES Builder (Core Algorithm)
# ===========================================================================

def build_full_smiles(seq, seq_type="DNA", direction="5to3"):
    """
    核心算法：逐核苷酸拼接完整原子级 SMILES。
    Core algorithm: build complete atom-level SMILES by nucleotide concatenation.

    拼接逻辑 / Concatenation logic:
      full_smiles = "O" (5'-OH)
      + internal[base] × (N-1)   (含磷酸二酯)
      + terminal[base] × 1       (3'-OH, 无磷酸)

    Parameters:
        seq: 已验证的序列字符串（大写）
        seq_type: "DNA" 或 "RNA"
        direction: "5to3" 或 "3to5"
    Returns:
        完整 SMILES 字符串
    """
    if direction == "3to5":
        seq = seq[::-1]

    if seq_type == "DNA":
        internal, terminal = DNA_INTERNAL, DNA_TERMINAL
    else:
        internal, terminal = RNA_INTERNAL, RNA_TERMINAL

    full_smiles = "O"  # 5'-OH
    for i, base in enumerate(seq):
        if i < len(seq) - 1:
            full_smiles += internal[base]
        else:
            full_smiles += terminal[base]

    return full_smiles


def build_fragment_smiles(seq, seq_type="DNA", start=0, length=3):
    """
    截取子序列构建片段 SMILES（用于长序列预览图）。
    Build SMILES for a subsequence fragment (for preview images of long sequences).

    Parameters:
        seq: 已验证的序列字符串
        seq_type: "DNA" 或 "RNA"
        start: 起始位置（0-indexed）
        length: 片段长度
    Returns:
        片段的完整 SMILES 字符串
    """
    fragment = seq[start:start + length]
    return build_full_smiles(fragment, seq_type, "5to3")


# ===========================================================================
# Section 4: BigSMILES 生成器 / BigSMILES Generator
# ===========================================================================

def generate_bigsmiles(seq_type="DNA"):
    """
    返回 DNA/RNA 的 BigSMILES 表示（聚合物类别，与序列无关）。
    Return BigSMILES representation for DNA/RNA polymer class (sequence-independent).

    复用本项目示例库 13.1（ssDNA）和 13.2（ssRNA）。

    Parameters:
        seq_type: "DNA" 或 "RNA"
    Returns:
        BigSMILES 字符串
    """
    return BIGSMILES_DNA if seq_type == "DNA" else BIGSMILES_RNA


# ===========================================================================
# Section 5: 图像生成器 / Image Generator
#
#   图1: 3D 结构文件 (.sdf)          — 真实空间构象，用 PyMOL/ChimeraX 打开
#   图2: 2D 全原子着色图 (RDKit)     — 按核苷酸着色的化学结构
#   两者均对任意长度序列生成，运行前给出预估时间。
# ===========================================================================

# 碱基颜色 RGB 0-1 / Base colors for RDKit highlighting
BASE_COLORS_RGB = {
    "A": (0.17, 0.63, 0.17), "T": (0.84, 0.15, 0.16),
    "U": (0.84, 0.15, 0.16), "G": (1.00, 0.50, 0.06),
    "C": (0.12, 0.47, 0.71),
}


def _estimate_time(n):
    """
    根据序列长度预估各步骤耗时（秒）。
    Estimate time in seconds for each step based on sequence length.
    经验公式由 4/12/45-nt 基准测试拟合。
    """
    t_3d = 0.008 * n ** 3          # ETKDG embed + UFF optimize
    t_2d = max(1.0, 0.05 * n ** 1.5)  # CoordGen + MolDraw2D
    return t_3d, t_2d


def _format_time(seconds):
    """将秒数格式化为人类可读字符串。"""
    if seconds < 60:
        return f"~{max(1, int(seconds))} 秒 / sec"
    elif seconds < 3600:
        return f"~{seconds / 60:.1f} 分钟 / min"
    else:
        return f"~{seconds / 3600:.1f} 小时 / hr"


def generate_images(smiles, seq, seq_type, output_dir=None):
    """
    生成 3D 结构 (.sdf) 和 2D 着色全原子图 (.png)，适用于任意长度。
    Generate 3D structure (.sdf) and 2D colored all-atom image (.png)
    for any sequence length. Prints estimated time before starting.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "output", "sequence_images")
    os.makedirs(output_dir, exist_ok=True)

    n = len(seq)
    prefix = _safe_prefix(seq, seq_type)

    # ---- 预估时间 / Time estimation ----
    t_3d, t_2d = _estimate_time(n)
    print(f"\n[INFO] 序列长度 / Sequence length: {n} nt")
    print(f"[INFO] 预估时间 / Estimated time:")
    print(f"         3D 结构 (.sdf) : {_format_time(t_3d)}")
    print(f"         2D 着色 (.png) : {_format_time(t_2d)}")
    print()

    generated = []

    # --- 图1: 3D 结构文件 / 3D structure (.sdf) ---
    path = _generate_3d_structure(smiles, seq, seq_type, output_dir, prefix)
    if path:
        generated.append(path)

    # --- 图2: 2D 着色全原子图 / 2D colored all-atom (.png) ---
    path = _generate_2d_colored(smiles, seq, seq_type, output_dir, prefix)
    if path:
        generated.append(path)

    return generated


# ---- 辅助函数 / Helpers ----

def _safe_prefix(seq, seq_type):
    """生成安全文件名前缀。"""
    if len(seq) <= 20:
        return f"{seq_type}_{seq}"
    return f"{seq_type}_{seq[:6]}...{seq[-3:]}_({len(seq)}nt)"


def _count_smiles_atoms(fragment):
    """
    统计 SMILES 片段中的重原子数（不含氢）。
    Count heavy atoms in a SMILES fragment string.
    """
    count = 0
    i = 0
    while i < len(fragment):
        c = fragment[i]
        if c == "[":
            count += 1
            i = fragment.index("]", i) + 1
        elif c == "C" and i + 1 < len(fragment) and fragment[i + 1] == "l":
            count += 1
            i += 2
        elif c == "B" and i + 1 < len(fragment) and fragment[i + 1] == "r":
            count += 1
            i += 2
        elif c in "BCNOPSFI":
            count += 1
            i += 1
        elif c in "bcnops":
            count += 1
            i += 1
        else:
            i += 1
    return count


# -----------------------------------------------------------------------
# 图1: 3D 空间结构 / 3D Structure (.sdf) — 任意长度
# -----------------------------------------------------------------------

def _generate_3d_structure(smiles, seq, seq_type, output_dir, prefix):
    """
    用 RDKit ETKDG 生成 3D 构象并导出 .sdf 文件（任意长度）。
    Generate 3D conformation via ETKDG and export as .sdf (any length).
    可用 PyMOL / ChimeraX / 3Dmol.js 打开查看空间结构。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("[WARN] RDKit 未安装，跳过 3D 生成 / "
              "RDKit not installed, skipping 3D generation")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # ETKDG 3D 嵌入 / 3D embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    n = len(seq)
    if n > 4:
        # 较长序列需要随机初始坐标 / longer sequences need random init
        params.useRandomCoords = True
    params.maxIterations = max(5000, n * 200)

    print(f"[...] 3D 嵌入中 / 3D embedding ...", end="", flush=True)
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        params.useRandomCoords = True
        params.maxIterations = max(10000, n * 500)
        result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        print(" 失败 / FAILED")
        return None
    print(" 完成 / done", flush=True)

    # 力场优化 / force-field optimization
    print(f"[...] 力场优化中 / Force-field optimizing ...", end="", flush=True)
    max_iters = max(2000, n * 100)
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
        except Exception:
            pass
    print(" 完成 / done", flush=True)

    mol = Chem.RemoveHs(mol)

    mol.SetProp("Sequence", seq)
    mol.SetProp("Type", seq_type)
    mol.SetProp("Length", str(n))

    filepath = os.path.join(output_dir, f"{prefix}_3d.sdf")
    mol_block = Chem.MolToMolBlock(mol)
    props = (f">  <Sequence>\n{seq}\n\n"
             f">  <Type>\n{seq_type}\n\n"
             f">  <Length>\n{n}\n\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(mol_block)
        f.write(props)
        f.write("$$$$\n")

    print(f"[OK] 3D 结构 / 3D structure: {os.path.basename(filepath)}  "
          f"(用 PyMOL/ChimeraX 打开 / open with PyMOL/ChimeraX)")
    return filepath


# -----------------------------------------------------------------------
# 图2: 2D 着色全原子图 / 2D Colored All-Atom Image — 任意长度
# -----------------------------------------------------------------------

def _generate_2d_colored(smiles, seq, seq_type, output_dir, prefix):
    """
    用 RDKit MolDraw2DCairo 生成按核苷酸着色的 2D 全原子结构图（任意长度）。
    Generate per-nucleotide colored 2D all-atom image via MolDraw2DCairo (any length).
    画布尺寸按原子数自动缩放。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        print("[WARN] RDKit 未安装，跳过 2D 着色图 / "
              "RDKit not installed, skipping 2D colored image")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 用 rdCoordGen 生成更优 2D 坐标 / better 2D layout via CoordGen
    print("[...] 2D 坐标生成中 / Generating 2D coords ...", end="", flush=True)
    _coord_method = "CoordGen"
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mol)
    except Exception:
        _coord_method = "Compute2DCoords"
        AllChem.Compute2DCoords(mol)
    print(" 完成 / done", flush=True)

    # ---- 计算每个核苷酸的原子索引范围 / atom index ranges per nucleotide ----
    if seq_type == "DNA":
        internal, terminal = DNA_INTERNAL, DNA_TERMINAL
    else:
        internal, terminal = RNA_INTERNAL, RNA_TERMINAL

    offset = 1  # 5'-OH 的 "O" 占 1 个原子
    nuc_ranges = []
    for i, base in enumerate(seq):
        frag = internal[base] if i < len(seq) - 1 else terminal[base]
        n_atoms = _count_smiles_atoms(frag)
        nuc_ranges.append((offset, offset + n_atoms, base))
        offset += n_atoms

    # ---- 构建高亮字典 / build highlight dicts ----
    highlight_atoms = []
    highlight_atom_colors = {}
    highlight_bonds = []
    highlight_bond_colors = {}

    for start, end, base in nuc_ranges:
        color = BASE_COLORS_RGB[base]
        atom_set = set(range(start, end))
        for idx in atom_set:
            highlight_atoms.append(idx)
            highlight_atom_colors[idx] = color
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in atom_set and a2 in atom_set:
                bi = bond.GetIdx()
                if bi not in highlight_bond_colors:
                    highlight_bonds.append(bi)
                    highlight_bond_colors[bi] = color

    # ---- 画布尺寸按原子数缩放 / canvas size scales with atom count ----
    n_atoms = mol.GetNumAtoms()
    w = max(900, min(5000, int(n_atoms ** 0.6 * 50)))
    h = int(w * 0.75)

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    except Exception:
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)

    opts = drawer.drawOptions()
    opts.bondLineWidth = max(1.0, 3.0 - len(seq) * 0.03)
    opts.highlightBondWidthMultiplier = 16
    opts.padding = 0.12

    def _do_draw(drawer_obj, mol_obj, hl_atoms, hl_acols, hl_bonds, hl_bcols):
        drawer_obj.DrawMolecule(
            mol_obj,
            highlightAtoms=hl_atoms,
            highlightAtomColors=hl_acols,
            highlightBonds=hl_bonds,
            highlightBondColors=hl_bcols,
        )
        drawer_obj.FinishDrawing()
        return drawer_obj.GetDrawingText()

    try:
        data = _do_draw(drawer, mol, highlight_atoms, highlight_atom_colors,
                        highlight_bonds, highlight_bond_colors)
    except RuntimeError:
        # rdCoordGen 可能产生零长度向量，回退到 Compute2DCoords
        # CoordGen may produce zero-length vectors; fall back to Compute2DCoords
        if _coord_method == "CoordGen":
            print("[WARN] CoordGen 布局失败，回退 Compute2DCoords / "
                  "CoordGen layout failed, falling back to Compute2DCoords",
                  flush=True)
            AllChem.Compute2DCoords(mol)
            try:
                drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
            except Exception:
                drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
            opts = drawer.drawOptions()
            opts.bondLineWidth = max(1.0, 3.0 - len(seq) * 0.03)
            opts.highlightBondWidthMultiplier = 16
            opts.padding = 0.12
            data = _do_draw(drawer, mol, highlight_atoms, highlight_atom_colors,
                            highlight_bonds, highlight_bond_colors)
        else:
            raise

    if isinstance(data, str):
        filepath = os.path.join(output_dir, f"{prefix}_2d_colored.svg")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        filepath = os.path.join(output_dir, f"{prefix}_2d_colored.png")
        with open(filepath, "wb") as f:
            f.write(data)

    print(f"[OK] 2D 着色图 / 2D colored: {os.path.basename(filepath)}  "
          f"(A=绿/green  T/U=红/red  G=橙/orange  C=蓝/blue)")
    return filepath


# ===========================================================================
# Section 6: 主编排函数 / Main Orchestration Function
# ===========================================================================

def sequence_to_representations(seq, seq_type="DNA", direction="5to3",
                                generate_imgs=True, output_dir=None):
    """
    主编排函数：将核酸序列转换为 BigSMILES + Full SMILES 双表示。
    Main orchestration: convert nucleic acid sequence to dual representations.

    Parameters:
        seq: 核酸序列字符串
        seq_type: "DNA" 或 "RNA"
        direction: "5to3" 或 "3to5"
        generate_imgs: 是否生成结构图
        output_dir: 图片输出目录
    Returns:
        包含所有结果的字典
    """
    # 验证序列 / Validate sequence
    clean_seq = validate_sequence(seq, seq_type)

    # 构建表示 / Build representations
    full_smiles = build_full_smiles(clean_seq, seq_type, direction)
    bigsmiles = generate_bigsmiles(seq_type)

    result = {
        "input_sequence": seq,
        "clean_sequence": clean_seq,
        "sequence_type": seq_type,
        "direction": direction,
        "length": len(clean_seq),
        "bigsmiles": bigsmiles,
        "full_smiles": full_smiles,
        "images": [],
    }

    # RDKit 验证 / RDKit validation
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(full_smiles)
        result["rdkit_valid"] = mol is not None
        if mol:
            result["atom_count"] = mol.GetNumAtoms()
    except ImportError:
        result["rdkit_valid"] = None

    # BigSMILES 检查器验证 / BigSMILES checker validation
    try:
        from bigsmiles_checker import check_bigsmiles
        result["bigsmiles_valid"] = check_bigsmiles(bigsmiles, verbose=False)
    except ImportError:
        result["bigsmiles_valid"] = None

    # 生成图片 / Generate images
    if generate_imgs:
        result["images"] = generate_images(full_smiles, clean_seq, seq_type, output_dir)

    # 自动保存 JSON 记录 / Auto-save JSON record
    _save_record(result)

    return result


def _save_record(result):
    """
    将转换结果自动保存为 JSON 文件到 output/sequence_records/。
    Auto-save conversion result as JSON to output/sequence_records/.
    """
    records_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "sequence_records")
    os.makedirs(records_dir, exist_ok=True)

    seq = result["clean_sequence"]
    seq_type = result["sequence_type"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 文件名：类型_序列(截断)_时间戳.json
    if len(seq) <= 20:
        seq_tag = seq
    else:
        seq_tag = f"{seq[:6]}...{seq[-3:]}_({result['length']}nt)"
    filename = f"{seq_type}_{seq_tag}_{timestamp}.json"

    filepath = os.path.join(records_dir, filename)
    record = {
        "timestamp": datetime.now().isoformat(),
        "sequence": result["clean_sequence"],
        "sequence_type": result["sequence_type"],
        "direction": result["direction"],
        "length": result["length"],
        "bigsmiles": result["bigsmiles"],
        "full_smiles": result["full_smiles"],
        "rdkit_valid": result.get("rdkit_valid"),
        "atom_count": result.get("atom_count"),
        "bigsmiles_valid": result.get("bigsmiles_valid"),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[OK] 记录已保存 / Record saved: {filepath}")


# ===========================================================================
# Section 7: CLI 接口 / CLI Interface
# ===========================================================================

def main():
    """命令行入口。/ CLI entry point."""
    parser = argparse.ArgumentParser(
        description="核酸序列 → BigSMILES / Full SMILES 转换器 | "
                    "Nucleic Acid Sequence → BigSMILES / Full SMILES Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例 / Examples:\n"
            "  python sequence_to_bigsmiles.py ACGT\n"
            "  python sequence_to_bigsmiles.py ACGT --type DNA --direction 5to3\n"
            "  python sequence_to_bigsmiles.py AUGC --type RNA\n"
            "  python sequence_to_bigsmiles.py ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT\n"
            "  python sequence_to_bigsmiles.py ACGT --no-images --json\n"
        ),
    )
    parser.add_argument("sequence", help="核酸序列 / Nucleic acid sequence (e.g., ACGT)")
    parser.add_argument("--type", choices=["DNA", "RNA"], default="DNA",
                        help="序列类型 / Sequence type (default: DNA)")
    parser.add_argument("--direction", choices=["5to3", "3to5"], default="5to3",
                        help="链方向 / Chain direction (default: 5to3)")
    parser.add_argument("--no-images", action="store_true",
                        help="不生成结构图 / Skip image generation")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出 / Output in JSON format")
    parser.add_argument("--output-dir", default=None,
                        help="图片输出目录 / Image output directory")

    args = parser.parse_args()

    try:
        result = sequence_to_representations(
            args.sequence, args.type, args.direction,
            generate_imgs=not args.no_images, output_dir=args.output_dir,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_result(result)


def _print_result(result):
    """格式化终端输出。/ Formatted terminal output."""
    print(f"\n{'=' * 70}")
    print(f"  核酸序列 → BigSMILES / Full SMILES 转换结果")
    print(f"  Nucleic Acid Sequence → BigSMILES / Full SMILES Result")
    print(f"{'=' * 70}")
    print(f"  输入序列 / Input:     {result['input_sequence']}")
    print(f"  类型 / Type:          {result['sequence_type']}")
    print(f"  方向 / Direction:     {result['direction']}")
    print(f"  长度 / Length:        {result['length']} nt")

    print(f"\n  BigSMILES (聚合物类别 / polymer class):")
    # 分行显示 BigSMILES / Display BigSMILES with line breaks
    bs = result["bigsmiles"]
    if len(bs) > 80:
        print(f"    {bs[:80]}")
        remaining = bs[80:]
        while remaining:
            print(f"    {remaining[:80]}")
            remaining = remaining[80:]
    else:
        print(f"    {bs}")

    print(f"\n  Full SMILES (特定序列 / specific sequence):")
    smiles = result["full_smiles"]
    if len(smiles) > 80:
        print(f"    {smiles[:80]}...")
        print(f"    (共 / total {len(smiles)} 字符 / characters)")
    else:
        print(f"    {smiles}")

    if result.get("rdkit_valid") is not None:
        status = "PASS" if result["rdkit_valid"] else "FAIL"
        print(f"\n  RDKit 验证 / validation: {status}")
    if result.get("atom_count"):
        print(f"  原子数 / Atom count:     {result['atom_count']}")
    if result.get("bigsmiles_valid") is not None:
        status = "PASS" if result["bigsmiles_valid"] else "FAIL"
        print(f"  BigSMILES 检查 / check:  {status}")

    if result["images"]:
        print(f"\n  生成图片 / Generated images:")
        for img in result["images"]:
            print(f"    - {img}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
