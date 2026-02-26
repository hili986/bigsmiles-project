# BigSMILES 编码复现、示例库、语法检查器与核酸序列转换工具

SITP 项目"人工智能辅助高分子材料设计"交付物，基于 Lin et al. 2019 论文复现。

## 项目结构

```
bigsmiles_project/
├── bigsmiles_examples.py        # 示例库（39 个聚合物 BigSMILES 编码）
├── bigsmiles_checker.py         # 语法检查器（分词 → 解析 → 校验）
├── sequence_to_bigsmiles.py     # 核酸序列 → BigSMILES / Full SMILES 转换工具
├── test_bigsmiles.py            # 示例库 & 检查器测试套件（65 个用例）
├── test_sequence_to_bigsmiles.py # 序列转换工具测试套件（31 个用例）
├── 方法论.md                     # 核酸序列 BigSMILES 表示方法论文档
├── README.md
└── output/
    ├── bigsmiles_examples.json  # 示例库 JSON 导出
    ├── images/                  # RDKit 生成的重复单元结构图（PNG）
    ├── sequence_images/         # 核酸序列 3D 结构(.sdf) + 2D 着色图(.png)
    └── sequence_records/        # 每次转换自动保存的 JSON 记录
```

## 环境要求

- Python >= 3.10
- RDKit（用于 SMILES 校验和结构图生成，非必需——缺失时自动跳过相关功能）

## 快速开始

```bash
cd bigsmiles_project

# 1. 运行全部测试（验证全部功能）
python -m unittest test_bigsmiles -v
python -m unittest test_sequence_to_bigsmiles -v

# 2. 终端查看全部 39 个示例
python bigsmiles_examples.py

# 3. 导出 JSON / 生成结构图
python bigsmiles_examples.py --json
python bigsmiles_examples.py --images

# 4. 检查任意 BigSMILES 字符串
python bigsmiles_checker.py "{[$]CC[$]}"
python bigsmiles_checker.py "{[>]CCO[<]}"

# 5. 核酸序列转换
python sequence_to_bigsmiles.py ACGT
python sequence_to_bigsmiles.py AUGC --type RNA
python sequence_to_bigsmiles.py ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT
python sequence_to_bigsmiles.py ACGT --no-images --json
```

## 功能说明

### 1. 示例库 (`bigsmiles_examples.py`)

收录 **39 个聚合物**，覆盖 **13 类结构**：

| # | 类别 | 数量 | 代表性示例 |
|---|------|------|-----------|
| 1 | 线型均聚物 | 8 | PE, PEG, PS, PP, PVC, PMMA, PTFE, PAN |
| 2 | 聚异戊二烯立体异构 | 4 | cis-1,4 / trans-1,4 / 3,4-加成 / 混合 |
| 3 | 无规共聚物 | 3 | PE-co-PB, PS-co-PMMA, SBR |
| 4 | 缩聚物/交替共聚物 | 4 | Nylon-6,6, PET, BPA-PC, PLA |
| 5 | 嵌段共聚物 | 3 | PS-b-PMMA, Pluronic, SBS |
| 6 | 接枝共聚物 | 2 | PIB-g-PMMA, PS-g-PEG |
| 7 | 支化聚合物 | 2 | LDPE, 超支化聚酯 |
| 8 | 网状/交联 | 2 | 硫化橡胶, 环氧-胺网络 |
| 9 | 环形聚合物 | 2 | 环形 PS, 环形 PEG |
| 10 | 端基指定 | 2 | AIBN-PS 确定性/随机端基 |
| 11 | 嵌套随机对象 | 1 | 聚氨酯 + PEG 软段 |
| 12 | 其他重要聚合物 | 4 | Nylon-6, PCL, PAI, PDMS |
| 13 | 核酸聚合物 | 2 | ssDNA (PD-L1 适配体), ssRNA |

每个条目包含：

```python
{
    "id": "1.1",
    "category_cn": "线型均聚物",       "category_en": "Linear Homopolymer",
    "name_cn": "聚乙烯",              "name_en": "Polyethylene (PE)",
    "mechanism_cn": "自由基/配位聚合",
    "bonding_type": "AA",              # AA($) 或 AB(<>)
    "bigsmiles": "{[$]CC[$]}",
    "smiles_repeat_unit": "*CC*",      # RDKit 可解析的重复单元 SMILES
    "structure_ascii": "—[CH2-CH2]n—",
    "explanation_cn": "...",            "explanation_en": "...",
    "source": "Lin et al. 2019, Fig. 2",
}
```

**API 函数：**

| 函数 | 说明 |
|------|------|
| `get_examples()` | 返回全部 39 个示例的列表 |
| `to_json(filepath)` | 导出为 JSON 文件 |
| `generate_images(output_dir)` | 用 RDKit 为每个重复单元生成 PNG 结构图 |
| `print_library()` | 终端格式化输出全部示例 |

### 2. 语法检查器 (`bigsmiles_checker.py`)

三阶段流水线架构：

```
输入字符串 → [分词器 Tokenizer] → Token 流 → [递归下降解析器 Parser] → AST → [语义校验器 Validator] → 结果
```

**分词器**识别的 Token 类型：

| Token | 含义 | 示例 |
|-------|------|------|
| `STOCH_OPEN/CLOSE` | 随机对象边界 | `{` `}` |
| `BOND_DESC` | 键连接描述符 | `[$]` `[>1]` `[<]` |
| `TERM_DESC_EMPTY` | 空终端描述符 | `[]` |
| `ATOM_ORGANIC` | 有机原子 | `C` `N` `Cl` `c` |
| `ATOM_BRACKET` | 方括号原子 | `[NH]` `[Si]` `[C@@H]` |
| `BOND` | 键符号 | `-` `=` `#` `/` `\` |
| `REPEAT_SEP` | 重复单元分隔（`{}` 内） | `,` |
| `ENDGROUP_SEP` | 端基分隔（`{}` 内） | `;` |
| `BRANCH_OPEN/CLOSE` | 分支括号 | `(` `)` |
| `RING_DIGIT` | 环闭合数字 | `1` `%10` |
| `DOT` | 断开结构分隔符 | `.` |

**语义校验器**执行 7 项检查：

| # | 检查项 | 说明 |
|---|--------|------|
| 1 | 括号匹配 | `{}` `[]` `()` 正确嵌套 |
| 2 | 描述符语法 | `$`/`<`/`>` + 可选键型 + 可选数字 ID |
| 3 | 随机对象结构 | 至少 1 个非空重复单元 |
| 4 | 终端描述符配对 | `[>n]` 与 `[<n]` 的 ID 必须匹配 |
| 5 | SMILES 有效性 | 描述符替换为 `*` 后由 RDKit 校验 |
| 6 | 描述符一致性 | 同一 `{}` 内不混用 `$` 和 `<>` |
| 7 | 最少描述符数 | 重复单元 >= 2 个，端基 = 1 个 |

**公共 API：**

```python
from bigsmiles_checker import check_bigsmiles

check_bigsmiles("{[$]CC[$]}")          # True，打印 [OK]
check_bigsmiles("{[$]CC[>]}")          # False，打印中英文错误信息
check_bigsmiles("{[$]CC[$]}", verbose=False)  # True，静默模式
```

**错误报告格式：**

```
  Input: {[$]CC[>]}

  [ERROR] (位置/position 0)
    EN: Mixed AA-type ($) and AB-type (<>) descriptors in same stochastic object at position 0
    CN: 位置 0 的随机对象中混用了 AA 型($)和 AB 型(<>)描述符
```

### 3. 核酸序列转换工具 (`sequence_to_bigsmiles.py`)

将核酸序列（DNA/RNA）转换为**双表示**：

- **BigSMILES** — 描述聚合物类别（与序列无关，复用示例库 13.1/13.2）
- **Full SMILES** — 逐核苷酸拼接完整原子级 SMILES，编码精确碱基序列

**核心问题**：BigSMILES 是为随机聚合物设计的，无法编码特定碱基序列。Full SMILES 拼接法是唯一能同时编码序列并生成结构图的可行方案。详见 [`方法论.md`](方法论.md)。

**支持的核酸类型：**

| 类型 | 碱基 | 糖基 |
|------|------|------|
| DNA | A, T, G, C | 脱氧核糖（无 2'-OH） |
| RNA | A, U, G, C | 核糖（含 2'-OH） |

**CLI 用法：**

```bash
# 基本用法（默认 DNA, 5'→3'）
python sequence_to_bigsmiles.py ACGT

# 指定类型和方向
python sequence_to_bigsmiles.py ACGT --type DNA --direction 5to3
python sequence_to_bigsmiles.py AUGC --type RNA

# PD-L1 适配体（45 nt）— 端到端验证
python sequence_to_bigsmiles.py ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT

# JSON 输出 / 不生成图片
python sequence_to_bigsmiles.py ACGT --no-images --json

# 指定输出目录
python sequence_to_bigsmiles.py ACGT --output-dir ./my_output
```

**输出示例（ACGT）：**

```
[INFO] 序列长度 / Sequence length: 4 nt
[INFO] 预估时间 / Estimated time:
         3D 结构 (.sdf) : ~1 秒 / sec
         2D 着色 (.png) : ~1 秒 / sec

  输入序列 / Input:     ACGT
  类型 / Type:          DNA
  方向 / Direction:     5to3
  长度 / Length:        4 nt

  BigSMILES (polymer class):
    O{[>]CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O[<],...}

  Full SMILES (specific sequence):
    OCC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)OCC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O...

  RDKit 验证 / validation: PASS
  原子数 / Atom count:     79
  BigSMILES 检查 / check:  PASS

[OK] 3D 结构: DNA_ACGT_3d.sdf
[OK] 2D 着色图: DNA_ACGT_2d_colored.png
[OK] 记录已保存: output/sequence_records/DNA_ACGT_20260222_HHMMSS.json
```

**API 函数：**

| 函数 | 说明 |
|------|------|
| `validate_sequence(seq, type)` | 验证序列字符，去除方向标记 |
| `build_full_smiles(seq, type, dir)` | 核心：逐核苷酸拼接完整 SMILES |
| `build_fragment_smiles(seq, type, start, length)` | 截取子序列构建片段 SMILES |
| `generate_bigsmiles(type)` | 返回 DNA/RNA 的 BigSMILES |
| `generate_images(smiles, seq, type, output_dir)` | RDKit 生成 PNG 结构图 |
| `sequence_to_representations(...)` | 主编排函数，返回完整结果 dict |

**图像生成策略（任意序列长度均适用）：**

| 输出 | 格式 | 说明 |
|------|------|------|
| 3D 结构 | `.sdf` | ETKDG 嵌入 + UFF/MMFF 力场优化，用 PyMOL/ChimeraX 打开 |
| 2D 着色图 | `.png` | 按碱基着色（A=绿 T/U=红 G=橙 C=蓝），画布自动缩放 |

运行开始时会显示预估耗时（3D 嵌入对长序列较慢，如 45-nt ≈ 12 分钟）。

**自动记录：** 每次转换自动在 `output/sequence_records/` 保存一份 JSON，包含序列、BigSMILES、Full SMILES、RDKit 验证结果和原子数。

### 4. 测试套件

共 **96 个测试用例**，分布在两个测试文件中：

**`test_bigsmiles.py`（65 个用例）：**

| 类别 | 数量 | 说明 |
|------|------|------|
| 正向测试 (`TestPositive`) | 40 | 39 个示例全部通过 + 条目数量校验 |
| 负向测试 (`TestNegative`) | 10 | 括号不匹配、描述符混用、SMILES 无效等 |
| 边界测试 (`TestBoundary`) | 6 | 嵌套随机对象、立体化学、端基、嵌段共聚物 |
| 分词器测试 (`TestTokenizer`) | 5 | Token 类型识别、描述符格式 |
| 示例库测试 (`TestExamplesLibrary`) | 4 | 字段完整性、类别覆盖、JSON 导出 |

**`test_sequence_to_bigsmiles.py`（31 个用例）：**

| 类别 | 数量 | 说明 |
|------|------|------|
| 序列验证 (`TestSequenceValidator`) | 10 | 有效/无效序列、大小写、方向标记、空序列 |
| SMILES 构建 (`TestFullSMILESBuilder`) | 12 | 单/二/三/四核苷酸、45-nt 适配体、RDKit 验证、方向反转、DNA vs RNA、片段 |
| BigSMILES 生成 (`TestBigSMILESGenerator`) | 5 | 匹配 13.1/13.2 示例、通过检查器、序列无关性 |
| 图像生成 (`TestImageGeneration`) | 4 | 短/中/长序列图像生成、端到端编排 |

### 5. 方法论文档 (`方法论.md`)

详细阐述核酸序列 BigSMILES 表示的方法选择依据：

1. **问题陈述** — BigSMILES 的随机性本质 vs 序列特异性需求
2. **文献综述** — BigSMILES / HELM / Full SMILES / PSMILES / G-BigSMILES 五种方案
3. **方法对比表** — 序列特异性、紧凑度、工具支持、图像生成能力
4. **我们的策略** — 双表示混合法（BigSMILES + Full SMILES）
5. **核苷酸构建模块** — 8 种碱基的 SMILES 片段（DNA: dA/dT/dG/dC, RNA: rA/rU/rG/rC）
6. **磷酸二酯键拼接机制** — 5'-OH + [核苷酸-磷酸]×(N-1) + [核苷酸-3'-OH]
7. **验证与讨论** — RDKit 验证结果、长度极限、局限性
8. **参考文献**

## BigSMILES 简介

BigSMILES 是 SMILES 的扩展表示法，用于描述聚合物的随机（stochastic）结构。核心概念：

- **随机对象** `{...}`：表示由重复单元随机组成的聚合物链段
- **键连接描述符**：标记重复单元之间的连接点
  - **AA 型** `[$]`：两端等价（如聚乙烯的 C-C 键）
  - **AB 型** `[>]` `[<]`：两端不等价（如 PEG 的 C-O 键 vs C-C 键）
- **逗号** `,`：分隔多种重复单元（表示随机排列）
- **分号** `;`：分隔重复单元与端基
- **嵌套** `{...{...}...}`：多级聚合物结构（如接枝共聚物）
- **相邻随机对象** `{...}{...}`：嵌段共聚物

**示例：**

```
{[$]CC[$]}                          聚乙烯 (PE)
{[>]CCO[<]}                         聚乙二醇 (PEG)
{[$]CC[$],[$]CC(CC)[$]}             聚(乙烯-co-丁烯) 无规共聚物
{[$]CC(c1ccccc1)[$]}{[>]CCO[<]}    PS-b-PEG 嵌段共聚物
{[$]CC(c1ccccc1)[$];[$]Cl}          含随机端基的聚苯乙烯
O{[>]...[<];[>]...O}               核酸聚合物（ssDNA/ssRNA）
```

## 参考文献

- Lin, T. S.; Coley, C. W.; Mochigase, H.; Beesam, H. K.; Bilodeau, C.; et al. *BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules.* ACS Cent. Sci. 2019, 5, 1523-1531.
- Zhang, Z.; Interrante, L. M.; Greer, S. C.; Lin, T.-S. *G-BigSMILES: A Graph-Based Extension of BigSMILES.* Digital Discovery 2024.
- Pistoia Alliance. *HELM Notation.* https://pistoiaalliance.atlassian.net/wiki/spaces/PUB/pages/6619143/HELM+Notation
- RDKit: Open-Source Cheminformatics. https://www.rdkit.org/
