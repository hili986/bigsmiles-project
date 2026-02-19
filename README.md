# BigSMILES 编码复现、示例库与语法检查器

SITP 项目"人工智能辅助高分子材料设计"交付物，基于 Lin et al. 2019 论文复现。

## 项目结构

```
bigsmiles_project/
├── bigsmiles_examples.py        # 示例库（37 个聚合物 BigSMILES 编码）
├── bigsmiles_checker.py         # 语法检查器（分词 → 解析 → 校验）
├── test_bigsmiles.py            # 测试套件（63 个用例）
├── README.md
└── output/
    ├── bigsmiles_examples.json  # 示例库 JSON 导出
    └── images/                  # RDKit 生成的重复单元结构图（PNG）
```

## 环境要求

- Python >= 3.10
- RDKit（用于 SMILES 校验和结构图生成，非必需——缺失时自动跳过相关功能）

## 快速开始

```bash
cd bigsmiles_project

# 1. 运行测试（验证全部功能）
python -m unittest test_bigsmiles -v

# 2. 终端查看全部 37 个示例
python bigsmiles_examples.py

# 3. 导出 JSON / 生成结构图
python bigsmiles_examples.py --json
python bigsmiles_examples.py --images

# 4. 检查任意 BigSMILES 字符串
python bigsmiles_checker.py "{[$]CC[$]}"
python bigsmiles_checker.py "{[>]CCO[<]}"
```

## 功能说明

### 1. 示例库 (`bigsmiles_examples.py`)

收录 **37 个聚合物**，覆盖 **12 类结构**：

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
| `get_examples()` | 返回全部 37 个示例的列表 |
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

### 3. 测试套件 (`test_bigsmiles.py`)

共 **63 个测试用例**：

| 类别 | 数量 | 说明 |
|------|------|------|
| 正向测试 (`TestPositive`) | 38 | 37 个示例全部通过 + 条目数量校验 |
| 负向测试 (`TestNegative`) | 10 | 括号不匹配、描述符混用、SMILES 无效等 |
| 边界测试 (`TestBoundary`) | 6 | 嵌套随机对象、立体化学、端基、嵌段共聚物 |
| 分词器测试 (`TestTokenizer`) | 5 | Token 类型识别、描述符格式 |
| 示例库测试 (`TestExamplesLibrary`) | 4 | 字段完整性、类别覆盖、JSON 导出 |

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
```

## 参考文献

- Lin, T. S.; Coley, C. W.; Mochigase, H.; Beesam, H. K.; Bilodeau, C.; et al. *BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules.* ACS Cent. Sci. 2019, 5, 1523-1531.
