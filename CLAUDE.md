# BigSMILES Project

Core codebase for BigSMILES tools: example library, syntax checker, parser/generator, Bicerano Tg dataset, structural fingerprints, nucleic acid converter, ML models, experiment framework, property annotation, and web demo.

## Commands

- Run all tests: `python -m unittest discover -v`
- Run specific test file: `python -m unittest test_bigsmiles -v`
- Generate example images: `python bigsmiles_examples.py` (requires RDKit)
- Convert sequence: `python sequence_to_bigsmiles.py --sequence ACGT --type DNA`
- Fingerprint demo: `python bigsmiles_fingerprint.py --demo`
- Dataset export: `python bicerano_tg_dataset.py --csv --json --validate`
- ML experiment: `python ml_experiment.py --all` (model comparison + ablation + sweep)
- Annotation demo: `python bigsmiles_annotation.py --demo`
- Web demo: `python web_demo.py --port 8765` (opens at http://127.0.0.1:8765)

## Architecture

**bigsmiles_checker.py** — Three-stage pipeline:
1. **Tokenizer**: String → Token stream (11 token types, context-aware `{}` nesting)
2. **Parser**: Recursive descent → AST (StochasticObject, RepeatUnit, Atom, Bond nodes)
3. **Validator**: 7 semantic checks including RDKit SMILES validation

**bigsmiles_examples.py** — 39 polymers across 13 categories. Each entry: BigSMILES + repeat unit SMILES + bilingual explanation + ASCII structure + literature source.

**bigsmiles_parser.py** — Parser & Generator API wrapping the checker's Tokenizer/Parser:
- `BigSMILESParser.parse(string) → AST`, `generate(ast) → string`, `round_trip(string) → string`
- Extraction: `get_repeat_units()`, `get_bonding_descriptors()`, `get_topology()`
- Topology detection: small_molecule, linear_homopolymer, random_copolymer, block_copolymer, graft_copolymer, branched

**bicerano_tg_dataset.py** — 304 linear homopolymers with Tg(K) from Bicerano/Choi (2018):
- Hardcoded dataset: (name, repeat_unit_SMILES, BigSMILES, Tg_K) tuples
- API: `load_dataset()`, `get_names()`, `get_smiles()`, `to_csv()`, `to_json()`, `validate_all()`
- `shorthand_to_bracket()` converter: `{$CC$}` → `{[$]CC[$]}`

**bigsmiles_fingerprint.py** — Structural fingerprints bridging BigSMILES to ML:
- Morgan/ECFP fingerprints via RDKit (bit vector + count vector)
- 15 functional group SMARTS patterns (fragment counting)
- 14 polymer-level descriptors (MW, heteroatom ratios, aromatic fraction, logP, TPSA, bonding type)
- Combined feature vector + Tg ridge regression demo (no numpy/sklearn dependency)

**sequence_to_bigsmiles.py** — Dual-representation: BigSMILES (polymer class) + Full SMILES (exact nucleotide sequence). 16 nucleotide SMILES fragments (4 bases × 2 forms × DNA/RNA).

**ml_models.py** — Pure Python ML regression models (no numpy/sklearn):
- 7 models: Ridge, Lasso, ElasticNet, KNN, DecisionTree, RandomForest, GradientBoosting
- Unified `fit(X, y)` / `predict(X)` / `fit_predict()` interface
- Utilities: `normalize()`, `train_test_split()`, `k_fold_indices()`, `cross_validate()`
- Metrics: `r2_score()`, `mae_score()`, `rmse_score()`, `mape_score()`

**ml_experiment.py** — Tg prediction experiment framework:
- `build_dataset()`: Bicerano data → feature matrix (Morgan + fragments + descriptors)
- `run_model_comparison()`: 7-model benchmark via k-fold CV
- `run_feature_ablation()`: compare Morgan-only, fragments-only, etc.
- `run_morgan_sweep()`: sweep Morgan radius × bits

**bigsmiles_annotation.py** — BigSMILES property annotation extension:
- Syntax: `{[$]CC[$]}|Tg=373K;Mn=50000|`
- 15 known polymer properties with aliases, units, types
- API: `parse_annotation()`, `add_annotation()`, `validate_annotation()`

**web_demo.py** — End-to-end web demo (stdlib http.server):
- Pipeline: BigSMILES input → syntax check → parse → fingerprint → Tg prediction
- JSON API: `/api/check`, `/api/parse`, `/api/fingerprint`, `/api/predict`, `/api/pipeline`
- Single-page HTML frontend with live analysis

## Code Patterns

- Uses `@dataclass` for Token, ASTNode types
- Public API: `check_bigsmiles(string) -> bool` with printed diagnostics
- Regex for bond descriptors: `[=#/\\]?[\$<>]\d*`
- RDKit imports wrapped in try/except for graceful degradation
- All user-facing messages are bilingual (English + Chinese)

## Gotchas

- Ring closure numbering in nucleotide fragments uses local closures (1,2,3) per fragment — no inter-nucleotide conflicts
- `bigsmiles_examples.py` is ~1200 lines — close to the 800-line guideline; consider splitting if adding more examples
- The `output/` directory is gitignored — generated artifacts only
