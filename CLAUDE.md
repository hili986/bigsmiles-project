# BigSMILES Project

Core codebase for BigSMILES tools: example library, syntax checker, parser/generator, Bicerano Tg dataset, structural fingerprints, and nucleic acid converter.

## Commands

- Run all tests: `python -m unittest discover -v`
- Run specific test file: `python -m unittest test_bigsmiles -v`
- Generate example images: `python bigsmiles_examples.py` (requires RDKit)
- Convert sequence: `python sequence_to_bigsmiles.py --sequence ACGT --type DNA`
- Fingerprint demo: `python bigsmiles_fingerprint.py --demo`
- Dataset export: `python bicerano_tg_dataset.py --csv --json --validate`

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
