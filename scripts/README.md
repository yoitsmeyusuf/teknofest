Convert JSONL (tokens+labels) to CONLL for BERT NER

This small script converts a JSONL dataset where each line is a JSON object
containing at least `tokens` (list) and `labels` (list) into token-level
CONLL files and corresponding JSONL splits.

Usage examples:

1) Convert a single file:
   python3 scripts/convert_to_conll.py -i egitimverisi/21_22_23_filtered.jsonl -o egitimverisi/converted

2) Convert all JSONL files in a directory:
   python3 scripts/convert_to_conll.py -i egitimverisi -o egitimverisi/converted

Output:
- egitimverisi/converted/train.conll, dev.conll, test.conll
- egitimverisi/converted/train.jsonl, dev.jsonl, test.jsonl

Split: random with seed (default 42), proportions: 80% train, 10% dev, 10% test.

Notes:
- The script verifies token/label length equality and skips malformed lines.
- If you need deterministic per-sentence balancing for labels, consider
  stratified sampling (not implemented here).
