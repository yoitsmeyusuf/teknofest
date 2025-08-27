#!/usr/bin/env python3
"""Convert JSONL dataset (tokens+labels) into token-level CONLL and JSONL splits.

Writes train.conll / dev.conll / test.conll and train.jsonl / dev.jsonl / test.jsonl
into a target output directory.

Usage:
  python3 scripts/convert_to_conll.py --input /path/to/file.jsonl --outdir ./egitimverisi/converted --seed 42
If --input is a directory, all .jsonl files inside will be used.
"""
import argparse
import json
import random
from pathlib import Path


def read_jsonl_paths(p: Path):
    if p.is_dir():
        return sorted([x for x in p.glob('*.jsonl')])
    return [p]


def load_records(paths):
    records = []
    for p in paths:
        with p.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Skipping malformed line {i} in {p}: {e}")
                    continue
                tokens = obj.get('tokens')
                labels = obj.get('labels')
                if tokens is None or labels is None:
                    # try alternative keys
                    continue
                if len(tokens) != len(labels):
                    print(f"Skipping record in {p} line {i}: token/label length mismatch ({len(tokens)} vs {len(labels)})")
                    continue
                records.append({'tokens': tokens, 'labels': labels})
    return records


def write_conll(records, path: Path):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            for tok, lab in zip(rec['tokens'], rec['labels']):
                f.write(f"{tok}\t{lab}\n")
            f.write('\n')


def write_jsonl(records, path: Path):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')


def split_records(records, seed=42):
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n = len(records)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    train_idx = idxs[:n_train]
    dev_idx = idxs[n_train:n_train + n_dev]
    test_idx = idxs[n_train + n_dev:]
    train = [records[i] for i in train_idx]
    dev = [records[i] for i in dev_idx]
    test = [records[i] for i in test_idx]
    return train, dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True, help='input .jsonl file or directory')
    ap.add_argument('--outdir', '-o', default='egitimverisi/converted', help='output directory')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = read_jsonl_paths(inp)
    if not paths:
        print('No input files found')
        return

    records = load_records(paths)
    print(f'Loaded {len(records)} valid records from {len(paths)} file(s)')
    if not records:
        print('No usable records. Exiting.')
        return

    train, dev, test = split_records(records, seed=args.seed)
    print(f'Split: train={len(train)}, dev={len(dev)}, test={len(test)}')

    # write conll
    write_conll(train, outdir / 'train.conll')
    write_conll(dev, outdir / 'dev.conll')
    write_conll(test, outdir / 'test.conll')

    # write JSONL alternative
    write_jsonl(train, outdir / 'train.jsonl')
    write_jsonl(dev, outdir / 'dev.jsonl')
    write_jsonl(test, outdir / 'test.jsonl')

    print('Wrote files to', outdir)


if __name__ == '__main__':
    main()
