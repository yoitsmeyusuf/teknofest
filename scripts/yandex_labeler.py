#!/usr/bin/env python3
"""
Yandex Geocoder -> BIO tagger for Turkish addresses

Usage (example):
  export YA_API_KEY=your_key_here
  python3 scripts/yandex_labeler.py --input data/raw_addresses.csv --out labeled.jsonl --conll conll.txt

Implements: normalization, caching, rate-limiting, retries, component extraction,
fuzzy alignment with rapidfuzz, optional bbox, and outputs: labeled.jsonl, conll.txt,
and review_needed.jsonl for low-confidence cases.
"""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm.auto import tqdm


DEFAULT_CACHE_DIR = ".yandex_cache"
GEOCODE_URL = "https://geocode-maps.yandex.ru/1.x/"


TR_MAP = str.maketrans("ÇĞİÖŞÜÂÎÛçğıöşüâîû", "CGIOSUAIUcgiosuaiu")


def normalize(s: Optional[str]) -> str:
    s = (s or "")
    # Normalize unicode, lowercase
    s = unicodedata.normalize("NFKD", s).lower()
    # Replace Turkish chars to ASCII-ish equivalents (use shared map)
    s = s.translate(TR_MAP)
    # Replace separators with space
    s = s.replace("/", " ").replace("\\", " ").replace("_", " ").replace("-", " ")
    # Remove extra punctuation except alnum and spaces
    s = re.sub(r"[^\w\sçğıöşüâîûÇĞİÖŞÜÂÎÛ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cache_path(cache_dir: str, q: str) -> str:
    key = hashlib.md5(q.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def yandex_geocode(addr: str, api_key: str, cache_dir: str = DEFAULT_CACHE_DIR, bbox: Optional[str] = None,
                   max_retries: int = 3, qps_delay: float = 0.15) -> Dict:
    os.makedirs(cache_dir, exist_ok=True)
    cp = cache_path(cache_dir, addr if not bbox else f"{addr}|{bbox}")
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    params = {
        "apikey": api_key,
        "geocode": addr,
        "lang": "tr_TR",
        "format": "json",
        "results": 5,
    }
    if bbox:
        params["bbox"] = bbox
        params["rspn"] = 1

    for i in range(max_retries):
        try:
            r = requests.get(GEOCODE_URL, params=params, timeout=10)
            if r.status_code == 429:
                # rate limited: exponential-ish backoff
                time.sleep(1.0 + i * 1.5)
                continue
            r.raise_for_status()
            data = r.json()
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            # Respect QPS
            time.sleep(qps_delay)
            return data
        except Exception as e:
            if i == max_retries - 1:
                # store a minimal error payload to avoid retry storms later
                err = {"error": str(e)}
                try:
                    with open(cp, "w", encoding="utf-8") as f:
                        json.dump(err, f)
                except Exception:
                    pass
                return err
            time.sleep(1.0 + i)


def extract_components(y_json: Dict) -> Tuple[Dict[str, Optional[str]], Dict]:
    comp = {"IL": None, "ILCE": None, "MAHALLE": None, "CADDE": None, "SOKAK": None, "NO": None,
            "POI": None}
    meta = {"precision": None, "kind": None}
    try:
        members = y_json.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
        if not members:
            return comp, {"precision": "none"}
        # Pick best candidate: prefer precision exact if present
        best_idx = 0
        for idx, m in enumerate(members):
            md = m.get("GeoObject", {}).get("metaDataProperty", {}).get("GeocoderMetaData", {})
            if md.get("precision") == "exact":
                best_idx = idx
                break

        obj = members[best_idx].get("GeoObject", {})
        md = obj.get("metaDataProperty", {}).get("GeocoderMetaData", {})
        meta["precision"] = md.get("precision")
        meta["kind"] = md.get("kind")
        comps = md.get("Address", {}).get("Components", [])
        # Build kind->name map (note: Yandex uses kinds like province, locality, district, street, house, etc.)
        kd = {}
        for c in comps:
            k = c.get("kind")
            n = c.get("name")
            if k and n:
                # Keep first occurrence for that kind
                if k not in kd:
                    kd[k] = n

        # Map to our component slots
        comp["IL"] = kd.get("province") or kd.get("area") or kd.get("locality")
        comp["ILCE"] = kd.get("district") or kd.get("subdistrict")
        comp["MAHALLE"] = kd.get("neighborhood") or kd.get("other") or kd.get("subpremise")
        # street/thoroughfare may represent street/avenue; place into CADDE/SOKAK
        street = kd.get("street") or kd.get("thoroughfare")
        comp["CADDE"] = street
        comp["SOKAK"] = street
        comp["NO"] = kd.get("house") or kd.get("building")
        # POI could be amenity, premise, or name when amenity present
        comp["POI"] = kd.get("amenity") or kd.get("premise") or kd.get("building") or kd.get("establishment")

        # Also try to extract lat/lon
        pos = obj.get("Point", {}).get("pos") or obj.get("boundedBy", {}).get("Envelope", {}).get("lowerCorner")
        if pos and isinstance(pos, str):
            try:
                lon, lat = [float(x) for x in pos.split()[:2]]
                meta["lat"] = lat
                meta["lon"] = lon
            except Exception:
                pass

        return comp, meta
    except Exception:
        return comp, {"precision": "error"}


def tokens_from_norm(norm: str) -> List[str]:
    if not norm:
        return []
    return norm.split()


def bio_align(address_norm: str, components: Dict[str, Optional[str]], min_score: int = 85,
              max_ngram: int = 6) -> Tuple[List[str], List[str], Dict[str, int]]:
    toks = tokens_from_norm(address_norm)
    labels = ["O"] * len(toks)
    # track matched score per component
    comp_scores: Dict[str, int] = {}

    # priority: higher number means higher precedence when overlapping
    priority = {"IL": 90, "ILCE": 80, "MAHALLE": 70, "CADDE": 60, "SOKAK": 60, "NO": 50, "POI": 40}

    def normalize_local(x: Optional[str]) -> str:
        return normalize(x or "")

    # helper to attempt tag assignment while respecting priority
    def try_assign(tag: str, val: Optional[str]):
        if not val:
            return
        v = normalize_local(val)
        if not v:
            return
        best = None  # (score, i, j)
        n = len(toks)
        for i in range(n):
            for j in range(i, min(n, i + max_ngram)):
                frag = " ".join(toks[i:j + 1])
                sc = fuzz.WRatio(frag, v)
                if sc >= min_score and (best is None or sc > best[0]):
                    best = (sc, i, j)
        if best:
            sc, s, e = best
            # check overlapping and priority
            for k in range(s, e + 1):
                cur = labels[k]
                if cur != "O":
                    # existing tag
                    cur_tag = cur.split("-", 1)[-1] if "-" in cur else None
                    cur_pr = priority.get(cur_tag, 0) if cur_tag else 0
                    if cur_pr > priority.get(tag, 0):
                        # cannot override
                        return
            labels[s] = f"B-{tag}"
            for k in range(s + 1, e + 1):
                labels[k] = f"I-{tag}"
            comp_scores[tag] = int(sc)

    # apply tags in desired order (higher priority first)
    for tag in ["IL", "ILCE", "MAHALLE", "CADDE", "SOKAK", "NO", "POI"]:
        try_assign(tag, components.get(tag))

    return toks, labels, comp_scores


def write_outputs(rows: List[Dict], out_jsonl: str, conll_path: Optional[str] = None,
                  review_path: Optional[str] = None, low_score_thresh: int = 85):
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if conll_path:
        with open(conll_path, "w", encoding="utf-8") as fc:
            for r in rows:
                toks = r.get("tokens", [])
                labs = r.get("labels", [])
                for t, l in zip(toks, labs):
                    fc.write(f"{t}\t{l}\n")
                fc.write("\n")

    if review_path:
        with open(review_path, "w", encoding="utf-8") as fr:
            for r in rows:
                # include rows where any matched comp score < low_score_thresh or no components matched
                scores: Dict[str, int] = r.get("metadata", {}).get("comp_scores", {}) or {}
                if not scores or any(v < low_score_thresh for v in scores.values()):
                    fr.write(json.dumps(r, ensure_ascii=False) + "\n")


def label_addresses(input_csv: str, out_jsonl: str, api_key: str, bbox: Optional[str] = None,
                    cache_dir: str = DEFAULT_CACHE_DIR, conll: Optional[str] = None,
                    review: Optional[str] = None, qps: float = 0.2, min_score: int = 85,
                    incremental: bool = True):
    """Label addresses. When incremental=True, append each processed record to output files
    immediately (flush+fsync) so that Ctrl+C won't lose already-processed rows. Supports
    resuming by skipping lines already present in the output JSONL.
    """
    df = pd.read_csv(input_csv)

    # Determine how many rows already written (resume support)
    start_idx = 0
    if incremental and os.path.exists(out_jsonl):
        try:
            with open(out_jsonl, "r", encoding="utf-8") as f:
                for i, _ in enumerate(f, 1):
                    pass
            start_idx = i
        except Exception:
            start_idx = 0

    # If conll exists and incremental, we append to it; same for review
    try:
        iterator = list(df.iterrows())
    except Exception:
        iterator = list(df.iterrows())

    total = len(df)
    for idx, (_, r) in enumerate(tqdm(iterator, total=total, desc="Labeling")):
        # Skip already-written records when resuming
        if incremental and idx < start_idx:
            continue

        _id = r.get("Id") if "Id" in r else None
        orig = str(r.get("address", ""))
        norm = normalize(orig)
        yj = yandex_geocode(orig, api_key, cache_dir=cache_dir, bbox=bbox, qps_delay=qps)
        comps, meta = extract_components(yj)
        toks, labs, comp_scores = bio_align(norm, comps, min_score=min_score)

        row = {
            "Id": int(_id) if _id is not None and not pd.isna(_id) else None,
            "original_text": orig,
            "normalized_text": norm,
            "tokens": toks,
            "labels": labs,
            "metadata": {"provider": "yandex", **(meta or {}), "comp_scores": comp_scores},
        }

        if incremental:
            # append to JSONL
            try:
                with open(out_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
            except Exception:
                # if append fails, continue to next but keep cache
                pass

            # append to conll if requested
            if conll:
                try:
                    with open(conll, "a", encoding="utf-8") as fc:
                        for t, l in zip(toks, labs):
                            fc.write(f"{t}\t{l}\n")
                        fc.write("\n")
                        fc.flush()
                        try:
                            os.fsync(fc.fileno())
                        except Exception:
                            pass
                except Exception:
                    pass

            # append to review file if needed
            if review:
                try:
                    low_score = not comp_scores or any(v < min_score for v in comp_scores.values())
                    if low_score:
                        with open(review, "a", encoding="utf-8") as fr:
                            fr.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fr.flush()
                            try:
                                os.fsync(fr.fileno())
                            except Exception:
                                pass
                except Exception:
                    pass

        else:
            # non-incremental: collect and write at end
            # reuse previous behavior
            try:
                rows.append(row)  # type: ignore[name-defined]
            except Exception:
                rows = [row]

    # If not incremental, write outputs in one go
    if not incremental:
        try:
            write_outputs(rows, out_jsonl, conll_path=conll, review_path=review)  # type: ignore[name-defined]
        except Exception:
            pass


def cli():
    p = argparse.ArgumentParser(description="Yandex -> BIO address labeler")
    p.add_argument("--input", required=True, help="Input CSV with Id,address columns")
    p.add_argument("--out", required=True, help="Output labeled jsonl")
    p.add_argument("--conll", help="Optional conll output path")
    p.add_argument("--review", help="Optional review-needed jsonl path")
    p.add_argument("--bbox", help="Optional bbox to restrict geocoding (minLon,minLat,maxLon,maxLat)")
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Cache directory")
    p.add_argument("--qps", type=float, default=0.2, help="Seconds between requests (min delay)")
    p.add_argument("--min-score", type=int, default=85, help="Minimum fuzzy score to accept match")
    p.add_argument("--api-key", help="Yandex API key (or set YA_API_KEY env var)")
    args = p.parse_args()

    api_key = args.api_key or os.getenv("YA_API_KEY")
    if not api_key:
        raise SystemExit("Provide Yandex API key via --api-key or YA_API_KEY env var")

    label_addresses(args.input, args.out, api_key, bbox=args.bbox, cache_dir=args.cache,
                    conll=args.conll, review=args.review, qps=args.qps, min_score=args.min_score)


if __name__ == "__main__":
    cli()
