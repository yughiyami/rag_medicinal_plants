"""
Emit the public record list of the indexed corpus (paper R2).

Writes data/corpus_ids.csv: one row per indexed DOCUMENT with its identifiers,
source repository and matched species. Abstract text is deliberately NOT
included -- the abstracts are third-party copyrighted material, so the corpus is
released as a reproducible list of identifiers that anyone can re-fetch from the
original repositories, not as a redistribution of their text.

Usage:
    python make_corpus_ids.py --store <vectorstore dir> [--out data/corpus_ids.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = ["record_id", "pmid", "doi", "source", "year", "journal", "species", "n_chunks"]


def load_chunk_metadata(store: Path) -> list[dict]:
    """Return the per-chunk metadata list, whatever key the store wraps it in."""
    blob = json.loads((store / "metadata.json").read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return blob
    for value in blob.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    raise SystemExit("could not locate the chunk metadata list in metadata.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True, help="vectorstore directory")
    ap.add_argument("--out", type=Path, default=Path("data/corpus_ids.csv"))
    args = ap.parse_args()

    chunks = load_chunk_metadata(args.store)

    docs: dict[str, dict] = {}
    for c in chunks:
        pmid = (c.get("pmid") or "").strip()
        doi = (c.get("doi") or "").strip()
        key = f"pmid:{pmid}" if pmid else (f"doi:{doi}" if doi else f"title:{c.get('title', '')[:120]}")
        rec = docs.setdefault(key, {
            "record_id": key, "pmid": pmid, "doi": doi,
            "source": c.get("source", ""), "year": str(c.get("year") or ""),
            "journal": c.get("journal", ""), "species": set(), "n_chunks": 0,
        })
        rec["species"].update(c.get("species") or [])
        rec["n_chunks"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rec in sorted(docs.values(), key=lambda r: r["record_id"]):
            w.writerow({**rec, "species": "; ".join(sorted(rec["species"]))})

    species = {s for r in docs.values() for s in r["species"]}
    print(f"wrote {args.out}: {len(docs)} documents, {len(chunks)} chunks, {len(species)} species")


if __name__ == "__main__":
    main()
