"""
Build unified Benchmark B dataset for CheckCite semantic verification.

Loads all sources, applies label mappings, combines into a single CSV.

Sources:
  - Sarol et al. 2024 (raw annotations, test split only) — CC BY 4.0
  - SemanticCite (sebsigma/SemanticCite-Dataset) — CC BY-NC-4.0
  - CiteME (bethgelab/CiteME) — CC BY-SA 4.0, all VALID
  - CiteBio (human-annotated, biomedical) — all VALID
  - CiteML (human-annotated, ML/CS) — all VALID

Label mapping (2-class: VALID vs MISREPRESENTED):
  Sarol:        ACCURATE → VALID
                CONTRADICT, MISQUOTE, NOT_SUBSTANTIATE, OVERSIMPLIFY → MISREPRESENTED
                IRRELEVANT → MISREPRESENTED (citing an unrelated paper)
                INDIRECT, INDIRECT_NOT_REVIEW, ETIQUETTE → dropped
  SemanticCite: SUPPORTED → VALID
                UNSUPPORTED → MISREPRESENTED
                PARTIALLY_SUPPORTED → dropped (ambiguous)
                UNCERTAIN → dropped
  CiteME/CiteBio/CiteML: all → VALID (ground-truth real citations)

Output columns:
  source, citing_sentence, cited_paper_title, cited_paper_abstract,
  label, original_label
"""

import json
import csv
import re
import os
import zipfile
from collections import Counter
from pathlib import Path

# ── Settings ─────────────────────────────────────────────────────────────────
SAROL_ANNOTATIONS_ZIP = Path("Citation-Integrity/Data/annotations.zip")
SEMANTICCITE_JSON = Path("SemanticCite_dataset.json")
CITEME_CSV = Path("datasets/Citeme.csv")
CITEBIO_CSV = Path("datasets/CiteBio.csv")
CITEML_CSV = Path("datasets/CiteML.csv")
OUTPUT_CSV = Path("benchmark_b_unified.csv")
# ─────────────────────────────────────────────────────────────────────────────

SAROL_LABEL_MAP = {
    "ACCURATE": "VALID",
    "CONTRADICT": "MISREPRESENTED",
    "MISQUOTE": "MISREPRESENTED",
    "NOT_SUBSTANTIATE": "MISREPRESENTED",
    "OVERSIMPLIFY": "MISREPRESENTED",
    "IRRELEVANT": "MISREPRESENTED",
}
SAROL_DROP_LABELS = {"INDIRECT", "INDIRECT_NOT_REVIEW", "ETIQUETTE"}

SEMANTICCITE_LABEL_MAP = {
    "SUPPORTED": "VALID",
    "UNSUPPORTED": "MISREPRESENTED",
}
SEMANTICCITE_DROP_LABELS = {"UNCERTAIN", "PARTIALLY_SUPPORTED"}


def load_sarol() -> list[dict]:
    """Load Sarol et al. test split from raw annotations zip.

    Structure inside the zip:
      annotations/Test/references/<ref_id>.txt    — full text of cited paper
      annotations/Test/citations/<ref_id>/<citing_pmc>_N.json — one citation instance

    Each citation JSON contains:
      citing_paragraph, citation_context (list of {text, start, end}),
      evidence_segments (list of {text, start, end}), label

    The reference .txt files have the title on line 1 (after '# ')
    and the abstract between '## Abstract' and the next '## ' heading.
    """
    rows = []

    with zipfile.ZipFile(SAROL_ANNOTATIONS_ZIP) as z:
        # Step 1: Build reference paper lookup {ref_id -> {title, abstract}}
        ref_files = [
            n for n in z.namelist()
            if "Test/references/" in n and n.endswith(".txt")
            and "__MACOSX" not in n
        ]

        ref_papers = {}
        for rf in ref_files:
            ref_id = os.path.basename(rf).replace(".txt", "")
            content = z.read(rf).decode("utf-8")
            lines = content.strip().split("\n")

            # Title: first line, strip markdown heading
            title = lines[0].lstrip("# ").strip()

            # Abstract: text between '## Abstract' and the next '##' heading
            abstract_lines = []
            in_abstract = False
            for line in lines[1:]:
                if line.strip().lower().startswith("## abstract"):
                    in_abstract = True
                    continue
                if in_abstract and line.strip().startswith("##"):
                    break
                if in_abstract:
                    abstract_lines.append(line.strip())
            abstract = " ".join(abstract_lines).strip()

            ref_papers[ref_id] = {"title": title, "abstract": abstract}

        # Step 2: Parse each citation JSON
        cit_files = [
            n for n in z.namelist()
            if "Test/citations/" in n and n.endswith(".json")
            and "__MACOSX" not in n
        ]

        for cf in cit_files:
            # Path: annotations/Test/citations/<ref_id>/<citing_file>.json
            parts = cf.split("/")
            ref_id = parts[-2]

            data = json.loads(z.read(cf))
            original_label = data["label"]

            # Skip labels outside our mapping
            if original_label in SAROL_DROP_LABELS:
                continue
            mapped_label = SAROL_LABEL_MAP.get(original_label)
            if mapped_label is None:
                continue

            # Citing sentence from citation_context entries
            contexts = data.get("citation_context", [])
            citing_sentence = " ".join(
                c["text"] for c in contexts
            ).replace("<|cit|>", "").replace("<|other_cit|>", "").strip()

            # Reference paper metadata
            ref = ref_papers.get(ref_id, {})

            rows.append({
                "source": "sarol_2024",
                "citing_sentence": citing_sentence,
                "cited_paper_title": ref.get("title", ""),
                "cited_paper_abstract": ref.get("abstract", ""),
                "label": mapped_label,
                "original_label": original_label,
            })

    return rows


def parse_ref_metadata(meta_str: str) -> dict:
    """Parse SemanticCite ref_metadata string into title and abstract."""
    title = ""
    abstract = ""

    title_match = re.search(r"^Title:\s*(.+)$", meta_str, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    abstract_match = re.search(r"Abstract:\s*\n?(.*)", meta_str, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    return {"title": title, "abstract": abstract}


def load_semanticcite() -> list[dict]:
    """Load SemanticCite dataset as unified rows."""
    with open(SEMANTICCITE_JSON) as f:
        data = json.load(f)

    rows = []
    for instance in data:
        original_label = instance["output"]["classification"]
        if original_label in SEMANTICCITE_DROP_LABELS:
            continue
        mapped_label = SEMANTICCITE_LABEL_MAP.get(original_label)
        if mapped_label is None:
            continue

        claim = instance["input"]["claim"]
        meta = parse_ref_metadata(instance["input"].get("ref_metadata", ""))

        rows.append({
            "source": "semanticcite",
            "citing_sentence": claim,
            "cited_paper_title": meta["title"],
            "cited_paper_abstract": meta["abstract"],
            "label": mapped_label,
            "original_label": original_label,
        })

    return rows


def load_citeme() -> list[dict]:
    """Load CiteME dataset — all instances are ground-truth VALID."""
    rows = []
    with open(CITEME_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source": "citeme",
                "citing_sentence": row["excerpt"],
                "cited_paper_title": row["target_paper_title"],
                "cited_paper_abstract": "",
                "label": "VALID",
                "original_label": "VALID",
            })
    return rows


def load_citebio() -> list[dict]:
    """Load CiteBio (human-annotated biomedical) — all VALID."""
    rows = []
    with open(CITEBIO_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source": "citebio",
                "citing_sentence": row.get("citation_excerpt", "").strip().strip('"'),
                "cited_paper_title": row.get("target_paper_title", ""),
                "cited_paper_abstract": "",
                "label": "VALID",
                "original_label": "VALID",
            })
    return rows


def load_citeml() -> list[dict]:
    """Load CiteML (human-annotated ML/CS) — all VALID."""
    rows = []
    with open(CITEML_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source": "citeml",
                "citing_sentence": row.get("citation_excerpt", "").strip().strip('"'),
                "cited_paper_title": row.get("target_paper_title", ""),
                "cited_paper_abstract": "",
                "label": "VALID",
                "original_label": "VALID",
            })
    return rows


def main():
    all_sources = [
        ("Sarol et al. (test, raw annotations)", load_sarol),
        ("SemanticCite", load_semanticcite),
        ("CiteME", load_citeme),
        ("CiteBio", load_citebio),
        ("CiteML", load_citeml),
    ]

    all_rows = []
    for name, loader in all_sources:
        print(f"Loading {name}...")
        rows = loader()
        labels = Counter(r["label"] for r in rows)
        print(f"  {len(rows)} instances — {dict(labels)}")
        all_rows.extend(rows)

    # Combined summary
    print(f"\n{'='*60}")
    print(f"Combined dataset: {len(all_rows)} instances")
    combined_labels = Counter(r["label"] for r in all_rows)
    for label in ["VALID", "MISREPRESENTED"]:
        count = combined_labels.get(label, 0)
        pct = count / len(all_rows) * 100
        print(f"  {label:20s}: {count:5d}  ({pct:.1f}%)")

    # Per-source breakdown
    print(f"\nPer-source breakdown:")
    for source in ["sarol_2024", "semanticcite", "citeme", "citebio", "citeml"]:
        subset = [r for r in all_rows if r["source"] == source]
        labels = Counter(r["label"] for r in subset)
        print(f"  {source:15s}: N={len(subset):4d}, {dict(labels)}")

    # Verify no empty citing sentences
    empty = [r for r in all_rows if not r["citing_sentence"].strip()]
    if empty:
        print(f"\n  WARNING: {len(empty)} rows have empty citing_sentence!")
    else:
        print(f"\n  All citing sentences non-empty.")

    # Verify all Sarol rows have titles
    sarol_no_title = [
        r for r in all_rows
        if r["source"] == "sarol_2024" and not r["cited_paper_title"].strip()
    ]
    if sarol_no_title:
        print(f"  WARNING: {len(sarol_no_title)} Sarol rows missing cited_paper_title!")
    else:
        print(f"  All Sarol rows have cited_paper_title.")

    # Save CSV
    fieldnames = [
        "source", "citing_sentence", "cited_paper_title",
        "cited_paper_abstract", "label", "original_label",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
