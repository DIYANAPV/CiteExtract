
from __future__ import annotations

import csv
import json
import tempfile


def _write_batch_csv(per_paper: list[dict]) -> str:
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", prefix="citeextract_batch_", delete=False, mode="w",
        newline="", encoding="utf-8",
    )
    writer = csv.writer(tmp)
    writer.writerow([
        "paper", "ref_id", "verdict", "action", "mode",
        "matched_title", "matched_doi", "source", "explanation",
    ])
    for p in per_paper:
        if p["status"] != "ok" or p["paper_report"] is None:
            continue
        name = p["name"]
        for v in p["paper_report"].verdicts:
            ex = v.existence
            writer.writerow([
                name, v.ref_id, v.verdict, v.action, v.mode,
                (ex.matched_title or "") if ex else "",
                (ex.matched_doi or "") if ex else "",
                (ex.source or "") if ex else "",
                v.explanation,
            ])
    tmp.close()
    return tmp.name


def _write_batch_json(per_paper: list[dict], mode: str, elapsed: float) -> str:
    payload = {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "papers": [],
    }
    for p in per_paper:
        entry = {
            "name": p["name"],
            "status": p["status"],
        }
        if p["status"] == "ok":
            entry["total_refs"] = p.get("total_refs", 0)
            entry["counts"] = p.get("counts", {})
            if p["paper_report"]:
                entry["verification"] = p["paper_report"].model_dump(mode="json")
            if p["comp_report"]:
                entry["comprehension"] = p["comp_report"].model_dump(mode="json")
        else:
            entry["error"] = p.get("error", "unknown error")
        payload["papers"].append(entry)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", prefix="citeextract_batch_", delete=False, mode="w",
        encoding="utf-8",
    )
    tmp.write(json.dumps(payload, indent=2, default=str))
    tmp.close()
    return tmp.name
