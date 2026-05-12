#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGING="${1:-$HOME/Desktop/checkcitation_release}"
ZIP="${STAGING%/}.zip"

if ! command -v rsync >/dev/null; then
    echo "error: rsync is required" >&2
    exit 1
fi

rm -rf "$STAGING" "$ZIP"
mkdir -p "$STAGING"

rsync -a \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='_trash/' \
    --exclude='make_release.sh' \
    --exclude='prevalence/data/papers/' \
    --exclude='*.log' \
    --exclude='results/metadata/partial/' \
    --exclude='results/semantic/partial/' \
    "$PAPER_DIR/" "$STAGING/"

cat >"$STAGING/RELEASE_NOTES.md" <<'EOF'
# Notes for reviewers

This is a trimmed copy of `experiment/paper/`. The following items have been
omitted to keep the archive small; everything is reproducible from what is
included:

- **Cited-paper PDFs** (~160 MB, 20 NeurIPS 2025 papers). The corpus manifest
  at `prevalence/data/corpus_manifest.csv` lists the 20 paper IDs and
  OpenReview URLs. Re-fetch with:

  ```bash
  python -m experiment.paper.prevalence.00_fetch_neurips
  ```

- **Resume checkpoints** (`results/{task}/partial/*.jsonl`). Runners write
  these for resumability; the CSVs in `results/{task}/csv/` are derived from
  them and are included.

- **Run logs** (`*.log`). Debug traces only; not used in the analysis.
EOF

cd "$(dirname "$STAGING")"
zip -rq "$ZIP" "$(basename "$STAGING")"

printf '\nRelease built:\n'
printf '  staging dir : %s (%s)\n' "$STAGING" "$(du -sh "$STAGING" | cut -f1)"
printf '  archive     : %s (%s)\n' "$ZIP" "$(du -sh "$ZIP" | cut -f1)"
printf '\nLocal working tree under %s is unchanged.\n' "$PAPER_DIR"
