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

cd "$(dirname "$STAGING")"
zip -rq "$ZIP" "$(basename "$STAGING")"

printf '\nRelease built:\n'
printf '  staging dir : %s (%s)\n' "$STAGING" "$(du -sh "$STAGING" | cut -f1)"
printf '  archive     : %s (%s)\n' "$ZIP" "$(du -sh "$ZIP" | cut -f1)"
printf '\nLocal working tree under %s is unchanged.\n' "$PAPER_DIR"
