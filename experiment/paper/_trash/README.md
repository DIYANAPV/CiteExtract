# _trash/

Items moved here during the 2026-05-12 reorganization. Kept (not deleted) in case
they turn out to be needed; safe to remove later.

## Contents

- **`*.jsonl.bak`** (4 files) — backups of partial-result JSONLs that were created
  manually during earlier runs. Originals are still in `results/{task}/partial/`.

- **`*_smoke*` artifacts** (6 files) — outputs from `--smoke` test runs. The smoke
  pipeline still works; these snapshots were ad-hoc, not referenced by analysis.

- **`openweight_full_summary.json`** — combined open-weight summary written by the
  old `openweight/runner.py`. After the refactor, open-weight runs append to
  `results/{task}/summaries/full.json` instead, so this file will regenerate per
  task on the next run.
