# s3_word_diff.py

Scans the S3 bucket `portal-daf-yomi-fixed-text` for `{media_id}.txt` / `{media_id}.pre-fix.time` file pairs, computes word count differences, and writes per-date aggregate stats to `s3_word_diff.csv`. Also serves as the date source for `lecturer_analysis.py`.

## Commands

```bash
# Fetch new S3 data (incremental — only files newer than the latest CSV date)
uv run python s3_word_diff.py --fetch

# Fetch with correct calendar dates (S3 LastModified dates are unreliable)
uv run python s3_word_diff.py --fetch --lesson-dates 2026-03-24,2026-03-25

# Plot P25/P50/P75/TrimAvg over time
uv run python s3_word_diff.py --plot [--last N]

# Plot word-diff bin distributions per date
uv run python s3_word_diff.py --plot-bins [--last N] [--date YYYY-MM-DD]

# Per-file detail for a single date
uv run python s3_word_diff.py --date YYYY-MM-DD [--plot-max N]
```

## Key Arguments

```
--fetch                  Incremental fetch from S3; appends/merges new dates into CSV
--lesson-dates DATES     Comma-separated calendar dates (YYYY-MM-DD). When provided,
                         queries DB (Calendar + View_Media) to map each media_id to its
                         correct calendar date instead of using S3 LastModified.
                         New rows are merged into existing CSV rows if dates collide.
--plot                   Plot metrics from CSV (no S3 access)
--plot-bins              Plot bin distributions from CSV
--date YYYY-MM-DD        Single-date detail mode (per-file breakdown + scatter plot)
--plot-max N             Exclude points above N from --date scatter plot (default: 100000)
--last N                 Only use the last N dates for --plot / --plot-bins
--csv FILE               CSV file path (default: s3_word_diff.csv)
```

## CSV Structure (`s3_word_diff.csv`)

One row per date, aggregated across all file pairs uploaded that day.

| Column | Description |
|--------|-------------|
| Date | Calendar date (YYYY-MM-DD) |
| N | Number of file pairs |
| P25 | 25th percentile of abs word diff |
| P50 | Median abs word diff |
| P75 | 75th percentile of abs word diff |
| TrimAvg | Mean abs word diff for files with diff ≤ 150 |
| >150 | Count of files with abs diff > 150 |
| Bins | Semicolon-separated bin counts: [0-10), [10-20), ..., [140-150), [150+] |

## Date Assignment

- **Default (`--fetch` only):** Uses S3 `LastModified` timestamp as the date. Can be wrong if files are uploaded late or reprocessed.
- **With `--lesson-dates`:** Queries the `vps_daf-yomi` DB (`Calendar` → `View_Media`) to map each `media_id` to its actual study date. S3 files not matched to any provided date are skipped.

## Duplicate Date Merging

If a date already exists in the CSV and new data arrives for the same date (e.g., via `--lesson-dates`), the rows are merged:
- **N, >150, Bins** — summed exactly
- **TrimAvg** — weighted average by non-outlier count
- **P25, P50, P75** — weighted average by N (approximation; raw values not stored)

## DB Connection

Uses `.env` from `C:\portal\transcription\audio_manager\.env`. Connects to `vps_daf-yomi` MSSQL via SQLAlchemy + pyodbc (same as `lecturer_analysis.py`).

## Output

- `s3_word_diff.csv` — main stats file (read by `lecturer_analysis.py` for its date list)
- `output/s3_summary.png` — P25/P50/P75/TrimAvg plot
- `output/s3_bins.png` — bin distribution plot
- `output/s3_word_diff.png` — per-file scatter (date mode only)
