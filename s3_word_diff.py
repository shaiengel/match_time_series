"""Scan S3 bucket for {id}.txt / {id}.pre-fix.time pairs and compare word counts."""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import boto3
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ENV_PATH = Path(r"C:\portal\transcription\audio_manager\.env")
load_dotenv(ENV_PATH, override=True)

BUCKET = "portal-daf-yomi-fixed-text"
VTT_BUCKET = "final-transcription"
TIMESTAMP_RE = re.compile(r"^\[\d+\]\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}:\s*")
VTT_TIME_RE = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+(\d{2}):(\d{2}):(\d{2})\.(\d{3})')
CSV_PATH = "s3_word_diff.csv"
TRIM_CAP = 150
BIN_EDGES = list(range(0, 160, 10))  # [0, 10, 20, ..., 150]


def get_engine():
    driver = os.getenv("DB_DRIVER_WINDOWS", "ODBC Driver 17 for SQL Server")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "1433")
    database = os.getenv("DB_NAME", "vps_daf-yomi")
    user = os.getenv("DB_USER", "readonly")
    password = os.getenv("DB_PASSWORD", "")
    conn_str = (
        f"mssql+pyodbc://{user}:{quote_plus(password)}@{host}:{port}/{database}"
        f"?driver={quote_plus(driver)}"
    )
    return create_engine(conn_str)


def get_media_ids_for_dates(dates):
    """Query DB to get a mapping of media_id -> calendar_date for the given dates."""
    engine = get_engine()
    media_id_to_date = {}
    with engine.connect() as conn:
        for date_str in dates:
            calendar_rows = conn.execute(
                text("SELECT DISTINCT MassechetId, DafId FROM [vps_daf-yomi].[dbo].[Calendar] WHERE Date = :d"),
                {"d": date_str},
            ).fetchall()
            for massechet_id, daf_id in calendar_rows:
                media_rows = conn.execute(
                    text("""
                        SELECT [media_id] FROM [vps_daf-yomi].[dbo].[View_Media]
                        WHERE massechet_id = :mid AND daf_id = :did AND language_en = 'hebrew'
                    """),
                    {"mid": massechet_id, "did": daf_id},
                ).fetchall()
                for (media_id,) in media_rows:
                    media_id_to_date[str(media_id)] = date_str
    return media_id_to_date


def parse_vtt_duration(content: str):
    """Return duration in seconds from the last cue end-timestamp in a VTT file."""
    last_match = None
    for m in VTT_TIME_RE.finditer(content):
        last_match = m
    if not last_match:
        return None
    h, m, s, ms = int(last_match.group(1)), int(last_match.group(2)), int(last_match.group(3)), int(last_match.group(4))
    return h * 3600 + m * 60 + s + ms / 1000


def get_vtt_duration(s3, file_id: str):
    """Fetch {file_id}.vtt from VTT_BUCKET and return duration in seconds, or None if missing."""
    try:
        obj = s3.get_object(Bucket=VTT_BUCKET, Key=f"{file_id}.vtt")
        content = obj["Body"].read().decode("utf-8")
        return parse_vtt_duration(content)
    except s3.exceptions.NoSuchKey:
        return None


def strip_timestamps(text: str) -> str:
    """Remove [N] HH:MM:SS.mmm - HH:MM:SS.mmm: prefix from each line."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        cleaned.append(TIMESTAMP_RE.sub("", line))
    return "\n".join(cleaned)


def count_words(text: str) -> int:
    return len(text.split())


def fetch_all_pairs(s3, after_date=None):
    """Fetch file pairs from S3. If after_date given, only files modified after that date."""
    paginator = s3.get_paginator("list_objects_v2")
    all_objects = []
    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            all_objects.append(obj)

    if after_date:
        all_objects = [obj for obj in all_objects if obj["LastModified"].strftime("%Y-%m-%d") > after_date]
        print(f"  Filtering S3 objects after {after_date}: {len(all_objects)} files")

    key_to_date = {obj["Key"]: obj["LastModified"].strftime("%Y-%m-%d") for obj in all_objects}
    all_keys = set(key_to_date.keys())

    txt_files = {k for k in all_keys if k.endswith(".txt") and not k.endswith(".pre-fix.time")}
    prefix_files = {k for k in all_keys if k.endswith(".pre-fix.time")}

    txt_by_id = {}
    for k in txt_files:
        file_id = k.rsplit(".", 1)[0]
        txt_by_id[file_id] = k

    prefix_by_id = {}
    for k in prefix_files:
        file_id = k.replace(".pre-fix.time", "")
        prefix_by_id[file_id] = k

    common_ids = sorted(set(txt_by_id.keys()) & set(prefix_by_id.keys()))
    print(f"  Found {len(common_ids)} pairs")

    results = []
    for i, file_id in enumerate(common_ids):
        if i % 50 == 0:
            print(f"    Processing {i}/{len(common_ids)}...")
        txt_obj = s3.get_object(Bucket=BUCKET, Key=txt_by_id[file_id])
        txt_content = txt_obj["Body"].read().decode("utf-8")
        txt_words = count_words(txt_content)

        pfx_obj = s3.get_object(Bucket=BUCKET, Key=prefix_by_id[file_id])
        pfx_content = pfx_obj["Body"].read().decode("utf-8")
        pfx_cleaned = strip_timestamps(pfx_content)
        pfx_words = count_words(pfx_cleaned)

        abs_diff = abs(txt_words - pfx_words)
        duration = get_vtt_duration(s3, file_id)
        date = key_to_date[txt_by_id[file_id]]
        results.append((date, file_id, abs_diff, duration))

    return results


def compute_bins(diffs):
    """Compute bin counts for bins [0,10), [10,20), ..., [140,150), [150,inf). Returns 16 values."""
    num_bins = len(BIN_EDGES)  # 16: indices 0-14 for [0,10)..[140,150), index 15 for [150,inf)
    counts = [0] * num_bins
    for d in diffs:
        if d >= TRIM_CAP:
            counts[num_bins - 1] += 1
        else:
            idx = int(d // 10)
            counts[idx] += 1
    return counts


def compute_date_stats(diffs):
    """Compute stats for a list of abs_diffs."""
    arr = np.array(diffs)
    n = len(arr)
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    trimmed = arr[arr <= TRIM_CAP]
    trim_avg = float(np.mean(trimmed)) if len(trimmed) > 0 else 0.0
    above_150 = int(np.sum(arr > TRIM_CAP))
    bins = compute_bins(diffs)
    return {
        'N': n, 'P25': round(p25, 1), 'P50': round(p50, 1), 'P75': round(p75, 1),
        'TrimAvg': round(trim_avg, 1), '>150': above_150,
        'Bins': ';'.join(str(b) for b in bins),
    }


def merge_date_stats(existing, new):
    """Merge two stats dicts for the same date. Returns a new merged dict."""
    n1, n2 = int(existing['N']), int(new['N'])
    n = n1 + n2
    above1, above2 = int(existing['>150']), int(new['>150'])
    trim1 = n1 - above1
    trim2 = n2 - above2
    trim_total = trim1 + trim2

    def wavg(v1, v2):
        return round((float(v1) * n1 + float(v2) * n2) / n, 1) if n else 0.0

    bins1 = [int(b) for b in existing['Bins'].split(';')]
    bins2 = [int(b) for b in new['Bins'].split(';')]
    merged_bins = [a + b for a, b in zip(bins1, bins2)]

    if trim_total > 0:
        trim_avg = round((float(existing['TrimAvg']) * trim1 + float(new['TrimAvg']) * trim2) / trim_total, 1)
    else:
        trim_avg = 0.0

    dur1 = float(existing.get('TotalDuration') or 0)
    dur2 = float(new.get('TotalDuration') or 0)

    return {
        'Date': existing['Date'],
        'N': n,
        'P25': wavg(existing['P25'], new['P25']),
        'P50': wavg(existing['P50'], new['P50']),
        'P75': wavg(existing['P75'], new['P75']),
        'TrimAvg': trim_avg,
        '>150': above1 + above2,
        'Bins': ';'.join(str(b) for b in merged_bins),
        'TotalDuration': int(dur1 + dur2),
    }


def read_csv(csv_path):
    """Read existing CSV, return list of row dicts and the latest date (or None)."""
    if not os.path.exists(csv_path):
        return [], None
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return [], None
    latest_date = max(row['Date'] for row in rows)
    return rows, latest_date


def write_csv(csv_path, rows):
    """Write rows (list of dicts) to CSV, sorted by Date ascending."""
    rows.sort(key=lambda r: r['Date'])
    fieldnames = ['Date', 'N', 'P25', 'P50', 'P75', 'TrimAvg', '>150', 'Bins', 'TotalDuration']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(rows)


def fetch_mode(s3, csv_path, lesson_dates=None):
    """Fetch from S3 (incremental), compute per-date stats, write CSV."""
    existing_rows, latest_date = read_csv(csv_path)

    if latest_date:
        print(f"CSV has data up to {latest_date}. Fetching newer dates...")
    else:
        print("No existing CSV. Fetching all dates...")

    results = fetch_all_pairs(s3, after_date=latest_date)

    if not results:
        print("No new data to fetch. CSV is up to date.")
        return

    if lesson_dates:
        print(f"Looking up media IDs in DB for dates: {', '.join(lesson_dates)}")
        media_id_to_date = get_media_ids_for_dates(lesson_dates)
        print(f"  Found {len(media_id_to_date)} media IDs across {len(lesson_dates)} date(s)")
        mapped_results = []
        unmatched = 0
        for _s3_date, file_id, abs_diff, duration in results:
            cal_date = media_id_to_date.get(file_id)
            if cal_date:
                mapped_results.append((cal_date, file_id, abs_diff, duration))
            else:
                unmatched += 1
        if unmatched:
            print(f"  {unmatched} S3 file(s) not matched to any provided date — skipped")
        results = mapped_results

    if not results:
        print("No matched data to add to CSV.")
        return

    # Group by date
    by_date = defaultdict(list)
    by_date_duration = defaultdict(float)
    for date, file_id, abs_diff, duration in results:
        by_date[date].append(abs_diff)
        if duration is not None:
            by_date_duration[date] += duration

    new_dates = sorted(by_date.keys())
    print(f"\n  New dates: {len(new_dates)}")
    print(f"\n{'Date':<14} {'N':>5} {'P25':>8} {'P50':>8} {'P75':>8} {'TrimAvg':>8} {'>150':>6} {'Duration':>12}")
    print("-" * 75)

    new_rows = []
    for date in new_dates:
        stats = compute_date_stats(by_date[date])
        stats['Date'] = date
        stats['TotalDuration'] = int(by_date_duration[date])
        new_rows.append(stats)
        dur_secs = stats['TotalDuration']
        dur_str = f"{dur_secs // 3600}h{(dur_secs % 3600) // 60}m"
        print(f"{date:<14} {stats['N']:>5} {stats['P25']:>8} {stats['P50']:>8} {stats['P75']:>8} {stats['TrimAvg']:>8} {stats['>150']:>6} {dur_str:>12}")

    existing_by_date = {r['Date']: r for r in existing_rows}
    for row in new_rows:
        date = row['Date']
        if date in existing_by_date:
            print(f"  Merging duplicate date {date} (existing N={existing_by_date[date]['N']}, new N={row['N']})")
            existing_by_date[date] = merge_date_stats(existing_by_date[date], row)
        else:
            existing_by_date[date] = row
    write_csv(csv_path, list(existing_by_date.values()))
    print(f"\nCSV written to {csv_path} ({len(existing_by_date)} dates total)")


def plot_mode(csv_path, last=None):
    """Read CSV and plot P25/P50/P75/TrimAvg + >150 bars."""
    rows, _ = read_csv(csv_path)
    if not rows:
        print(f"No data in {csv_path}. Run --fetch first.")
        return
    if last:
        rows = rows[-last:]

    dates = [r['Date'] for r in rows]
    p25s = [float(r['P25']) for r in rows]
    p50s = [float(r['P50']) for r in rows]
    p75s = [float(r['P75']) for r in rows]
    trim_avgs = [float(r['TrimAvg']) for r in rows]
    high_counts = [int(r['>150']) for r in rows]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(dates))
    ax1.plot(x, p25s, 'o-', color='green', label='P25', markersize=5)
    ax1.plot(x, p50s, 's-', color='blue', label='P50 (median)', markersize=5)
    ax1.plot(x, p75s, '^-', color='orange', label='P75', markersize=5)
    ax1.plot(x, trim_avgs, 'D-', color='purple', label=f'Trimmed avg (≤{TRIM_CAP})', markersize=5)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Word difference')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(dates, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.bar(x, high_counts, alpha=0.2, color='red', label=f'Count > {TRIM_CAP}', width=0.4)
    ax2.set_ylabel(f'Count > {TRIM_CAP}', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Word Difference Metrics by Date')
    plt.tight_layout()
    plt.savefig("output/s3_summary.png", dpi=150)
    print(f"Plot saved to output/s3_summary.png")
    plt.show()


def plot_bins_mode(csv_path, last=None, date=None):
    """Read CSV and plot bin distributions — one line per date."""
    rows, _ = read_csv(csv_path)
    if not rows:
        print(f"No data in {csv_path}. Run --fetch first.")
        return
    if date:
        rows = [r for r in rows if r['Date'] == date]
        if not rows:
            print(f"No data for date {date} in {csv_path}.")
            return
    elif last:
        rows = rows[-last:]

    # Bin labels for x-axis
    bin_labels = [f"{i}-{i+10}" for i in range(0, 150, 10)] + ["150+"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(bin_labels))

    single_date = len(rows) == 1
    for row in rows:
        bins = [int(b) for b in row['Bins'].split(';')]
        if single_date:
            ax.bar(x, bins, color='steelblue', alpha=0.8, label=row['Date'])
        else:
            ax.plot(x, bins, 'o-', label=row['Date'], markersize=4)

    ax.set_xlabel('Word difference range')
    ax.set_ylabel('Number of files')
    ax.set_xticks(list(x))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.title('Distribution of Word Differences by Date')
    plt.tight_layout()
    plt.savefig("output/s3_bins.png", dpi=150)
    print(f"Plot saved to output/s3_bins.png")
    plt.show()


def date_mode(s3, args):
    """Original per-file detail mode for a single date."""
    paginator = s3.get_paginator("list_objects_v2")
    all_objects = []
    for page in paginator.paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            all_objects.append(obj)

    filter_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    all_objects = [obj for obj in all_objects if obj["LastModified"].date() == filter_date]
    print(f"Filtering by date: {args.date} ({len(all_objects)} files match)\n")

    all_keys = {obj["Key"] for obj in all_objects}

    txt_files = {k for k in all_keys if k.endswith(".txt") and not k.endswith(".pre-fix.time")}
    prefix_files = {k for k in all_keys if k.endswith(".pre-fix.time")}

    txt_by_id = {}
    for k in txt_files:
        file_id = k.rsplit(".", 1)[0]
        txt_by_id[file_id] = k

    prefix_by_id = {}
    for k in prefix_files:
        file_id = k.replace(".pre-fix.time", "")
        prefix_by_id[file_id] = k

    common_ids = sorted(set(txt_by_id.keys()) & set(prefix_by_id.keys()))
    print(f"Found {len(common_ids)} pairs\n")

    if not common_ids:
        print("No pairs found.")
        return

    OUTLIER_THRESHOLD = 150

    abs_diffs = []
    pfx_word_counts = []
    outliers = []
    print(f"{'ID':<20} {'txt words':>12} {'prefix words':>12} {'diff':>10} {'abs diff':>10}")
    print("-" * 70)

    for file_id in common_ids:
        txt_obj = s3.get_object(Bucket=BUCKET, Key=txt_by_id[file_id])
        txt_content = txt_obj["Body"].read().decode("utf-8")
        txt_words = count_words(txt_content)

        pfx_obj = s3.get_object(Bucket=BUCKET, Key=prefix_by_id[file_id])
        pfx_content = pfx_obj["Body"].read().decode("utf-8")
        pfx_cleaned = strip_timestamps(pfx_content)
        pfx_words = count_words(pfx_cleaned)

        diff = txt_words - pfx_words
        abs_diff = abs(diff)
        abs_diffs.append(abs_diff)
        pfx_word_counts.append(pfx_words)

        marker = " <<<" if abs_diff > OUTLIER_THRESHOLD else ""
        print(f"{file_id:<20} {txt_words:>12} {pfx_words:>12} {diff:>10} {abs_diff:>10}{marker}")

        if abs_diff > OUTLIER_THRESHOLD:
            outliers.append((file_id, txt_words, pfx_words, diff, abs_diff))

    filtered_diffs = [d for d in abs_diffs if d <= args.plot_max]
    avg_all = sum(abs_diffs) / len(abs_diffs)
    avg_filtered = sum(filtered_diffs) / len(filtered_diffs) if filtered_diffs else 0
    print("-" * 70)
    print(f"Average absolute difference (all): {avg_all:.2f}")
    print(f"Average absolute difference (plot-max <= {args.plot_max}): {avg_filtered:.2f} ({len(filtered_diffs)}/{len(abs_diffs)} files)")

    if outliers:
        outliers.sort(key=lambda x: x[4], reverse=True)
        print(f"\n{'='*70}")
        print(f"OUTLIERS: {len(outliers)} files with abs diff > {OUTLIER_THRESHOLD} words")
        print(f"{'='*70}")
        print(f"{'ID':<20} {'txt words':>12} {'prefix words':>12} {'diff':>10} {'abs diff':>10}")
        print("-" * 70)
        for file_id, tw, pw, d, ad in outliers:
            print(f"{file_id:<20} {tw:>12} {pw:>12} {d:>10} {ad:>10}")
        print("-" * 70)

    plot_x = [x for x, d in zip(pfx_word_counts, abs_diffs) if d <= args.plot_max]
    plot_y = [d for d in abs_diffs if d <= args.plot_max]
    excluded = len(abs_diffs) - len(plot_y)

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_x, plot_y, s=20, alpha=0.6)
    plt.xlabel("Pre-fix word count")
    plt.ylabel("Absolute word difference")
    title = f"Word count difference vs pre-fix file size (plot-max={args.plot_max})"
    if excluded:
        title += f"\n({excluded} points with abs diff > {args.plot_max} excluded)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig("output/s3_word_diff.png", dpi=150)
    print(f"\nPlot saved to output/s3_word_diff.png ({excluded} points excluded from plot)")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare word counts between S3 file pairs")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch from S3 (incremental) and write per-date stats to CSV")
    parser.add_argument("--lesson-dates", type=str, default=None,
                        help="Comma-separated calendar dates (YYYY-MM-DD) to assign new S3 files to via DB lookup")
    parser.add_argument("--plot", action="store_true",
                        help="Plot metrics from CSV (no S3 access)")
    parser.add_argument("--plot-bins", action="store_true",
                        help="Plot bin distributions from CSV (one line per date)")
    parser.add_argument("--date", type=str, default=None,
                        help="Single-date detail mode (YYYY-MM-DD)")
    parser.add_argument("--plot-max", type=int, default=100000,
                        help="For --date mode: exclude points above this value (default: 100000)")
    parser.add_argument("--csv", type=str, default=CSV_PATH,
                        help=f"CSV file path (default: {CSV_PATH})")
    parser.add_argument("--last", type=int, default=None,
                        help="Only use the last K dates (for --plot and --plot-bins)")
    args = parser.parse_args()

    if args.fetch:
        session = boto3.Session(profile_name="portal")
        s3 = session.client("s3")
        lesson_dates = [d.strip() for d in args.lesson_dates.split(",")] if args.lesson_dates else None
        fetch_mode(s3, args.csv, lesson_dates=lesson_dates)
    elif args.plot:
        plot_mode(args.csv, last=args.last)
    elif args.plot_bins:
        plot_bins_mode(args.csv, last=args.last, date=args.date)
    elif args.date:
        session = boto3.Session(profile_name="portal")
        s3 = session.client("s3")
        date_mode(s3, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
