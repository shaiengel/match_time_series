"""Scan S3 bucket for {id}.txt / {id}.pre-fix.time pairs and compare word counts."""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import boto3
import matplotlib.pyplot as plt
import numpy as np


BUCKET = "portal-daf-yomi-fixed-text"
TIMESTAMP_RE = re.compile(r"^\[\d+\]\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}:\s*")
CSV_PATH = "s3_word_diff.csv"
TRIM_CAP = 150
BIN_EDGES = list(range(0, 160, 10))  # [0, 10, 20, ..., 150]


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
        date = key_to_date[txt_by_id[file_id]]
        results.append((date, file_id, abs_diff))

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
    fieldnames = ['Date', 'N', 'P25', 'P50', 'P75', 'TrimAvg', '>150', 'Bins']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_mode(s3, csv_path):
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

    # Group by date
    by_date = defaultdict(list)
    for date, file_id, abs_diff in results:
        by_date[date].append(abs_diff)

    new_dates = sorted(by_date.keys())
    print(f"\n  New dates: {len(new_dates)}")
    print(f"\n{'Date':<14} {'N':>5} {'P25':>8} {'P50':>8} {'P75':>8} {'TrimAvg':>8} {'>150':>6}")
    print("-" * 60)

    new_rows = []
    for date in new_dates:
        stats = compute_date_stats(by_date[date])
        stats['Date'] = date
        new_rows.append(stats)
        print(f"{date:<14} {stats['N']:>5} {stats['P25']:>8} {stats['P50']:>8} {stats['P75']:>8} {stats['TrimAvg']:>8} {stats['>150']:>6}")

    all_rows = existing_rows + new_rows
    write_csv(csv_path, all_rows)
    print(f"\nCSV written to {csv_path} ({len(all_rows)} dates total)")


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


def plot_bins_mode(csv_path, last=None):
    """Read CSV and plot bin distributions — one line per date."""
    rows, _ = read_csv(csv_path)
    if not rows:
        print(f"No data in {csv_path}. Run --fetch first.")
        return
    if last:
        rows = rows[-last:]

    # Bin labels for x-axis
    bin_labels = [f"{i}-{i+10}" for i in range(0, 150, 10)] + ["150+"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(bin_labels))

    for row in rows:
        date = row['Date']
        bins = [int(b) for b in row['Bins'].split(';')]
        ax.plot(x, bins, 'o-', label=date, markersize=4)

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
        fetch_mode(s3, args.csv)
    elif args.plot:
        plot_mode(args.csv, last=args.last)
    elif args.plot_bins:
        plot_bins_mode(args.csv, last=args.last)
    elif args.date:
        session = boto3.Session(profile_name="portal")
        s3 = session.client("s3")
        date_mode(s3, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
