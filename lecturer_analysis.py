"""Build lecturers.json mapping each lecturer to their lessons with word diffs."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import boto3
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ENV_PATH = Path(r"C:\portal\transcription\audio_manager\.env")
CSV_PATH = "s3_word_diff.csv"
JSON_PATH = "lecturers.json"
BUCKET = "portal-daf-yomi-fixed-text"
TIMESTAMP_RE = re.compile(r"^\[\d+\]\s+\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}:\s*")

load_dotenv(ENV_PATH, override=True)


def get_engine():
    driver = os.getenv("DB_DRIVER_WINDOWS", "ODBC Driver 17 for SQL Server")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "1433")
    database = os.getenv("DB_NAME", "vps_daf-yomi")
    user = os.getenv("DB_USER", "readonly")
    password = os.getenv("DB_PASSWORD", "")
    encoded_password = quote_plus(password)
    encoded_driver = quote_plus(driver)
    conn_str = f"mssql+pyodbc://{user}:{encoded_password}@{host}:{port}/{database}?driver={encoded_driver}"
    return create_engine(conn_str)


def strip_timestamps(text_content: str) -> str:
    lines = text_content.splitlines()
    return "\n".join(TIMESTAMP_RE.sub("", line) for line in lines)


def count_words(text_content: str) -> int:
    return len(text_content.split())


def get_word_diff(s3, media_id):
    try:
        txt_obj = s3.get_object(Bucket=BUCKET, Key=f"{media_id}.txt")
        txt_content = txt_obj["Body"].read().decode("utf-8")
        txt_words = count_words(txt_content)

        pfx_obj = s3.get_object(Bucket=BUCKET, Key=f"{media_id}.pre-fix.time")
        pfx_content = pfx_obj["Body"].read().decode("utf-8")
        pfx_words = count_words(strip_timestamps(pfx_content))

        return abs(txt_words - pfx_words)
    except s3.exceptions.NoSuchKey:
        return None


def read_csv_dates():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sorted(row["Date"] for row in reader)


def load_existing_json():
    if not os.path.exists(JSON_PATH):
        return None
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def fetch_lecturers(conn):
    result = conn.execute(text(
        "SELECT [id], [description], [language] FROM [vps_daf-yomi].[daf-yom_sql-user].[maggid_shiur]"
    )).fetchall()
    return {str(row[0]): {"description": row[1], "language": row[2]} for row in result}


def get_calendar_entries(conn, target_date):
    result = conn.execute(
        text("SELECT DISTINCT MassechetId, DafId FROM [vps_daf-yomi].[dbo].[Calendar] WHERE Date = :target_date"),
        {"target_date": target_date},
    ).fetchall()
    return [(row[0], row[1]) for row in result]


def get_media_for_daf(conn, massechet_id, daf_id):
    result = conn.execute(
        text("""
            SELECT [media_id], [maggid_id], [language_en],
                   [maggid_first_name], [maggid_last_name]
            FROM [vps_daf-yomi].[dbo].[View_Media]
            WHERE massechet_id = :massechet_id AND daf_id = :daf_id
              AND language_en = 'hebrew'
        """),
        {"massechet_id": massechet_id, "daf_id": daf_id},
    ).fetchall()
    return [(row[0], row[1], row[2], row[3], row[4]) for row in result]


def plot_lecturers(from_date=None, lecturer_id=None):
    data = load_existing_json()
    if not data:
        print("No lecturers.json found. Run build mode first.")
        return

    lecturers = data["lecturers"]
    if lecturer_id:
        if lecturer_id not in lecturers:
            print(f"Lecturer id={lecturer_id} not found in lecturers.json")
            return
        lecturers = {lecturer_id: lecturers[lecturer_id]}

    all_dates = read_csv_dates()
    if from_date:
        all_dates = [d for d in all_dates if d >= from_date]
    date_to_x = {d: i for i, d in enumerate(all_dates)}

    fig, ax = plt.subplots(figsize=(14, 7))

    for mid, lec in lecturers.items():
        lessons = lec["lessons"]
        valid = [(l["date"], l["word_diff"]) for l in lessons
                 if l.get("word_diff") is not None and l["date"] in date_to_x]
        if not valid:
            continue
        xs = [date_to_x[d] for d, _ in valid]
        diffs = [wd for _, wd in valid]
        label = lec.get("description") or f"id={mid}"
        ax.plot(xs, diffs, "o-", label=label, markersize=4)

    ax.set_xlabel("Date")
    ax.set_ylabel("Word difference (abs)")
    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize="small", ncol=2)
    plt.title("Word Difference per Lecturer over Time")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/lecturers_word_diff.png", dpi=150)
    print("Plot saved to output/lecturers_word_diff.png")
    plt.show()


def build(force_date=None):
    all_dates = read_csv_dates()
    existing = load_existing_json()

    if existing:
        latest_date = existing.get("latest_date", "")
        new_dates = [d for d in all_dates if d > latest_date]
        lecturers = existing.get("lecturers", {})
    else:
        new_dates = all_dates
        lecturers = {}

    if force_date:
        if force_date not in new_dates:
            new_dates.append(force_date)
            new_dates.sort()
        # Remove existing lessons for this date so they get re-fetched
        for lec in lecturers.values():
            lec["lessons"] = [l for l in lec["lessons"] if l.get("date") != force_date]
        print(f"Forcing re-fetch of {force_date}")

    if not new_dates:
        print(f"No new dates. Already up to date (latest: {existing.get('latest_date', 'N/A')}).")
        return

    print(f"Processing {len(new_dates)} new date(s): {new_dates[0]} to {new_dates[-1]}")

    # Build set of existing media_ids to skip S3 lookups
    existing_media_ids = set()
    for lec in lecturers.values():
        for lesson in lec["lessons"]:
            existing_media_ids.add(lesson["media_id"] if isinstance(lesson, dict) else lesson)

    session = boto3.Session(profile_name="portal")
    s3 = session.client("s3")

    engine = get_engine()
    with engine.connect() as conn:
        all_lecturers_info = fetch_lecturers(conn)

        for date_str in new_dates:
            entries = get_calendar_entries(conn, date_str)
            if not entries:
                print(f"  {date_str}: no calendar entries")
                continue

            date_media_count = 0
            for massechet_id, daf_id in entries:
                media_rows = get_media_for_daf(conn, massechet_id, daf_id)
                for media_id, maggid_id, language_en, first_name, last_name in media_rows:
                    if maggid_id is None:
                        continue
                    if media_id in existing_media_ids:
                        continue
                    mid = str(maggid_id)
                    if mid not in lecturers:
                        info = all_lecturers_info.get(mid, {})
                        lecturers[mid] = {
                            "description": info.get("description", "Unknown"),
                            "language": language_en or "",
                            "maggid_first_name": first_name or "",
                            "maggid_last_name": last_name or "",
                            "lessons": [],
                        }
                    word_diff = get_word_diff(s3, media_id)
                    lecturers[mid]["lessons"].append({
                        "media_id": media_id,
                        "date": date_str,
                        "word_diff": word_diff,
                    })
                    lecturers[mid]["lessons"].sort(key=lambda x: x["date"])
                    existing_media_ids.add(media_id)
                    date_media_count += 1

            print(f"  {date_str}: {len(entries)} calendar entries, {date_media_count} new lessons added")

            # Recalculate averages and save after each date
            for lec in lecturers.values():
                diffs = [l["word_diff"] for l in lec["lessons"] if l.get("word_diff") is not None]
                lec["avg_word_diff"] = round(sum(diffs) / len(diffs), 1) if diffs else None
            output = {"latest_date": date_str, "lecturers": lecturers}
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

    total_lessons = sum(len(l["lessons"]) for l in lecturers.values())
    print(f"\nDone. {len(lecturers)} lecturers, {total_lessons} total lessons. Written to {JSON_PATH}")


def stats():
    data = load_existing_json()
    if not data:
        print("No lecturers.json found. Run build mode first.")
        return

    lecturers = data["lecturers"]

    increasing = []
    above_avg = []

    for mid, lec in lecturers.items():
        valid = [l for l in lec["lessons"] if l.get("word_diff") is not None]
        if len(valid) < 2:
            continue
        last = valid[-1]["word_diff"]
        prev = valid[-2]["word_diff"]
        desc = lec.get("description") or f"id={mid}"
        avg = lec.get("avg_word_diff")

        if last > prev:
            increasing.append((mid, desc, prev, valid[-2]["date"], last, valid[-1]["date"]))

        if avg is not None and last > avg:
            above_avg.append((mid, desc, last, valid[-1]["date"], avg))

    print(f"=== Lecturers where latest > previous ({len(increasing)}) ===")
    print(f"{'ID':<6} {'Lecturer':<30} {'Previous':>10} {'Date':>12} {'Latest':>10} {'Date':>12}")
    print("-" * 82)
    for mid, desc, prev, prev_date, last, last_date in sorted(increasing, key=lambda x: x[4], reverse=True):
        print(f"{mid:<6} {desc:<30} {prev:>10} {prev_date:>12} {last:>10} {last_date:>12}")

    print(f"\n=== Lecturers where latest > average ({len(above_avg)}) ===")
    print(f"{'ID':<6} {'Lecturer':<30} {'Latest':>10} {'Date':>12} {'Average':>10}")
    print("-" * 70)
    for mid, desc, last, last_date, avg in sorted(above_avg, key=lambda x: x[2], reverse=True):
        print(f"{mid:<6} {desc:<30} {last:>10} {last_date:>12} {avg:>10}")


def main():
    parser = argparse.ArgumentParser(description="Lecturer performance analysis")
    parser.add_argument("--build", action="store_true", help="Build/update lecturers.json from DB + S3")
    parser.add_argument("--date", type=str, default=None, help="Re-fetch a specific date (YYYY-MM-DD), even if already in lecturers.json")
    parser.add_argument("--plot", action="store_true", help="Plot word diff per lecturer")
    parser.add_argument("--stats", action="store_true", help="Print lecturer statistics")
    parser.add_argument("--from-date", type=str, default=None, help="Start plot from this date (YYYY-MM-DD)")
    parser.add_argument("--lecturer-id", type=str, default=None, help="Plot only a specific lecturer by maggid_id")
    args = parser.parse_args()

    if args.build:
        build(force_date=args.date)
    elif args.plot:
        plot_lecturers(from_date=args.from_date, lecturer_id=args.lecturer_id)
    elif args.stats:
        stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
