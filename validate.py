"""
validate.py
===========
Validate và report quality của dataset đã sinh.
Chạy sau generate_data.py để kiểm tra trước khi train.

Usage:
    python validate.py --input data/raw_samples.jsonl
    python validate.py --input data/train.jsonl --report
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
DANGER_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)

VALID_TABLES = {
    "users", "organizations", "artists", "albums", "songs",
    "genres", "genre_song", "playlists", "playlist_song", "playlist_user",
    "playlist_folders", "playlist_playlist_folder", "interactions",
    "favorites", "ratings", "podcasts", "podcast_user", "radio_stations",
    "audits", "queue_states", "themes", "transcodes",
}

REFUSAL_KEYWORDS = [
    "không được phép", "không hỗ trợ",
    "ngoài phạm vi", "chỉ hỗ trợ truy vấn select",
    "không có trong schema",
]


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in REFUSAL_KEYWORDS)


def validate_sample(sample: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_issues)"""
    issues = []

    messages = sample.get("messages", [])
    if len(messages) != 3:
        issues.append(f"Expected 3 messages, got {len(messages)}")
        return False, issues

    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        issues.append(f"Wrong roles: {roles}")
        return False, issues

    question = messages[1].get("content", "").strip()
    sql      = messages[2].get("content", "").strip()

    if not question:
        issues.append("Empty question")

    if not sql:
        issues.append("Empty SQL")
        return False, issues

    # Check if it's a refusal (valid for some cases)
    if is_refusal(sql):
        return True, []  # Refusal is valid

    # For non-refusal, check SQL
    sql_upper = sql.upper().strip()

    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        issues.append(f"SQL doesn't start with SELECT/WITH: {sql[:80]}")

    if DANGER_PATTERN.search(sql):
        match = DANGER_PATTERN.search(sql)
        issues.append(f"Dangerous keyword found: {match.group()}")

    # Check markdown fences
    if "```" in sql:
        issues.append("SQL contains markdown fences")

    return len(issues) == 0, issues


def validate_file(input_path: Path, report: bool = False):
    samples  = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"  ✗ Line {i}: JSON parse error: {e}")

    print(f"\n📂 Validating: {input_path}")
    print(f"   Total samples: {len(samples)}")

    valid_count   = 0
    invalid_count = 0
    refusal_count = 0
    issue_counts  = defaultdict(int)
    invalid_lines = []

    for lineno, sample in samples:
        is_valid, issues = validate_sample(sample)

        if is_valid:
            valid_count += 1
            # Count refusals
            sql = sample["messages"][2]["content"]
            if is_refusal(sql):
                refusal_count += 1
        else:
            invalid_count += 1
            invalid_lines.append((lineno, issues))
            for issue in issues:
                issue_counts[issue] += 1

    # Tier distribution
    tier_dist = defaultdict(int)
    for _, sample in samples:
        tier = sample.get("_tier", "unknown")
        tier_dist[tier] += 1

    print(f"\n{'='*50}")
    print(f"  ✅ Valid   : {valid_count:4d} ({valid_count/len(samples)*100:.1f}%)")
    print(f"  ❌ Invalid : {invalid_count:4d} ({invalid_count/len(samples)*100:.1f}%)")
    print(f"  🔄 Refusals: {refusal_count:4d} (subset of valid)")
    print(f"{'='*50}")

    if tier_dist:
        print("\n📊 Tier distribution:")
        for tier in sorted(tier_dist):
            count = tier_dist[tier]
            print(f"   {tier}: {count:4d} samples")

    if invalid_count > 0:
        print(f"\n⚠  Top issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   [{count:3d}x] {issue}")

        if report:
            report_path = input_path.parent / f"validation_report_{input_path.stem}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Validation Report: {input_path}\n")
                f.write(f"Total: {len(samples)} | Valid: {valid_count} | Invalid: {invalid_count}\n\n")
                f.write("Invalid samples:\n")
                for lineno, issues in invalid_lines[:50]:  # top 50
                    f.write(f"  Line {lineno}: {'; '.join(issues)}\n")
            print(f"\n📄 Full report saved: {report_path}")

    return valid_count, invalid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw_samples.jsonl")
    parser.add_argument("--report", action="store_true", help="Save detailed report")
    args = parser.parse_args()

    validate_file(Path(args.input), report=args.report)
