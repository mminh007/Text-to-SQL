"""
scripts/reprocess_prompts.py
=============================
Cập nhật system prompt trong các file JSONL hiện có (train/val/test)
mà KHÔNG cần generate lại data từ API.

Dùng khi:
  - Schema thay đổi (thêm bảng/cột) → cập nhật schema.py
  - System prompt thay đổi (rules, CoT format)
  - Muốn đồng bộ data với SYSTEM_PROMPT mới nhất từ schema.py

Data pipeline:
  raw_samples.jsonl (API-generated SQL pairs) — GIỮ NGUYÊN
         ↓  reprocess_prompts.py thay system prompt
  train.jsonl / val.jsonl / test.jsonl — cập nhật tại chỗ

Không gọi API, không xóa SQL, chỉ thay messages[0].content.

Usage:
    # Xem trước: in diff system prompt cũ vs mới (không thay đổi file)
    python scripts/reprocess_prompts.py --dry-run

    # Cập nhật train/val/test với system prompt mới (backup tự động)
    python scripts/reprocess_prompts.py

    # Cập nhật cả raw_samples.jsonl
    python scripts/reprocess_prompts.py --include-raw

    # Dùng CoT system prompt (cho model đã train với CoT data)
    python scripts/reprocess_prompts.py --with-cot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Thêm project root vào path để import từ src/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import build_system_prompt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_FILES = [
    DATA_DIR / "train.jsonl",
    DATA_DIR / "val.jsonl",
    DATA_DIR / "test.jsonl",
]
RAW_FILE = DATA_DIR / "raw_samples.jsonl"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(path: Path, samples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def backup(path: Path) -> Path:
    """Create a .bak copy before overwriting."""
    bak = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, bak)
    return bak


def get_old_prompt(sample: dict) -> str:
    """Extract current system prompt from a sample's messages."""
    msgs = sample.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        return msgs[0]["content"]
    return ""


def update_prompt(sample: dict, new_prompt: str) -> dict:
    """Return a new sample dict with updated system prompt."""
    msgs = list(sample["messages"])  # shallow copy
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {**msgs[0], "content": new_prompt}
    else:
        # No system message found — prepend one
        msgs.insert(0, {"role": "system", "content": new_prompt})
    return {**sample, "messages": msgs}


def diff_prompts(old: str, new: str) -> None:
    """Print a simple side-by-side comparison of old vs new system prompt."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    max_len = max(len(old_lines), len(new_lines))

    changes = 0
    for i in range(max_len):
        o = old_lines[i] if i < len(old_lines) else "<missing>"
        n = new_lines[i] if i < len(new_lines) else "<missing>"
        if o != n:
            changes += 1
            if changes <= 20:   # show first 20 diff lines
                print(f"  L{i+1:3d}  OLD: {o[:100]}")
                print(f"       NEW: {n[:100]}")
    if changes == 0:
        print("  (prompts are identical — no changes needed)")
    elif changes > 20:
        print(f"  ... and {changes - 20} more differing lines")
    print(f"\n  Total changed lines: {changes}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def reprocess(
    files: list[Path],
    new_prompt: str,
    dry_run: bool = False,
    backup_files: bool = True,
) -> None:
    """
    Update system prompt in all samples across the given files.

    Args:
        files      : list of JSONL paths to update
        new_prompt : new system prompt string (from schema.py)
        dry_run    : if True, print stats only — do NOT write any files
        backup_files: if True, create .bak before overwriting
    """
    for path in files:
        if not path.exists():
            print(f"  ⏭  {path.name} — not found, skipping")
            continue

        samples = load_jsonl(path)
        if not samples:
            print(f"  ⏭  {path.name} — empty, skipping")
            continue

        # Detect how many samples have a different prompt
        old_prompt = get_old_prompt(samples[0])
        n_changed = sum(
            1 for s in samples
            if get_old_prompt(s) != new_prompt
        )
        n_total = len(samples)

        print(f"\n📄 {path.name}  ({n_total} samples)")
        print(f"   Samples needing update: {n_changed}/{n_total}")

        if n_changed == 0:
            print("   ✅ Already up-to-date — skipping")
            continue

        if dry_run:
            print("   [DRY-RUN] System prompt diff (first sample):")
            diff_prompts(old_prompt, new_prompt)
            continue

        # Backup
        if backup_files:
            bak = backup(path)
            print(f"   📦 Backup → {bak.name}")

        # Rewrite
        updated = [update_prompt(s, new_prompt) for s in samples]
        save_jsonl(path, updated)
        print(f"   ✅ Updated {n_changed} samples → {path.name}")


def main() -> None:
    import sys as _sys
    # Fix Unicode output trên Windows terminal (cp1252 -> utf-8)
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Cap nhat system prompt trong data files ma khong generate lai"
    )
    parser.add_argument(
        "--dry-run",      action="store_true",
        help="Chi in diff, khong ghi file"
    )
    parser.add_argument(
        "--no-backup",    action="store_true",
        help="Bỏ qua tạo file .bak trước khi ghi đè"
    )
    parser.add_argument(
        "--include-raw",  action="store_true",
        help="Cũng cập nhật data/raw_samples.jsonl"
    )
    parser.add_argument(
        "--with-cot",     action="store_true",
        help="Dùng CoT system prompt (thêm <think> instruction)"
    )
    parser.add_argument(
        "--files", nargs="+", type=Path,
        help="Danh sách file cụ thể (mặc định: train/val/test.jsonl)"
    )
    args = parser.parse_args()

    # Build new system prompt from schema.py
    new_prompt = build_system_prompt(with_cot=args.with_cot)

    print("=" * 60)
    print("[*] Reprocess Prompts")
    print(f"    with_cot : {args.with_cot}")
    print(f"    dry_run  : {args.dry_run}")
    print(f"    backup   : {not args.no_backup}")
    print("=" * 60)
    print(f"\n[prompt] New system prompt ({len(new_prompt)} chars):")
    for line in new_prompt.splitlines()[:3]:
        print(f"   {line}")
    print("   ...")

    # Determine target files
    target_files = args.files if args.files else list(DEFAULT_FILES)
    if args.include_raw:
        target_files = [RAW_FILE] + target_files

    reprocess(
        files=target_files,
        new_prompt=new_prompt,
        dry_run=args.dry_run,
        backup_files=not args.no_backup,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Không có file nào bị thay đổi.")
        print("Chạy lại không có --dry-run để áp dụng.")
    else:
        print("\n✅ Xong. Các file cũ được lưu dưới dạng .bak nếu cần rollback.")
        print("   Để rollback: copy .bak lại thành .jsonl")
        print("\nBước tiếp theo:")
        print("  python scripts/reprocess_prompts.py --dry-run   # kiểm tra lần sau")
        print("  python src/training/train.py                    # train với data mới")


if __name__ == "__main__":
    main()
