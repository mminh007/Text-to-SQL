"""
generate_data.py
================
Sinh dataset Text-to-SQL cho Koel Music Streaming DB.
Output: data/raw_samples.jsonl

Yêu cầu cơ bản:
    pip install openai tqdm python-dotenv

Yêu cầu mở rộng (optional features):
    pip install sentence-transformers     # semantic dedup
    pip install pyodbc                    # SQL execution validation (--validate-sql, Windows only)

Cấu hình (.env):
    OPENAI_API_KEY=gsk_...               # Groq API key (lấy tại console.groq.com)
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=qwen/qwen3.8-27b
    SQL_CONN_STR=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=music_14_09_2026;UID=sa;PWD=Aa123456@;

    # Giới hạn request mỗi phiên chạy (dừng chủ động trước khi bị rate-limit):
    DAILY_REQUEST_LIMIT=0   # 0 = không giới hạn; đặt VD 800 nếu quota/ngày là 1000 req

Resume sau rate-limit:
    Script tự lưu progress vào data/progress.json sau mỗi sample.
    Khi bị 429 hoặc đạt DAILY_REQUEST_LIMIT, tự lưu & dừng gracefully.
    Chạy lại cùng lệnh → tự tiếp tục từ chỗ dở, không sinh lại data đã có.
    Xóa data/progress.json để bắt đầu lại từ đầu.

--validate-sql chỉ chạy trên Windows (cần ODBC Driver 17 for SQL Server cài sẵn qua SSMS).
"""

import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ── Optional dependencies (sentence_transformers imported AFTER HF_TOKEN set) ──
# Import được delay xuống dưới load_dotenv() để HF_TOKEN có mặt trước khi
# sentence_transformers khởi tạo HF Hub client.
_SEMANTIC_AVAILABLE = False  # sẽ được override bên dưới sau khi set token

try:
    import pyodbc
    _PYODBC_AVAILABLE = True
except ImportError:
    _PYODBC_AVAILABLE = False

load_dotenv()

# ── Hugging Face token (sentence-transformers dedup model) ───────
# Tránh warning "unauthenticated requests" và rate-limit thấp khi tải model.
_hf_token = os.getenv("HF_TOKEN")
if _hf_token and _hf_token != "hf_your_token_here":
    os.environ["HF_TOKEN"] = _hf_token          # huggingface_hub >= 0.17
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token  # fallback cũ

# Import sentence_transformers SAU KHI token đã được set
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    _SEMANTIC_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_OUTPUT      = OUTPUT_DIR / "raw_samples.jsonl"
PROGRESS_FILE   = OUTPUT_DIR / "progress.json"   # lưu state để resume
VALID_OUTPUT    = OUTPUT_DIR / "valid_samples.jsonl"
TRAIN_OUTPUT    = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT      = OUTPUT_DIR / "val.jsonl"
TEST_OUTPUT     = OUTPUT_DIR / "test.jsonl"

# ── LLM Provider (đọc từ .env, mặc định Groq) ───────────────────
LLM_BASE_URL  = os.getenv("LLM_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL         = os.getenv("LLM_MODEL",     "qwen/qwen3.8-27b")
TEMPERATURE   = 0.7
BATCH_SIZE    = 5                  # số samples mỗi lần gọi API

# ── Rate-limit / quota guard ─────────────────────────────────────
# Đặt số request tối đa mỗi phiên chạy.
# 0 = không giới hạn (chạy đến khi xong hoặc bị 429).
# Ví dụ: quota 200k TPD, mỗi request ~500 tokens → đặt 350 để an toàn.
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "0"))

# Diversity / dedup
DEDUP_SIMILARITY_THRESHOLD = 0.92  # cosine sim >= ngưỡng này → near-duplicate
RECENT_QUESTIONS_WINDOW    = 20    # inject N câu gần nhất vào prompt để tránh lặp

# SQL execution validation (Windows only — cần ODBC Driver 17 for SQL Server)
SQL_CONN_STR  = os.getenv("SQL_CONN_STR", "")  # để trống nếu không dùng --validate-sql

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=LLM_BASE_URL,
)

# ─────────────────────────────────────────────
# COMPACT SCHEMA
# ─────────────────────────────────────────────
COMPACT_SCHEMA = """
TABLE users(id INT PK, organization_id, name, email, invited_by_id INT, created_at)
TABLE organizations(id PK, name, slug)
TABLE artists(id PK, name, user_id INT)
TABLE albums(id PK, name, artist_id, artist_name, year SMALLINT, user_id INT, created_at)
TABLE songs(id PK, album_id, artist_id, title, length FLOAT[seconds], track INT, disc INT,
            is_public BIT, artist_name, album_name, file_size BIGINT, created_at)
TABLE genres(id BIGINT PK, name)
TABLE genre_song(song_id, genre_id BIGINT)  -- pivot songs <-> genres
TABLE playlists(id PK, name, description, created_at)
TABLE playlist_song(id PK, playlist_id, song_id, user_id INT, position INT)
TABLE playlist_user(id PK, user_id INT, playlist_id, role[owner|collaborator], position)
TABLE playlist_folders(id PK, name, user_id INT)
TABLE playlist_playlist_folder(folder_id, playlist_id)  -- pivot folders <-> playlists
TABLE interactions(id PK, user_id INT, song_id, play_count INT, last_played_at)
TABLE favorites(id PK, user_id INT, favoriteable_id, favoriteable_type[playable], position)
TABLE ratings(id PK, user_id INT, rateable_id, rateable_type[song], rating TINYINT[1-5])
TABLE podcasts(id PK, title, author, language, categories, last_synced_at)
TABLE podcast_user(id PK, user_id INT, podcast_id, state)
TABLE radio_stations(id PK, user_id INT, name, url, is_public BIT)
TABLE audits(id PK, user_id BIGINT, event, auditable_type, old_values, new_values, created_at)
TABLE queue_states(id PK, user_id INT, current_song_id, playback_position INT)
TABLE themes(id PK, user_id INT, name)
TABLE transcodes(id PK, song_id, bit_rate INT, file_size BIGINT)

NOTES:
- Dialect: T-SQL (SQL Server). Dùng TOP N thay vì LIMIT. Dùng GETDATE() thay vì NOW().
- songs.length tính bằng giây (seconds). 1 phút = 60 giây.
- favorites.favoriteable_type = 'playable' cho songs.
- ratings.rateable_type = 'song' cho songs.
- playlist_user.role: 'owner' = chủ sở hữu, 'collaborator' = cộng tác viên.
""".strip()

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""Bạn là SQL expert cho hệ thống Koel Music Streaming (SQL Server / T-SQL).

RULES:
1. Chỉ sinh câu lệnh SELECT. Tuyệt đối không sinh INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER.
2. Nếu câu hỏi yêu cầu thay đổi/xóa dữ liệu → trả lời: "Tôi chỉ hỗ trợ truy vấn SELECT. Thao tác này không được phép."
3. Nếu câu hỏi ngoài phạm vi schema → trả lời: "Thông tin này không có trong schema hiện tại."
4. Output: SQL thuần, không có markdown fence (```), không giải thích.
5. Tên bảng và cột phải chính xác theo schema.

SCHEMA:
{COMPACT_SCHEMA}"""

# ─────────────────────────────────────────────
# TIER DEFINITIONS
# (tier_id, count, description, example_questions)
# ─────────────────────────────────────────────
TIERS = [
    {
        "id": "T1",
        "count": 900,   # ↑ từ 600
        "label": "Single table, simple SELECT",
        "examples": [
            "Liệt kê tất cả nghệ sĩ theo thứ tự tên",
            "Có bao nhiêu bài hát trong hệ thống?",
            "Hiển thị danh sách tất cả thể loại nhạc",
            "Những bài hát nào được công khai?",
            "Liệt kê tất cả radio station",
            "Có bao nhiêu album trong database?",
            "Hiển thị thông tin tất cả user",
            "Liệt kê tất cả playlist theo tên",
            "Bài hát nào có file lớn hơn 50MB?",
            "Hiển thị tất cả tổ chức (organization)",
        ],
    },
    {
        "id": "T2",
        "count": 1100,  # ↑ từ 800
        "label": "WHERE, ORDER BY, GROUP BY, Aggregation",
        "examples": [
            "Album nào được phát hành sau năm 2015?",
            "Bài hát dài hơn 5 phút có những bài nào?",
            "Đếm số bài hát theo từng năm phát hành",
            "Top 5 bài hát ngắn nhất",
            "User nào được tạo gần đây nhất?",
            "Thể loại nào có nhiều bài nhất?",
            "Trung bình độ dài bài hát là bao nhiêu giây?",
            "Bài hát nào được phát hành năm 2016?",
            "Có bao nhiêu user trong mỗi organization?",
            "Playlist nào được tạo trong 30 ngày gần đây?",
        ],
    },
    {
        "id": "T3",
        "count": 1000,  # ↑ từ 700
        "label": "2-table JOIN",
        "examples": [
            "Liệt kê tên bài hát cùng tên nghệ sĩ",
            "Mỗi album có bao nhiêu bài hát?",
            "Bài hát nào được nghe nhiều nhất?",
            "User nào có nhiều bài yêu thích nhất?",
            "Playlist nào có nhiều bài nhất?",
            "Nghệ sĩ nào có nhiều album nhất?",
            "Bài hát nào chưa từng được ai nghe?",
            "User nào sở hữu nhiều playlist nhất?",
            "Album nào không có bài hát nào?",
            "Liệt kê bài hát cùng rating trung bình của nó",
        ],
    },
    {
        "id": "T4",
        "count": 800,   # ↑ từ 600
        "label": "Multi-table JOIN (3+ tables)",
        "examples": [
            "Top 10 bài hát được nghe nhiều nhất, kèm tên nghệ sĩ và album",
            "Bài hát thuộc thể loại Synthwave là những bài nào, tên nghệ sĩ là ai?",
            "Playlist của user Alice có những bài hát nào?",
            "Nghệ sĩ nào có tổng lượt nghe cao nhất?",
            "Thể loại nào có tổng lượt nghe cao nhất từ tất cả user?",
            "User nào có điểm rating trung bình cao nhất cho các bài đã đánh giá?",
            "Bài hát trong playlist 'Indie Essentials' thuộc thể loại gì?",
            "Liệt kê các bài hát được yêu thích bởi nhiều hơn 2 user",
        ],
    },
    {
        "id": "T5",
        "count": 600,   # ↑ từ 400
        "label": "Subquery, CTE, HAVING",
        "examples": [
            "Nghệ sĩ nào có trung bình số bài mỗi album cao nhất?",
            "User nào nghe nhiều hơn mức trung bình?",
            "Thể loại nào có tổng lượt nghe vượt 50?",
            "Album nào có tất cả bài dài hơn 3 phút?",
            "Bài hát nào xuất hiện trong nhiều hơn 2 playlist?",
            "Nghệ sĩ nào chưa có bài hát nào được ai nghe?",
            "User nào có nhiều loại favoriteable khác nhau nhất?",
        ],
    },
    {
        "id": "T6",
        "count": 325,   # ↑ từ 200
        "label": "Window Functions, Ranking",
        "examples": [
            "Xếp hạng bài hát theo lượt nghe trong từng thể loại",
            "Tính thứ hạng nghệ sĩ theo số album",
            "Tính tổng lũy kế lượt nghe của user 1 theo thứ tự bài hát",
            "Với mỗi nghệ sĩ, bài hát nào được nghe nhiều nhất (rank 1)?",
            "Xếp hạng album trong từng nghệ sĩ theo số bài hát",
        ],
    },
    {
        "id": "T7",
        "count": 275,   # ↑ từ 200
        "label": "Negative / Refusal cases",
        "examples": [
            "Xóa tất cả bài hát của Radiohead",
            "Cập nhật tên nghệ sĩ thành 'Unknown'",
            "Thêm một album mới vào database",
            "Doanh thu tháng này là bao nhiêu?",
            "DROP TABLE songs",
            "Số lượng đơn hàng hôm nay là bao nhiêu?",
            "Cập nhật mật khẩu của user alice@koel.dev",
            "Xóa playlist 'Synthwave Drive'",
        ],
    },
]
# Tổng: 900+1100+1000+800+600+325+275 = 5000


# ─────────────────────────────────────────────
# GENERATION PROMPT BUILDER
# ─────────────────────────────────────────────
def build_generation_prompt(
    tier: dict,
    n: int = 5,
    recent_questions: list[str] | None = None,
) -> str:
    examples_str = "\n".join(f"  - {e}" for e in tier["examples"])

    # Inject danh sách câu hỏi đã sinh để GPT tránh paraphrase lại
    avoid_block = ""
    if recent_questions:
        avoid_list = "\n".join(f"  - {q}" for q in recent_questions[-RECENT_QUESTIONS_WINDOW:])
        avoid_block = f"""
⚠️ TUYỆT ĐỐI KHÔNG sinh câu hỏi tương tự (về nghĩa) với các câu sau:
{avoid_list}
"""

    return f"""Hãy tạo {n} cặp (câu hỏi tiếng Việt → SQL T-SQL) cho loại: **{tier["label"]}**

Yêu cầu:
- Câu hỏi phải bằng tiếng Việt, tự nhiên, đa dạng (không lặp lại câu hỏi mẫu)
- SQL phải đúng T-SQL syntax, đúng schema đã cho
- Chỉ dùng SELECT (trừ T7 là refusal)
- Câu hỏi có thể dùng ngôn ngữ informal hoặc formal đều được
- Đa dạng cấu trúc câu: không phải câu nào cũng bắt đầu bằng "Liệt kê" hay "Có bao nhiêu"
{avoid_block}
Ví dụ câu hỏi dạng này:
{examples_str}

Trả về JSON array với format:
[
  {{
    "question": "câu hỏi tiếng Việt",
    "sql": "câu lệnh SQL hoặc refusal message"
  }},
  ...
]

Chỉ trả về JSON array, không giải thích gì thêm."""



# ─────────────────────────────────────────────
# RATE-LIMIT EXCEPTION
# ─────────────────────────────────────────────
class RateLimitHit(Exception):
    """Raise khi API trả về 429 — signal để dừng gracefully và lưu progress."""
    def __init__(self, retry_after: float = 0):
        self.retry_after = retry_after  # giây cần chờ (từ header hoặc message)
        super().__init__(f"Rate limit reached. Retry after {retry_after:.0f}s")


# ─────────────────────────────────────────────
# API CALL WITH RETRY
# ─────────────────────────────────────────────
def call_gpt(prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Gọi LLM API.
    - Retry tối đa max_retries lần với exponential backoff.
    - Nếu gặp 429 (rate limit) → raise RateLimitHit ngay, không retry.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "Bạn là data generator chuyên tạo dataset SQL."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            # Detect 429 rate-limit → dừng ngay, không retry
            if "429" in err_str or "rate_limit_exceeded" in err_str or "Rate limit" in err_str:
                # Cố parse retry-after từ message
                retry_after = 0.0
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m(?:in)?)?\s*s(?:econds?)?", err_str)
                if m:
                    retry_after = float(m.group(1))
                raise RateLimitHit(retry_after)
            wait = 2 ** attempt
            print(f"  ⚠ API error (attempt {attempt+1}): {e}. Retry in {wait}s...")
            time.sleep(wait)
    return None


# ─────────────────────────────────────────────
# PARSE GPT RESPONSE
# ─────────────────────────────────────────────
def parse_pairs(raw: str) -> list[dict]:
    # Strip markdown fences if present
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if "question" in d and "sql" in d]
    except json.JSONDecodeError:
        # Try to extract JSON array with regex
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return []


# ─────────────────────────────────────────────
# SAFETY VALIDATOR
# ─────────────────────────────────────────────
DANGER_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)

REFUSAL_KEYWORDS = [
    "không được phép",
    "không hỗ trợ",
    "ngoài phạm vi",
    "chỉ hỗ trợ truy vấn select",
    "không có trong schema",
]

def is_valid_sql(sql: str, tier_id: str) -> bool:
    """Validate SQL sample. T7 samples must be refusal messages."""
    if tier_id == "T7":
        # Refusal cases: must NOT be SQL, must contain refusal keywords
        lower = sql.lower()
        return any(kw in lower for kw in REFUSAL_KEYWORDS)

    # Non-T7: must be SELECT and not contain dangerous keywords
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return False
    if DANGER_PATTERN.search(sql):
        return False
    return True


# ─────────────────────────────────────────────
# BUILD TRAINING SAMPLE
# ─────────────────────────────────────────────
def build_sample(question: str, sql: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": sql},
        ]
    }


# ─────────────────────────────────────────────
# SEMANTIC DEDUPLICATION
# ─────────────────────────────────────────────
def semantic_dedup(samples: list[dict], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list[dict]:
    """
    Loại bỏ near-duplicate dựa trên cosine similarity của question embeddings.
    Exact-string dedup (trong generation loop) chỉ bắt được duplicate 100%,
    còn function này bắt được paraphrase (vd: 'dài hơn 5 phút' vs 'trên 5 phút').

    Yêu cầu: pip install sentence-transformers scikit-learn numpy
    """
    if not _SEMANTIC_AVAILABLE:
        print("⚠️  sentence-transformers không có → bỏ qua semantic dedup. "
              "Chạy: pip install sentence-transformers")
        return samples

    print(f"\n🔍 Semantic dedup (threshold={threshold})...")
    questions = [s["messages"][1]["content"] for s in samples]

    print("   Encoding embeddings...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(questions, show_progress_bar=True, batch_size=64)

    keep_flags = [True] * len(samples)
    removed    = 0

    # O(n²) — acceptable cho dataset < 10k
    sim_matrix = cosine_similarity(embeddings)

    for i in range(len(samples)):
        if not keep_flags[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep_flags[j]:
                continue
            if sim_matrix[i, j] >= threshold:
                keep_flags[j] = False
                removed += 1

    result = [s for s, keep in zip(samples, keep_flags) if keep]
    print(f"   Removed {removed} near-duplicates | {len(result)} samples remaining")
    return result


# ─────────────────────────────────────────────
# OPERATOR COVERAGE REPORT
# ─────────────────────────────────────────────
SQL_OPERATORS_TRACKED = [
    ("JOIN",            r"\bJOIN\b"),
    ("LEFT JOIN",       r"\bLEFT\s+JOIN\b"),
    ("GROUP BY",        r"\bGROUP\s+BY\b"),
    ("HAVING",          r"\bHAVING\b"),
    ("ORDER BY",        r"\bORDER\s+BY\b"),
    ("TOP N",           r"\bTOP\s+\d+\b"),
    ("Subquery",        r"\(\s*SELECT\b"),
    ("CTE (WITH)",      r"^\s*WITH\b"),
    ("Window (OVER)",   r"\bOVER\s*\("),
    ("UNION",           r"\bUNION\b"),
]

def report_operator_coverage(samples: list[dict]) -> None:
    """In bảng phân phối SQL operator. Warn nếu operator nào < 3%."""
    sql_samples = [
        s["messages"][2]["content"]
        for s in samples
        if not any(kw in s["messages"][2]["content"].lower() for kw in REFUSAL_KEYWORDS)
    ]
    total = len(sql_samples)
    if total == 0:
        return

    print("\n📊 SQL Operator Coverage Report")
    print(f"   (total non-refusal samples: {total})")
    print(f"   {'Operator':<20} {'Count':>6}  {'%':>6}  Status")
    print("   " + "-" * 45)

    for name, pattern in SQL_OPERATORS_TRACKED:
        count = sum(
            1 for sql in sql_samples
            if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE)
        )
        pct = count / total * 100
        status = "✅" if pct >= 3.0 else "⚠️  LOW"
        print(f"   {name:<20} {count:>6}  {pct:>5.1f}%  {status}")

    print()


# ─────────────────────────────────────────────
# SQL EXECUTION VALIDATION (optional)
# ─────────────────────────────────────────────
def validate_sql_on_db(sql: str, conn_str: str) -> tuple[bool, str]:
    """
    Chạy SQL với SET NOEXEC ON để kiểm tra syntax + schema mà không thực thi.
    Trả về (True, "") nếu hợp lệ, hoặc (False, error_message).

    Yêu cầu: pip install pyodbc + SQL Server connection string trong .env
    """
    if not _PYODBC_AVAILABLE:
        return True, "pyodbc not installed"
    try:
        with pyodbc.connect(conn_str, timeout=5) as conn:
            cursor = conn.cursor()
            cursor.execute("SET NOEXEC ON")
            cursor.execute(sql)
            cursor.execute("SET NOEXEC OFF")
        return True, ""
    except pyodbc.Error as e:
        return False, str(e)


# ─────────────────────────────────────────────
# PROGRESS: LOAD / SAVE
# ─────────────────────────────────────────────
def load_progress() -> dict:
    """
    Đọc progress.json để resume từ lần chạy trước.
    Trả về dict:
      {
        "completed_tiers": ["T1", "T2"],   # tiers đã hoàn thành
        "current_tier": "T3",              # tier đang chạy dở
        "current_generated": 150,           # số samples đã sinh trong tier hiện tại
        "seen_questions": ["câu 1", ...],   # danh sách câu đã sinh (lowercase)
        "recent_questions": ["câu", ...],   # sliding window
        "db_ok": 0, "db_fail": 0
      }
    """
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            print(f"📂 Resuming from progress: "
                  f"completed={data.get('completed_tiers', [])}, "
                  f"current_tier={data.get('current_tier', '-')}, "
                  f"current_generated={data.get('current_generated', 0)}")
            return data
        except Exception as e:
            print(f"⚠️  Không đọc được progress.json ({e}) → bắt đầu từ đầu.")
    return {
        "completed_tiers": [],
        "current_tier": None,
        "current_generated": 0,
        "seen_questions": [],
        "recent_questions": [],
        "db_ok": 0,
        "db_fail": 0,
    }


def save_progress(progress: dict) -> None:
    """Ghi progress xuống disk sau mỗi sample."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_sample_to_file(sample: dict) -> None:
    """Append một sample vào raw_samples.jsonl (không overwrite)."""
    with open(RAW_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────
def generate_dataset(validate_sql_db: bool = False) -> list[dict]:
    """
    Sinh dataset theo từng tier với khả năng resume sau rate-limit.

    Cơ chế:
    - Load progress từ data/progress.json nếu tồn tại.
    - Sau mỗi sample được lưu: append vào raw_samples.jsonl + cập nhật progress.json.
    - Khi gặp 429 (RateLimitHit) hoặc đạt DAILY_REQUEST_LIMIT:
        → lưu state và dừng gracefully.
    - Lần chạy sau: đọc progress, skip các tier đã xong,
        tiếp tục tier đang dở từ đúng số samples đã có.

    Args:
        validate_sql_db: Nếu True, chạy SET NOEXEC ON trên DB thật để
                         validate SQL syntax + schema (yêu cầu SQL_CONN_STR).
    """
    # ── Load progress ──────────────────────────────────────────────
    progress = load_progress()
    completed_tiers:  list[str] = progress["completed_tiers"]
    seen_questions:   set[str]  = set(progress["seen_questions"])
    recent_questions: list[str] = progress["recent_questions"]
    db_ok   = progress["db_ok"]
    db_fail = progress["db_fail"]
    request_count = 0  # đếm request trong phiên chạy này

    # ── Load existing samples từ file (để trả về sau khi done) ────
    all_samples: list[dict] = []
    if RAW_OUTPUT.exists():
        with open(RAW_OUTPUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"📄 Loaded {len(all_samples)} existing samples from {RAW_OUTPUT}")

    if validate_sql_db and not SQL_CONN_STR:
        print("⚠️  --validate-sql được bật nhưng SQL_CONN_STR chưa cấu hình trong .env → bỏ qua.")
        validate_sql_db = False

    rate_limited = False  # flag: dừng do rate-limit

    for tier in TIERS:
        tier_id = tier["id"]
        target  = tier["count"]
        label   = tier["label"]

        # ── Skip tier đã hoàn thành ────────────────────────────────
        if tier_id in completed_tiers:
            print(f"\n⏭  Tier {tier_id} ({label}): đã hoàn thành, skip.")
            continue

        # ── Tính số samples đã có của tier này ─────────────────────
        generated = sum(1 for s in all_samples if s.get("_tier") == tier_id)
        # Nếu đang ở tier dở dang, lấy số từ progress để chính xác hơn
        if progress["current_tier"] == tier_id:
            generated = max(generated, progress["current_generated"])

        print(f"\n{'='*60}")
        print(f"Tier {tier_id}: {label}")
        print(f"Target: {target} | Already: {generated} | Remaining: {target - generated}")
        print(f"{'='*60}")

        if generated >= target:
            print(f"  ✓ Tier {tier_id} đã đủ samples, đánh dấu hoàn thành.")
            if tier_id not in completed_tiers:
                completed_tiers.append(tier_id)
            progress["completed_tiers"] = completed_tiers
            progress["current_tier"] = None
            progress["current_generated"] = 0
            save_progress(progress)
            continue

        # Cập nhật current tier vào progress
        progress["current_tier"] = tier_id
        progress["current_generated"] = generated
        save_progress(progress)

        pbar = tqdm(total=target, initial=generated, desc=f"  T{tier_id}")

        while generated < target:
            # ── Kiểm tra request limit trước khi gọi API ───────────
            if DAILY_REQUEST_LIMIT > 0 and request_count >= DAILY_REQUEST_LIMIT:
                tqdm.write(f"\n🛑 Đã đạt DAILY_REQUEST_LIMIT ({DAILY_REQUEST_LIMIT} requests). "
                           f"Lưu progress và dừng.")
                rate_limited = True
                break

            remaining = target - generated
            batch_n   = min(BATCH_SIZE, remaining)

            prompt = build_generation_prompt(
                tier,
                n=batch_n,
                recent_questions=recent_questions if recent_questions else None,
            )

            try:
                raw = call_gpt(prompt)
                request_count += 1
            except RateLimitHit as e:
                tqdm.write(f"\n⛔ Rate limit hit! {e}")
                tqdm.write(f"   Đã lưu {generated} samples cho Tier {tier_id}.")
                tqdm.write(f"   Chạy lại script sau khi quota reset để tiếp tục.")
                rate_limited = True
                break

            if raw is None:
                tqdm.write("  ✗ Failed to get response, skipping batch")
                time.sleep(2)
                continue

            pairs = parse_pairs(raw)

            for pair in pairs:
                question = pair["question"].strip()
                sql      = pair["sql"].strip()

                if question.lower() in seen_questions:
                    continue

                if not is_valid_sql(sql, tier_id):
                    continue

                if validate_sql_db and tier_id != "T7":
                    ok, err = validate_sql_on_db(sql, SQL_CONN_STR)
                    if ok:
                        db_ok += 1
                    else:
                        db_fail += 1
                        tqdm.write(f"  ✗ DB validation failed: {err[:80]}")
                        continue

                # ── Lưu sample ngay lập tức ────────────────────────
                seen_questions.add(question.lower())
                recent_questions.append(question)

                sample = build_sample(question, sql)
                sample["_tier"] = tier_id
                all_samples.append(sample)
                append_sample_to_file(sample)  # append vào file

                generated += 1
                pbar.update(1)

                # Cập nhật progress sau mỗi sample
                progress["current_generated"] = generated
                progress["seen_questions"] = list(seen_questions)
                progress["recent_questions"] = recent_questions[-RECENT_QUESTIONS_WINDOW:]
                progress["db_ok"] = db_ok
                progress["db_fail"] = db_fail
                save_progress(progress)

                if generated >= target:
                    break

            time.sleep(0.5)

        pbar.close()

        if rate_limited:
            # Lưu progress và dừng toàn bộ vòng lặp tier
            progress["current_generated"] = generated
            save_progress(progress)
            print(f"  ⏸  Paused tại Tier {tier_id}: {generated}/{target} samples")
            break

        # Tier hoàn thành
        print(f"  ✓ Tier {tier_id} done: {generated}/{target} samples")
        completed_tiers.append(tier_id)
        progress["completed_tiers"] = completed_tiers
        progress["current_tier"] = None
        progress["current_generated"] = 0
        save_progress(progress)

    # ── Summary ───────────────────────────────────────────────────
    report_operator_coverage(all_samples)

    if validate_sql_db:
        print(f"  DB validation — OK: {db_ok} | Failed (dropped): {db_fail}")

    total = len(all_samples)
    if rate_limited:
        print(f"\n⏸  Dừng do rate-limit. Đã lưu {total} samples.")
        print(f"   Progress: {PROGRESS_FILE}")
        print(f"   Chạy lại cùng lệnh để tiếp tục từ chỗ dở.")
    else:
        # Xóa progress file khi hoàn thành toàn bộ
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print(f"\n✅ Dataset hoàn thành! {total} samples → {RAW_OUTPUT}")

    return all_samples


# ─────────────────────────────────────────────
# SPLIT: Train / Val / Test
# ─────────────────────────────────────────────
def split_dataset(samples: list[dict]) -> None:
    # Tách T7 (negative) ra riêng để đảm bảo phân bổ đều trong mỗi split
    t7_samples    = [s for s in samples if s.get("_tier") == "T7"]
    other_samples = [s for s in samples if s.get("_tier") != "T7"]

    random.shuffle(other_samples)
    random.shuffle(t7_samples)

    # 80 / 10 / 10 split
    def split(lst, train_r: float = 0.80, val_r: float = 0.10):
        n       = len(lst)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    o_train, o_val, o_test = split(other_samples)
    t_train, t_val, t_test = split(t7_samples)

    train = o_train + t_train
    val   = o_val   + t_val
    test  = o_test  + t_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Strip internal metadata before saving
    def clean(lst: list[dict]) -> list[dict]:
        return [{k: v for k, v in s.items() if k != "_tier"} for s in lst]

    def save(path: Path, lst: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for s in lst:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {len(lst):5d} samples → {path}")

    print("\n📂 Splitting dataset (80 / 10 / 10)...")
    save(TRAIN_OUTPUT, clean(train))
    save(VAL_OUTPUT,   clean(val))
    save(TEST_OUTPUT,  clean(test))

    # Per-tier breakdown
    print("\n📊 Tier distribution in each split:")
    header = f"  {'Tier':<6} {'Train':>6} {'Val':>6} {'Test':>6}"
    print(header)
    print("  " + "-" * 30)
    all_tiers = [t["id"] for t in TIERS]
    for tid in all_tiers:
        tr = sum(1 for s in train if s.get("_tier") == tid)
        vl = sum(1 for s in val   if s.get("_tier") == tid)
        ts = sum(1 for s in test  if s.get("_tier") == tid)
        print(f"  {tid:<6} {tr:>6} {vl:>6} {ts:>6}")

    print(f"\n✅ Split complete:")
    print(f"   Train : {len(train):5d}")
    print(f"   Val   : {len(val):5d}")
    print(f"   Test  : {len(test):5d}")
    print(f"   Total : {len(train)+len(val)+len(test):5d}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    # Fix Unicode output trên Windows terminal (cp1252 → utf-8)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Text-to-SQL dataset generator")
    parser.add_argument("--generate",     action="store_true", help="Sinh data mới")
    parser.add_argument("--split",        action="store_true", help="Split train/val/test từ raw_samples.jsonl")
    parser.add_argument("--all",          action="store_true", help="Chạy toàn bộ pipeline (generate → dedup → split)")
    parser.add_argument("--validate-sql", action="store_true", help="Validate SQL trên DB thật (yêu cầu SQL_CONN_STR trong .env)")
    parser.add_argument("--dedup-only",   action="store_true", help="Chỉ chạy semantic dedup trên raw_samples.jsonl có sẵn")
    args = parser.parse_args()

    samples: list[dict] = []

    # ── Step 1: Generate ──────────────────────
    if args.all or args.generate:
        samples = generate_dataset(validate_sql_db=args.validate_sql)

    # ── Step 2: Load existing raw (nếu không generate) ────
    if (args.split or args.dedup_only) and not samples:
        with open(RAW_OUTPUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} samples from {RAW_OUTPUT}")
        # Khôi phục _tier từ messages nếu cần (cho coverage report)
        report_operator_coverage(samples)

    # ── Step 3: Semantic dedup ────────────────
    if args.all or args.dedup_only:
        samples = semantic_dedup(samples)
        # Lưu lại file raw đã dedup
        with open(RAW_OUTPUT, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"✅ Deduped dataset saved → {RAW_OUTPUT} ({len(samples)} samples)")

    # ── Step 4: Split ─────────────────────────
    if args.all or args.split:
        split_dataset(samples)

    if not any([args.generate, args.split, args.all, args.dedup_only]):
        print("Usage:")
        print("  python generate_data.py --all                  # Sinh + dedup + split")
        print("  python generate_data.py --all --validate-sql   # Như trên + validate SQL trên DB")
        print("  python generate_data.py --generate             # Chỉ sinh data")
        print("  python generate_data.py --dedup-only           # Chỉ dedup raw có sẵn")
        print("  python generate_data.py --split                # Chỉ split từ raw có sẵn")

