"""
generate_data.py
================
Generate a Text-to-SQL dataset for the Koel Music Streaming DB.
Output: data/raw_samples.jsonl

Basic requirements:
    pip install openai tqdm python-dotenv

Optional features:
    pip install sentence-transformers     # semantic dedup
    pip install pyodbc                    # SQL execution validation (--validate-sql, Windows only)

Configuration (.env):
    OPENAI_API_KEY=gsk_...               # Groq API key (get it at console.groq.com)
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=qwen/qwen3.8-27b
    SQL_CONN_STR=Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=music_14_09_2026;UID=sa;PWD=Aa123456@;

    # Maximum requests per session (stop proactively before hitting rate-limit):
    DAILY_REQUEST_LIMIT=0   # 0 = unlimited; set e.g. 800 if daily quota is 1000 req

Resume after rate-limit:
    The script saves progress to data/progress.json after every sample.
    On 429 or when DAILY_REQUEST_LIMIT is reached, it saves state and stops gracefully.
    Re-run the same command -> resumes from where it left off, no data re-generated.
    Delete data/progress.json to start over from scratch.

--validate-sql only works on Windows (requires ODBC Driver 17 for SQL Server, installed via SSMS).
"""

import json
import logging
import logging.handlers
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

# ── Optional dependencies (sentence_transformers imported AFTER HF_TOKEN is set) ──
# Import is delayed until after load_dotenv() so HF_TOKEN is available before
# sentence_transformers initializes the HF Hub client.
_SEMANTIC_AVAILABLE = False  # will be overridden below after token is set

try:
    import pyodbc
    _PYODBC_AVAILABLE = True
except ImportError:
    _PYODBC_AVAILABLE = False

load_dotenv()

# ── Hugging Face token (sentence-transformers dedup model) ───────
# Avoids "unauthenticated requests" warnings and low rate-limits when downloading the model.
_hf_token = os.getenv("HF_TOKEN")
if _hf_token and _hf_token != "hf_your_token_here":
    os.environ["HF_TOKEN"] = _hf_token          # huggingface_hub >= 0.17
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token  # legacy fallback

# Import sentence_transformers AFTER token has been set
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
PROGRESS_FILE   = OUTPUT_DIR / "progress.json"   # saves state for resume
VALID_OUTPUT    = OUTPUT_DIR / "valid_samples.jsonl"
TRAIN_OUTPUT    = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT      = OUTPUT_DIR / "val.jsonl"
TEST_OUTPUT     = OUTPUT_DIR / "test.jsonl"

# ── LLM Provider (read from .env, defaults to Groq) ───────────────────
LLM_BASE_URL  = os.getenv("LLM_BASE_URL",  "https://api.groq.com/openai/v1")
MODEL         = os.getenv("LLM_MODEL",     "qwen/qwen3.8-27b")
TEMPERATURE   = 0.7
BATCH_SIZE    = 5                  # number of samples per API call

# ── Rate-limit / quota guard ─────────────────────────────────────
# Set the maximum number of requests per session.
# 0 = unlimited (run until done or until a 429 is received).
# Example: quota 200k TPD, ~500 tokens per request -> set 350 to be safe.
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "0"))

# ── OTPM (Output Tokens Per Minute) budget ───────────────────────
# Groq free tier: 1000 OTPM for qwen/qwen3.8-27b.
# min_interval is auto-computed per-tier based on tokens_per_sample * batch_size.
# Override with MIN_REQUEST_INTERVAL to force a fixed floor.
OTPM_LIMIT           = int(os.getenv("OTPM_LIMIT", "1000"))   # tokens per minute
OTPM_SAFETY_FACTOR   = float(os.getenv("OTPM_SAFETY_FACTOR", "1.3"))  # 30% headroom
MIN_REQUEST_INTERVAL = float(os.getenv("MIN_REQUEST_INTERVAL", "2.5"))  # hard floor (seconds)
_last_request_time: float = 0.0  # timestamp of the most recent API call


def compute_min_interval(batch_size: int, tokens_per_sample: int) -> float:
    """
    Compute the minimum seconds between requests to stay within OTPM_LIMIT.

    Formula:
        tokens_per_request = batch_size * tokens_per_sample
        requests_per_min   = OTPM_LIMIT / tokens_per_request   (with safety factor)
        min_interval (s)   = 60 / requests_per_min
    """
    tokens_per_request = batch_size * tokens_per_sample
    safe_otpm = OTPM_LIMIT / OTPM_SAFETY_FACTOR
    requests_per_min = safe_otpm / tokens_per_request
    computed = 60.0 / requests_per_min
    return max(computed, MIN_REQUEST_INTERVAL)


# Diversity / dedup
DEDUP_SIMILARITY_THRESHOLD = 0.92  # cosine sim >= this threshold -> near-duplicate
RECENT_QUESTIONS_WINDOW    = 20    # inject N recent questions into prompt to avoid repetition

# SQL execution validation (Windows only — requires ODBC Driver 17 for SQL Server)
SQL_CONN_STR  = os.getenv("SQL_CONN_STR", "")  # leave empty if not using --validate-sql

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=LLM_BASE_URL,
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
LOG_FILE = os.getenv("LOG_FILE", str(OUTPUT_DIR / "run.log"))

_log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler: rotates at 5 MB, keeps 3 backups
_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_log_formatter)

# Console handler (INFO+)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
_console_handler.setLevel(logging.WARNING)  # only warnings+ to stdout (tqdm handles info)

logger = logging.getLogger("text2sql")
logger.setLevel(logging.DEBUG)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)
logger.propagate = False

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
- Dialect: T-SQL (SQL Server). Use TOP N instead of LIMIT. Use GETDATE() instead of NOW().
- songs.length is in seconds. 1 minute = 60 seconds.
- favorites.favoriteable_type = 'playable' for songs.
- ratings.rateable_type = 'song' for songs.
- playlist_user.role: 'owner' = playlist owner, 'collaborator' = collaborator.
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
        "count": 900,
        "label": "Single table, simple SELECT",
        # Short SQL: ~60 tokens/sample. batch=5 -> 300 tokens/req.
        # compute_min_interval(5, 60) -> ~12s, but floor=2.5s is fine here.
        "batch_size": 5,
        "tokens_per_sample": 60,   # estimated average output tokens per sample
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
        "count": 1100,
        "label": "WHERE, ORDER BY, GROUP BY, Aggregation",
        # Moderate SQL: ~100 tokens/sample. batch=5 -> 500 tokens/req -> 2 req/min safe.
        "batch_size": 5,
        "tokens_per_sample": 100,
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
        "count": 1000,
        "label": "2-table JOIN",
        # 2-table JOIN: ~130 tokens/sample. batch=4 -> 520 tokens/req.
        # compute_min_interval(4, 130) -> ~38s. Safe.
        "batch_size": 4,
        "tokens_per_sample": 130,
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
        "count": 800,
        "label": "Multi-table JOIN (3+ tables)",
        # Complex SQL: ~200 tokens/sample. batch=2 -> 400 tokens/req.
        # compute_min_interval(2, 200) -> ~31s.
        "batch_size": 2,
        "tokens_per_sample": 200,
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
        "count": 600,
        "label": "Subquery, CTE, HAVING",
        # Subquery/CTE: ~220 tokens/sample. batch=2 -> 440 tokens/req.
        # compute_min_interval(2, 220) -> ~34s.
        # Previously batch=3 -> 660 tokens/req -> burst > 1000 OTPM -> FIX: batch=2
        "batch_size": 2,
        "tokens_per_sample": 220,
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
        "count": 325,
        "label": "Window Functions, Ranking",
        # Window functions: ~250 tokens/sample. batch=2 -> 500 tokens/req.
        # compute_min_interval(2, 250) -> ~39s.
        "batch_size": 2,
        "tokens_per_sample": 250,
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
        "count": 275,
        "label": "Negative / Refusal cases",
        # Refusal: ~25 tokens/sample. batch=5 -> 125 tokens/req -> very safe.
        "batch_size": 5,
        "tokens_per_sample": 25,
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
# Total: 900+1100+1000+800+600+325+275 = 5000


# ─────────────────────────────────────────────
# GENERATION PROMPT BUILDER
# ─────────────────────────────────────────────
def build_generation_prompt(
    tier: dict,
    n: int = 5,
    recent_questions: list[str] | None = None,
) -> str:
    examples_str = "\n".join(f"  - {e}" for e in tier["examples"])

    # Inject list of already-generated questions so the LLM avoids paraphrasing them
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
    """Raised when the API returns 429 (OTPM/RPM) repeatedly beyond the allowed retry count."""
    def __init__(self, retry_after: float = 0):
        self.retry_after = retry_after
        super().__init__(f"Rate limit reached. Retry after {retry_after:.0f}s")


class RequestTooLarge(Exception):
    """Raised when the API returns 429 with 'Request too large' — prompt exceeds context limit."""
    pass


def _parse_retry_after(err_str: str) -> float:
    """
    Parse the retry wait time from a Groq error message.

    Groq returns formats like:
      - "Please try again in 2m30s"  -> 150.0
      - "try again in 45.5s"          -> 45.5
      - "retry after 60 seconds"      -> 60.0
      - "try again in 300ms"          ->  0.0  (treat sub-second as 'no wait')
      - "rate_limit_exceeded"         ->  0.0  (fallback)

    IMPORTANT: milliseconds (ms) must be matched BEFORE the minutes regex,
    otherwise '300ms' is misread as '300 minutes' (18 000 s).
    """
    # ── Milliseconds: "300ms", "500 ms" ─────────────────────────────
    # Return 0.0 so the caller falls back to the safe tier_interval.
    if re.search(r"\d+\s*ms\b", err_str, re.IGNORECASE):
        return 0.0

    # ── Combined "XmYs" / "Xm Ys" (e.g. "2m30s", "1m 5.2s") ────────
    m = re.search(r"(\d+)\s*m(?:in)?\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))

    # ── Minutes only: "2m", "2min", "2mins" (NOT followed by 's' that would make 'ms') ──
    m = re.search(r"(\d+(?:\.\d+)?)\s*min(?:ute)?s?\b", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60

    # ── Seconds only: "45s", "45 seconds", "45.5s" ─────────────────
    m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return 0.0


# ─────────────────────────────────────────────
# API CALL WITH RETRY
# ─────────────────────────────────────────────
def call_gpt(
    prompt: str,
    min_interval: float = MIN_REQUEST_INTERVAL,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Call the LLM API with proactive throttling and smart retry on 429.

    Strategy:
    - Enforce `min_interval` seconds since the LAST request before sending.
      (Caller computes the correct interval per tier via compute_min_interval.)
    - On 429: wait exactly retry-after seconds, then enforce min_interval again.
    - If retry_after = 0 (parse failed): exponential backoff 10s → 20s → 40s.
    - After exhausting retries on 429 → raise RateLimitHit (graceful stop).
    - Other errors (5xx, timeout, network): short exponential backoff then retry.

    NOTE: Throttle is handled HERE only. Callers must NOT sleep before calling
    this function — doing so would double-count the wait time.
    """
    global _last_request_time

    # ── Enforce minimum gap since last request ──────────────────────
    elapsed = time.monotonic() - _last_request_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.debug(f"Throttle: sleeping {sleep_time:.1f}s (interval={min_interval:.1f}s)")
        time.sleep(sleep_time)

    rate_limit_attempts = 0
    max_rate_limit_retries = 5

    for attempt in range(max_retries):
        try:
            _last_request_time = time.monotonic()
            logger.debug(f"API call attempt {attempt + 1}/{max_retries}")
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "Bạn là data generator chuyên tạo dataset SQL."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content

            # Log token usage if available
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    f"API OK | prompt_tokens={usage.prompt_tokens} "
                    f"output_tokens={usage.completion_tokens} "
                    f"total={usage.total_tokens}"
                )
            else:
                logger.info("API OK | (usage info not available)")

            return content

        except Exception as e:
            err_str = str(e)

            # ── 429: Request too large (prompt exceeds context window) ───────
            # Groq returns 429 with "Request too large" when the INPUT tokens are
            # over the model limit. Retrying with the same prompt won't help.
            # Signal the caller to reduce the prompt (trim recent_questions).
            if ("429" in err_str or "rate_limit_exceeded" in err_str) and (
                "request too large" in err_str.lower() or "context_length_exceeded" in err_str.lower()
            ):
                logger.warning(f"Request too large — prompt too long. Signalling trim. | {err_str[:160]}")
                raise RequestTooLarge(err_str)

            # ── 429 OTPM / RPM rate-limit ─────────────────────────────────
            if "429" in err_str or "rate_limit_exceeded" in err_str or "rate limit" in err_str.lower():
                rate_limit_attempts += 1
                retry_after = _parse_retry_after(err_str)

                if rate_limit_attempts >= max_rate_limit_retries:
                    logger.error(
                        f"429 exhausted after {max_rate_limit_retries} retries. "
                        f"Raising RateLimitHit. | raw: {err_str[:200]}"
                    )
                    raise RateLimitHit(retry_after)

                # retry_after=0 -> use exponential backoff capped at min_interval
                wait = retry_after if retry_after > 0 else (10 * (2 ** (rate_limit_attempts - 1)))
                wait_total = max(wait, min_interval)
                msg = (
                    f"429 rate_limit (attempt {rate_limit_attempts}/{max_rate_limit_retries}): "
                    f"retry_after={retry_after:.1f}s | waiting {wait_total:.0f}s"
                )
                logger.warning(msg)
                tqdm.write(f"\n⚠ {msg} | raw: {err_str[:120]}")
                time.sleep(wait_total)
                _last_request_time = time.monotonic()
                continue

            # ── Other errors (5xx, timeout, network) ─────────────────────
            wait = 2 ** attempt
            msg = f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retry in {wait}s..."
            logger.warning(msg)
            tqdm.write(f"\n⚠ {msg}")
            time.sleep(wait)

    logger.error("call_gpt: exhausted all retries, returning None")
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
    Remove near-duplicates based on cosine similarity of question embeddings.
    Exact-string dedup (in the generation loop) only catches 100% duplicates;
    this function catches paraphrases (e.g. 'longer than 5 minutes' vs 'over 5 minutes').

    Requires: pip install sentence-transformers scikit-learn numpy
    """
    if not _SEMANTIC_AVAILABLE:
        print("⚠️  sentence-transformers not available -> skipping semantic dedup. "
              "Run: pip install sentence-transformers")
        return samples

    print(f"\n🔍 Semantic dedup (threshold={threshold})...")
    questions = [s["messages"][1]["content"] for s in samples]

    print("   Encoding embeddings...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(questions, show_progress_bar=True, batch_size=64)

    keep_flags = [True] * len(samples)
    removed    = 0

    # O(n²) — acceptable for datasets < 10k
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
    """Print SQL operator distribution table. Warn if any operator is < 3%."""
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
    Execute SQL with SET NOEXEC ON to check syntax and schema without actually running it.
    Returns (True, "") if valid, or (False, error_message) otherwise.

    Requires: pip install pyodbc + SQL Server connection string in .env
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
    Read progress.json to resume from a previous run.
    Returns a dict:
      {
        "completed_tiers": ["T1", "T2"],   # tiers that have been completed
        "current_tier": "T3",              # tier currently in progress
        "current_generated": 150,           # samples generated in the current tier
        "seen_questions": ["q1", ...],      # list of all generated questions (lowercase)
        "recent_questions": ["q", ...],     # sliding window
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
            print(f"⚠️  Could not read progress.json ({e}) -> starting from scratch.")
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
    """Write progress to disk after every sample."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_sample_to_file(sample: dict) -> None:
    """Append a single sample to raw_samples.jsonl (non-destructive)."""
    with open(RAW_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────
def generate_dataset(validate_sql_db: bool = False) -> list[dict]:
    """
    Generate the dataset tier by tier with resume support after rate-limiting.

    Mechanism:
    - Load progress from data/progress.json if it exists.
    - After each saved sample: append to raw_samples.jsonl and update progress.json.
    - On 429 (RateLimitHit) or when DAILY_REQUEST_LIMIT is reached:
        -> save state and stop gracefully.
    - On the next run: read progress, skip completed tiers,
        and continue the in-progress tier from exactly where it left off.

    Args:
        validate_sql_db: If True, run SET NOEXEC ON against the real DB to
                         validate SQL syntax + schema (requires SQL_CONN_STR).
    """
    # ── Load progress ──────────────────────────────────────────────

    progress = load_progress()
    completed_tiers:  list[str] = progress["completed_tiers"]
    seen_questions:   set[str]  = set(progress["seen_questions"])
    recent_questions: list[str] = progress["recent_questions"]
    db_ok   = progress["db_ok"]
    db_fail = progress["db_fail"]
    request_count = 0  # count requests in this session

    # ── Load existing samples from file (to return once done) ────
    all_samples: list[dict] = []
    if RAW_OUTPUT.exists():
        with open(RAW_OUTPUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        print(f"📄 Loaded {len(all_samples)} existing samples from {RAW_OUTPUT}")

    if validate_sql_db and not SQL_CONN_STR:
        print("⚠️  --validate-sql is enabled but SQL_CONN_STR is not configured in .env -> skipping.")
        validate_sql_db = False

    rate_limited = False  # flag: stopped due to rate-limiting

    for tier in TIERS:
        tier_id = tier["id"]
        target  = tier["count"]
        label   = tier["label"]

        # ── Skip completed tiers ────────────────────────────────
        if tier_id in completed_tiers:
            print(f"\n⏭  Tier {tier_id} ({label}): already completed, skipping.")
            logger.info(f"[{tier_id}] Skipping — already completed")
            continue

        # ── Count samples already generated for this tier ─────────────────────
        generated = sum(1 for s in all_samples if s.get("_tier") == tier_id)
        # If this is the in-progress tier, prefer the count from progress for accuracy
        if progress["current_tier"] == tier_id:
            generated = max(generated, progress["current_generated"])

        # ── Compute safe min_interval based on OTPM budget ────────────────
        tier_batch          = tier.get("batch_size", BATCH_SIZE)
        tokens_per_sample   = tier.get("tokens_per_sample", 150)
        tier_interval       = compute_min_interval(tier_batch, tokens_per_sample)

        print(f"\n{'='*60}")
        print(f"Tier {tier_id}: {label}")
        print(f"Target: {target} | Already: {generated} | Remaining: {target - generated}")
        print(f"batch_size={tier_batch} | tokens_per_sample~{tokens_per_sample} | "
              f"min_interval={tier_interval:.1f}s (OTPM_LIMIT={OTPM_LIMIT})")
        print(f"{'='*60}")
        logger.info(
            f"[{tier_id}] Starting | target={target} already={generated} "
            f"batch={tier_batch} tokens_per_sample={tokens_per_sample} "
            f"min_interval={tier_interval:.1f}s"
        )

        if generated >= target:
            print(f"  ✓ Tier {tier_id} already has enough samples, marking as complete.")
            logger.info(f"[{tier_id}] Already complete — marking done")
            if tier_id not in completed_tiers:
                completed_tiers.append(tier_id)
            progress["completed_tiers"] = completed_tiers
            progress["current_tier"] = None
            progress["current_generated"] = 0
            save_progress(progress)
            continue

        # Update current tier in progress
        progress["current_tier"] = tier_id
        progress["current_generated"] = generated
        save_progress(progress)

        pbar = tqdm(total=target, initial=generated, desc=f"  T{tier_id}")

        while generated < target:
            # ── Check request limit before calling the API ───────────
            if DAILY_REQUEST_LIMIT > 0 and request_count >= DAILY_REQUEST_LIMIT:
                msg = f"Reached DAILY_REQUEST_LIMIT ({DAILY_REQUEST_LIMIT} requests). Stopping."
                tqdm.write(f"\n🛑 {msg}")
                logger.warning(f"[{tier_id}] {msg}")
                rate_limited = True
                break

            remaining = target - generated
            batch_n   = min(tier_batch, remaining)

            prompt = build_generation_prompt(
                tier,
                n=batch_n,
                recent_questions=recent_questions if recent_questions else None,
            )

            logger.info(
                f"[{tier_id}] Request #{request_count + 1} | "
                f"batch={batch_n} | generated={generated}/{target} | "
                f"interval={tier_interval:.1f}s"
            )

            try:
                # Single throttle point: delegate entirely to call_gpt(min_interval)
                # Do NOT sleep here — call_gpt() handles it.
                raw = call_gpt(prompt, min_interval=tier_interval)
                request_count += 1
            except RequestTooLarge:
                # Prompt too long: trim the recent_questions avoid-block and retry immediately.
                old_len = len(recent_questions)
                recent_questions = recent_questions[len(recent_questions) // 2:]  # drop oldest half
                msg = (
                    f"Prompt too large \u2014 trimming recent_questions "
                    f"{old_len} \u2192 {len(recent_questions)} and retrying"
                )
                tqdm.write(f"\n\u26a0 {msg}")
                logger.warning(f"[{tier_id}] {msg}")
                continue  # rebuild prompt with shorter avoid-block, no extra sleep
            except RateLimitHit as e:
                msg = f"Rate limit hit after retries: {e}"
                tqdm.write(f"\n\u26d4 {msg}")
                logger.error(f"[{tier_id}] {msg} | saved={generated}/{target}")
                tqdm.write(f"   Saved {generated} samples for Tier {tier_id}.")
                tqdm.write(f"   Re-run the script after quota resets to continue.")
                rate_limited = True
                break

            if raw is None:
                tqdm.write("  \u2717 Failed to get response, skipping batch")
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

                # ── Save sample immediately ────────────────────────
                seen_questions.add(question.lower())
                recent_questions.append(question)

                sample = build_sample(question, sql)
                sample["_tier"] = tier_id
                all_samples.append(sample)
                append_sample_to_file(sample)  # append to file

                generated += 1
                pbar.update(1)
                logger.debug(f"[{tier_id}] Saved sample {generated}/{target}: {question[:60]}")

                # Update progress after every sample
                progress["current_generated"] = generated
                progress["seen_questions"] = list(seen_questions)
                progress["recent_questions"] = recent_questions[-RECENT_QUESTIONS_WINDOW:]
                progress["db_ok"] = db_ok
                progress["db_fail"] = db_fail
                save_progress(progress)

                if generated >= target:
                    break

            # No extra sleep — call_gpt() enforces tier_interval

        pbar.close()

        if rate_limited:
            # Save progress and exit the tier loop
            progress["current_generated"] = generated
            save_progress(progress)
            print(f"  ⏸  Paused at Tier {tier_id}: {generated}/{target} samples")
            logger.warning(f"[{tier_id}] Paused due to rate-limit: {generated}/{target} saved")
            break

        # Tier complete
        print(f"  ✓ Tier {tier_id} done: {generated}/{target} samples")
        logger.info(f"[{tier_id}] Complete: {generated}/{target} samples")
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
        print(f"\n⏸  Stopped due to rate-limit. Saved {total} samples.")
        print(f"   Progress: {PROGRESS_FILE}")
        print(f"   Re-run the same command to continue from where it left off.")
    else:
        # Delete progress file on full completion
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print(f"\n✅ Dataset complete! {total} samples -> {RAW_OUTPUT}")

    return all_samples


# ─────────────────────────────────────────────
# SPLIT: Train / Val / Test
# ─────────────────────────────────────────────
def split_dataset(samples: list[dict]) -> None:
    # Separate T7 (negative) samples to ensure even distribution across splits
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

    # Fix Unicode output on Windows terminal (cp1252 -> utf-8)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Text-to-SQL dataset generator")
    parser.add_argument("--generate",     action="store_true", help="Generate new data")
    parser.add_argument("--split",        action="store_true", help="Split train/val/test from raw_samples.jsonl")
    parser.add_argument("--all",          action="store_true", help="Run full pipeline (generate -> dedup -> split)")
    parser.add_argument("--validate-sql", action="store_true", help="Validate SQL against real DB (requires SQL_CONN_STR in .env)")
    parser.add_argument("--dedup-only",   action="store_true", help="Run semantic dedup only on existing raw_samples.jsonl")
    args = parser.parse_args()

    samples: list[dict] = []

    # ── Step 1: Generate ──────────────────────
    if args.all or args.generate:
        samples = generate_dataset(validate_sql_db=args.validate_sql)

    # ── Step 2: Load existing raw (if not generating) ────
    if (args.split or args.dedup_only) and not samples:
        with open(RAW_OUTPUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"Loaded {len(samples)} samples from {RAW_OUTPUT}")
        # Restore _tier from messages if needed (for coverage report)
        report_operator_coverage(samples)

    # ── Step 3: Semantic dedup ────────────────
    if args.all or args.dedup_only:
        samples = semantic_dedup(samples)
        # Save deduped raw file
        with open(RAW_OUTPUT, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"✅ Deduped dataset saved -> {RAW_OUTPUT} ({len(samples)} samples)")

    # ── Step 4: Split ─────────────────────────
    if args.all or args.split:
        split_dataset(samples)

    if not any([args.generate, args.split, args.all, args.dedup_only]):
        print("Usage:")
        print("  python generate_data.py --all                  # Generate + dedup + split")
        print("  python generate_data.py --all --validate-sql   # Same + validate SQL against DB")
        print("  python generate_data.py --generate             # Generate data only")
        print("  python generate_data.py --dedup-only           # Run dedup on existing raw")
        print("  python generate_data.py --split                # Split from existing raw")

