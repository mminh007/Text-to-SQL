"""
src/inference/pipeline.py
==========================
End-to-end Text-to-SQL inference pipeline với Self-Correction loop.

Components:
    1. SchemaInjector     — build system prompt (full hoặc dynamic schema via SchemaLinker)
    2. SQLGenerator       — call fine-tuned Qwen model (local) để generate SQL
    3. SafetyValidator    — syntax + dangerous-keyword check
    4. QueryExecutor      — execute SQL on SQL Server, trả về rows
    5. SelfCorrector      — nếu SQL lỗi, gửi error message lại model để sửa (multi-turn)
    6. ResultFormatter    — format raw rows cho API response

Usage (standalone CLI):
    python -m src.inference.pipeline \\
        --question "Top 10 bài hát được nghe nhiều nhất?" \\
        --model-path outputs/qwen25-7b-koel-sql/merged

Usage (as library):
    from src.inference.pipeline import InferencePipeline

    pipe = InferencePipeline(model_path="outputs/qwen25-7b-koel-sql/merged")
    result = pipe.run("Bài hát dài nhất là bài nào?")
    print(result.sql)
    print(result.rows)

Requirements:
    pip install transformers torch unsloth pyodbc python-dotenv
    SQL_CONN_STR in .env for live DB execution.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SQL_CONN_STR    = os.getenv("SQL_CONN_STR", "")
MAX_RETRIES     = int(os.getenv("SELF_CORRECTION_MAX_RETRIES", "2"))  # max correction attempts
FETCH_LIMIT     = int(os.getenv("RESULT_FETCH_LIMIT", "50"))          # max rows to return
GENERATE_TIMEOUT = int(os.getenv("GENERATE_TIMEOUT_S", "60"))         # model inference timeout (s)

# Schema linker feature flag
USE_SCHEMA_LINKER = os.getenv("USE_SCHEMA_LINKER", "false").lower() == "true"
SCHEMA_LINKER_TOP_K = int(os.getenv("SCHEMA_LINKER_TOP_K", "8"))

# System prompt path (fallback to inline default)
_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "system_prompt.txt"

# Dangerous SQL patterns to block before DB execution
_DANGER_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE)\b",
    re.IGNORECASE,
)

_REFUSAL_KEYWORDS = [
    "không được phép",
    "không hỗ trợ",
    "ngoài phạm vi",
    "chỉ hỗ trợ truy vấn select",
    "không có trong schema",
]


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Result from the full inference pipeline."""
    question:   str
    sql:        str                    # final SQL (after any corrections)
    rows:       list[dict]             # query result rows (list of column→value dicts)
    success:    bool                   # True if SQL executed without error
    error:      str = ""               # DB error message if success=False
    is_refusal: bool = False           # True if model refused (non-SQL answer)
    attempts:   int = 1                # total generation attempts (1 = no correction needed)
    correction_history: list[dict] = field(default_factory=list)
    # Each entry: {"attempt": N, "sql": "...", "error": "..."}


# ─────────────────────────────────────────────
# 1. SCHEMA INJECTOR
# ─────────────────────────────────────────────

class SchemaInjector:
    """
    Build the system prompt with schema.

    Modes:
    - Full schema   : always inject all 22 tables (safe default)
    - Dynamic schema: use HybridLinker to select relevant tables only
                      (requires USE_SCHEMA_LINKER=true + rank-bm25 or sentence-transformers)
    """

    def __init__(self, use_dynamic: bool = USE_SCHEMA_LINKER) -> None:
        self._use_dynamic = use_dynamic
        self._linker = None

        if use_dynamic:
            try:
                from src.data.schema_linker import HybridLinker
                self._linker = HybridLinker()
                print("✅ Schema linker enabled (BM25 + Semantic hybrid)")
            except (ImportError, RuntimeError) as e:
                print(f"⚠  Schema linker unavailable ({e}) — using full schema.")
                self._use_dynamic = False

    def get_system_prompt(self, question: str) -> str:
        """Return system prompt with schema injected for the given question."""
        base = _load_base_prompt()

        if self._use_dynamic and self._linker is not None:
            from src.data.schema_linker import build_dynamic_schema, build_full_schema
            selected, confidence = self._linker.select(question, top_k=SCHEMA_LINKER_TOP_K)
            if self._linker.should_fallback(confidence):
                schema_str = build_full_schema()
            else:
                schema_str = build_dynamic_schema(selected)
            # Replace SCHEMA section in base prompt
            return _inject_schema(base, schema_str)

        # Full schema — use prompt file as-is (already contains full schema)
        return base


def _load_base_prompt() -> str:
    """Load system prompt from file, or return the embedded default."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    # Fallback minimal inline prompt
    return textwrap.dedent("""
        Bạn là SQL expert cho hệ thống Koel music streaming (SQL Server / T-SQL).
        RULES:
        1. Chỉ sinh câu lệnh SELECT.
        2. Output: SQL thuần, không có markdown fence.
    """).strip()


def _inject_schema(base_prompt: str, schema_str: str) -> str:
    """Replace the SCHEMA: section in base_prompt with schema_str."""
    if "SCHEMA:" in base_prompt:
        # Replace everything after SCHEMA: until end-of-string (or a sentinel)
        return re.sub(r"SCHEMA:.*", f"SCHEMA:\n{schema_str}", base_prompt, flags=re.DOTALL)
    return base_prompt + f"\n\nSCHEMA:\n{schema_str}"


# ─────────────────────────────────────────────
# 2. SQL GENERATOR
# ─────────────────────────────────────────────

class SQLGenerator:
    """
    Generate SQL from a conversation using a local fine-tuned Qwen model.

    First call triggers model loading (~10–30s depending on hardware).
    Subsequent calls reuse the loaded model.

    Args:
        model_path : path to merged or adapter model directory.
        use_unsloth: if True, load via unsloth.FastLanguageModel (2× faster inference).
                     Falls back to plain transformers if unsloth is unavailable.
    """

    def __init__(self, model_path: str, use_unsloth: bool = True) -> None:
        self.model_path  = model_path
        self.use_unsloth = use_unsloth
        self._model      = None
        self._tokenizer  = None
        self._pipeline   = None

    def _load(self) -> None:
        """Lazy-load model on first use."""
        if self._pipeline is not None:
            return

        print(f"🔄 Loading model from {self.model_path} …")
        if self.use_unsloth:
            try:
                from unsloth import FastLanguageModel
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=4096,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(model)
                self._model     = model
                self._tokenizer = tokenizer
                print("✅ Loaded via unsloth (fast inference mode)")
                return
            except ImportError:
                print("⚠  unsloth not available, falling back to transformers")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model     = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("✅ Loaded via transformers")

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """
        Generate SQL given a list of chat messages.

        Args:
            messages      : OpenAI-style messages list
            max_new_tokens: max tokens to generate
        Returns:
            Generated text (SQL or refusal)
        """
        self._load()

        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy for SQL (deterministic)
                temperature=1.0,          # ignored when do_sample=False
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Strip <think>...</think> from output for the returned SQL
        # (The think block is preserved internally in messages for multi-turn context)
        return _strip_think_tags(raw)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> block and return only the SQL part."""
    # Remove the think block entirely
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove markdown fences
    cleaned = re.sub(r"```sql\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*",    "", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────
# 3. SAFETY VALIDATOR
# ─────────────────────────────────────────────

class SafetyValidator:
    """
    Block dangerous SQL before it reaches the database.
    Returns (is_safe, error_reason).
    """

    @staticmethod
    def validate(sql: str) -> tuple[bool, str]:
        lower = sql.lower().strip()

        # Check if it's a refusal message (non-SQL)
        if any(kw in lower for kw in _REFUSAL_KEYWORDS):
            return True, ""   # refusal is safe — caller will detect is_refusal

        # Must start with SELECT or WITH (CTE)
        if not (lower.startswith("select") or lower.startswith("with")):
            return False, f"SQL must start with SELECT or WITH, got: {sql[:40]!r}"

        # Block dangerous keywords
        m = _DANGER_RE.search(sql)
        if m:
            return False, f"Dangerous keyword detected: {m.group()!r}"

        return True, ""


# ─────────────────────────────────────────────
# 4. QUERY EXECUTOR
# ─────────────────────────────────────────────

class QueryExecutor:
    """
    Execute SQL on SQL Server via pyodbc.

    Returns (success, error_message, rows).
    rows is a list of dicts {column_name: value}.
    """

    def __init__(self, conn_str: str = SQL_CONN_STR) -> None:
        if not conn_str:
            raise ValueError(
                "SQL_CONN_STR is not configured. "
                "Set it in .env: SQL_CONN_STR=Driver={ODBC Driver 17 ...}"
            )
        self.conn_str = conn_str

    def execute(self, sql: str, fetch_limit: int = FETCH_LIMIT) -> tuple[bool, str, list[dict]]:
        """
        Execute sql and return (ok, error, rows).
        Rows are limited to fetch_limit to avoid memory issues.
        """
        try:
            import pyodbc
        except ImportError as e:
            raise ImportError("pyodbc not installed. Run: pip install pyodbc") from e

        try:
            with __import__("pyodbc").connect(self.conn_str, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                cols = [col[0] for col in cursor.description]
                rows = cursor.fetchmany(fetch_limit)
                return True, "", [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            return False, str(e), []


# ─────────────────────────────────────────────
# 5. SELF-CORRECTOR
# ─────────────────────────────────────────────

class SelfCorrector:
    """
    Multi-turn self-correction loop.

    If the first SQL attempt fails (DB error), adds the error message
    to the conversation and asks the model to fix the SQL.
    Repeats up to max_retries times.

    Important:
    - Only triggers on definitive DB errors (not on empty result sets).
    - Caps at max_retries to avoid "correction hallucination" (fixing correct SQL into wrong SQL).
    """

    def __init__(self, generator: SQLGenerator, executor: QueryExecutor, max_retries: int = MAX_RETRIES) -> None:
        self.generator   = generator
        self.executor    = executor
        self.max_retries = max_retries

    def run(
        self,
        initial_messages: list[dict],
        initial_sql:      str,
    ) -> tuple[str, bool, str, list[dict], list[dict]]:
        """
        Run the self-correction loop.

        Args:
            initial_messages: OpenAI-style messages used for first generation
            initial_sql     : SQL from first generation attempt

        Returns:
            (final_sql, success, error_msg, result_rows, correction_history)
        """
        messages = list(initial_messages)  # copy
        sql      = initial_sql
        history: list[dict] = []

        for attempt in range(self.max_retries + 1):
            is_safe, safety_err = SafetyValidator.validate(sql)
            if not is_safe:
                history.append({"attempt": attempt + 1, "sql": sql, "error": safety_err})
                # Build correction message
                messages.append({"role": "assistant", "content": sql})
                messages.append({
                    "role": "user",
                    "content": (
                        f"SQL trên bị từ chối bởi safety validator: {safety_err}\n"
                        "Hãy viết lại SQL đúng (chỉ SELECT)."
                    ),
                })
                sql = self.generator.generate(messages)
                continue

            ok, err, rows = self.executor.execute(sql)

            if ok:
                return sql, True, "", rows, history

            # DB error — record and attempt correction
            history.append({"attempt": attempt + 1, "sql": sql, "error": err})

            if attempt < self.max_retries:
                correction_msg = (
                    f"SQL trên bị lỗi khi thực thi:\n{err}\n\n"
                    "Hãy phân tích lỗi và viết lại SQL đúng."
                )
                messages.append({"role": "assistant", "content": sql})
                messages.append({"role": "user",      "content": correction_msg})

                sql = self.generator.generate(messages)

        # Exhausted retries
        return sql, False, err, [], history


# ─────────────────────────────────────────────
# 6. RESULT FORMATTER
# ─────────────────────────────────────────────

class ResultFormatter:
    """Format raw DB rows for display or API response."""

    @staticmethod
    def format_table(rows: list[dict], max_width: int = 120) -> str:
        """Render rows as a simple ASCII table."""
        if not rows:
            return "(no rows returned)"

        cols = list(rows[0].keys())
        col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        col_widths = {c: min(w, max_width // len(cols)) for c, w in col_widths.items()}

        header = " | ".join(c.ljust(col_widths[c]) for c in cols)
        sep    = "-+-".join("-" * col_widths[c] for c in cols)
        lines  = [header, sep]
        for row in rows:
            lines.append(" | ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols))
        return "\n".join(lines)

    @staticmethod
    def to_json_serializable(rows: list[dict]) -> list[dict]:
        """Convert DB-native types (date, decimal) to JSON-safe Python types."""
        import datetime, decimal
        result = []
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, (datetime.date, datetime.datetime)):
                    clean[k] = v.isoformat()
                elif isinstance(v, decimal.Decimal):
                    clean[k] = float(v)
                elif isinstance(v, bytes):
                    clean[k] = v.hex()
                else:
                    clean[k] = v
            result.append(clean)
        return result


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class InferencePipeline:
    """
    Orchestrates all components into a single .run(question) call.

    Args:
        model_path     : path to fine-tuned model (local directory)
        use_unsloth    : use unsloth for faster inference (default True)
        use_schema_linker: override USE_SCHEMA_LINKER env var
        max_retries    : max self-correction attempts (default MAX_RETRIES)
        conn_str       : SQL Server connection string (default from .env)
    """

    def __init__(
        self,
        model_path:        str,
        use_unsloth:       bool = True,
        use_schema_linker: bool = USE_SCHEMA_LINKER,
        max_retries:       int  = MAX_RETRIES,
        conn_str:          str  = SQL_CONN_STR,
    ) -> None:
        self.schema_injector = SchemaInjector(use_dynamic=use_schema_linker)
        self.generator       = SQLGenerator(model_path, use_unsloth=use_unsloth)
        self.executor        = QueryExecutor(conn_str) if conn_str else None
        self.corrector: Optional[SelfCorrector] = (
            SelfCorrector(self.generator, self.executor, max_retries)
            if self.executor else None
        )

    def run(self, question: str) -> InferenceResult:
        """
        Full pipeline: question → SQL → (execute + self-correct) → InferenceResult.

        If no SQL_CONN_STR is configured, skips DB execution and self-correction.
        """
        system_prompt = self.schema_injector.get_system_prompt(question)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ]

        # ── Step 1: Initial generation ──────────────────────────────
        sql = self.generator.generate(messages)

        # ── Step 2: Check for refusal ────────────────────────────────
        lower_sql = sql.lower().strip()
        if any(kw in lower_sql for kw in _REFUSAL_KEYWORDS):
            return InferenceResult(
                question=question,
                sql=sql,
                rows=[],
                success=True,
                is_refusal=True,
                attempts=1,
            )

        # ── Step 3: Safety check ─────────────────────────────────────
        is_safe, safety_err = SafetyValidator.validate(sql)
        if not is_safe:
            return InferenceResult(
                question=question,
                sql=sql,
                rows=[],
                success=False,
                error=f"Safety validation failed: {safety_err}",
                attempts=1,
            )

        # ── Step 4: DB execution (+ self-correction if enabled) ──────
        if self.executor is None:
            # No DB configured: return SQL only
            return InferenceResult(
                question=question,
                sql=sql,
                rows=[],
                success=True,
                attempts=1,
            )

        if self.corrector is not None:
            final_sql, ok, err, rows, history = self.corrector.run(messages, sql)
        else:
            ok, err, rows = self.executor.execute(sql)
            final_sql, history = sql, []

        attempts = 1 + len([h for h in history if h.get("error")])

        return InferenceResult(
            question=question,
            sql=final_sql,
            rows=ResultFormatter.to_json_serializable(rows),
            success=ok,
            error=err,
            attempts=attempts,
            correction_history=history,
        )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys
    import json as _json

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Text-to-SQL inference with self-correction")
    parser.add_argument("--question",   required=True,   help="User question (tiếng Việt)")
    parser.add_argument("--model-path", required=True,   help="Path to fine-tuned model directory")
    parser.add_argument("--no-unsloth", action="store_true", help="Disable unsloth (use raw transformers)")
    parser.add_argument("--no-db",      action="store_true", help="Skip DB execution (output SQL only)")
    parser.add_argument("--schema-linker", action="store_true", help="Enable dynamic schema linking")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help=f"Max self-correction attempts (default {MAX_RETRIES})")
    parser.add_argument("--json",       action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    conn = "" if args.no_db else SQL_CONN_STR

    pipe = InferencePipeline(
        model_path=args.model_path,
        use_unsloth=not args.no_unsloth,
        use_schema_linker=args.schema_linker,
        max_retries=args.max_retries,
        conn_str=conn,
    )

    result = pipe.run(args.question)

    if args.json:
        print(_json.dumps({
            "question":           result.question,
            "sql":                result.sql,
            "success":            result.success,
            "is_refusal":         result.is_refusal,
            "error":              result.error,
            "attempts":           result.attempts,
            "rows":               result.rows,
            "correction_history": result.correction_history,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\nQuestion : {result.question}")
        print(f"SQL      :\n{textwrap.indent(result.sql, '  ')}")
        print(f"Attempts : {result.attempts}")
        if result.is_refusal:
            print("Status   : REFUSAL")
        elif result.success:
            print(f"Status   : OK ({len(result.rows)} rows)")
            if result.rows:
                print("\nResult:")
                print(ResultFormatter.format_table(result.rows))
        else:
            print(f"Status   : FAILED — {result.error}")
            if result.correction_history:
                print("\nCorrection history:")
                for h in result.correction_history:
                    print(f"  Attempt {h['attempt']}: {h['error'][:100]}")
