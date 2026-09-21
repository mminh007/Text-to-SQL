"""
src/inference/pipeline.py
==========================
End-to-end Text-to-SQL inference pipeline.

Components:
    1. SchemaInjector  — inject Koel DB schema into system prompt
    2. SQLGenerator    — call fine-tuned model to generate SQL
    3. SQLValidator    — syntax + safety check (sqlglot + regex)
    4. QueryExecutor   — execute SQL on SQL Server
    5. ResultFormatter — format raw rows for API response

Usage (standalone):
    python src/inference/pipeline.py \\
        --question "Top 10 bài hát được nghe nhiều nhất?" \\
        --model-path outputs/qwen25-7b-koel-sql/merged

Phase 4 placeholder — implementation after Phase 3 evaluation.
"""

# TODO (Phase 4): Implement full inference pipeline
# See PROJECT_AUDIT.md Phase 4 for architecture details

raise NotImplementedError(
    "Pipeline not yet implemented. Implement after Phase 3 evaluation completes."
    " See src/api/main.py and PROJECT_AUDIT.md Phase 4 for details."
)
