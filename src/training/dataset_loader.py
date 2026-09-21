"""
src/training/dataset_loader.py
================================
Load + preprocess the Text-to-SQL JSONL dataset for Unsloth / TRL training.

Each sample in the JSONL has the structure:
    {
        "messages": [
            {"role": "system",    "content": "<system prompt>"},
            {"role": "user",      "content": "<natural language question>"},
            {"role": "assistant", "content": "<SQL or refusal>"}
        ],
        "_tier": "T3",          # optional metadata
        "_tables": ["songs", "albums"]
    }

The loader:
1. Reads train.jsonl / val.jsonl
2. Formats each sample using the tokenizer's chat template
3. Returns a HuggingFace Dataset ready for SFTTrainer
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


# ── Schema loading ─────────────────────────────────────────────────────────────

_KOEL_SCHEMA = """-- Koel Music Streaming DB (SQL Server / T-SQL)
CREATE TABLE artists (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name        NVARCHAR(255) NOT NULL,
    image       NVARCHAR(2048) NULL,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE albums (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    artist_id   UNIQUEIDENTIFIER NULL REFERENCES artists(id),
    name        NVARCHAR(255) NOT NULL,
    cover       NVARCHAR(2048) NULL,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE songs (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    album_id    UNIQUEIDENTIFIER NULL REFERENCES albums(id),
    artist_id   UNIQUEIDENTIFIER NULL REFERENCES artists(id),
    title       NVARCHAR(255) NOT NULL,
    length      FLOAT NOT NULL,          -- duration in seconds
    track       INT NULL,
    disc        INT NULL DEFAULT 1,
    lyrics      NVARCHAR(MAX) NULL,
    path        NVARCHAR(MAX) NOT NULL,
    mtime       INT NOT NULL,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE genres (
    id          INT IDENTITY PRIMARY KEY,
    name        NVARCHAR(255) NOT NULL UNIQUE
);
CREATE TABLE genre_song (
    genre_id    INT NOT NULL REFERENCES genres(id),
    song_id     UNIQUEIDENTIFIER NOT NULL REFERENCES songs(id),
    PRIMARY KEY (genre_id, song_id)
);
CREATE TABLE users (
    id              UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name            NVARCHAR(255) NOT NULL,
    email           NVARCHAR(255) NOT NULL UNIQUE,
    is_admin        BIT NOT NULL DEFAULT 0,
    invitation_accepted_at DATETIME2 NULL,
    created_at      DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at      DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE organizations (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    name        NVARCHAR(255) NOT NULL
);
CREATE TABLE playlists (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    name        NVARCHAR(255) NOT NULL,
    is_public   BIT NOT NULL DEFAULT 0,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE playlist_song (
    playlist_id UNIQUEIDENTIFIER NOT NULL REFERENCES playlists(id),
    song_id     UNIQUEIDENTIFIER NOT NULL REFERENCES songs(id),
    position    INT NULL,
    PRIMARY KEY (playlist_id, song_id)
);
CREATE TABLE playlist_user (
    playlist_id UNIQUEIDENTIFIER NOT NULL REFERENCES playlists(id),
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    PRIMARY KEY (playlist_id, user_id)
);
CREATE TABLE playlist_folders (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    name        NVARCHAR(255) NOT NULL
);
CREATE TABLE playlist_playlist_folder (
    playlist_id        UNIQUEIDENTIFIER NOT NULL REFERENCES playlists(id),
    playlist_folder_id UNIQUEIDENTIFIER NOT NULL REFERENCES playlist_folders(id),
    PRIMARY KEY (playlist_id, playlist_folder_id)
);
CREATE TABLE interactions (
    id              INT IDENTITY PRIMARY KEY,
    user_id         UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    song_id         UNIQUEIDENTIFIER NOT NULL REFERENCES songs(id),
    liked           BIT NOT NULL DEFAULT 0,
    play_count      INT NOT NULL DEFAULT 0,
    last_played_at  DATETIME2 NULL,
    created_at      DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at      DATETIME2 NOT NULL DEFAULT GETDATE(),
    UNIQUE (user_id, song_id)
);
CREATE TABLE favorites (
    id                  INT IDENTITY PRIMARY KEY,
    user_id             UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    favoriteable_type   NVARCHAR(255) NOT NULL,  -- 'songs', 'albums', 'artists'
    favoriteable_id     UNIQUEIDENTIFIER NOT NULL,
    created_at          DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE ratings (
    id              INT IDENTITY PRIMARY KEY,
    user_id         UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    rateable_type   NVARCHAR(255) NOT NULL,
    rateable_id     UNIQUEIDENTIFIER NOT NULL,
    rating          TINYINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at      DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE podcasts (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    title       NVARCHAR(255) NOT NULL,
    url         NVARCHAR(2048) NOT NULL UNIQUE,
    description NVARCHAR(MAX) NULL,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE podcast_user (
    podcast_id  UNIQUEIDENTIFIER NOT NULL REFERENCES podcasts(id),
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    PRIMARY KEY (podcast_id, user_id)
);
CREATE TABLE audits (
    id              INT IDENTITY PRIMARY KEY,
    user_type       NVARCHAR(255) NULL,
    user_id         UNIQUEIDENTIFIER NULL,
    event           NVARCHAR(255) NOT NULL,
    auditable_type  NVARCHAR(255) NULL,
    auditable_id    UNIQUEIDENTIFIER NULL,
    ip_address      NVARCHAR(45) NULL,
    url             NVARCHAR(2048) NULL,
    created_at      DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE queue_states (
    id          INT IDENTITY PRIMARY KEY,
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id) UNIQUE,
    song_ids    NVARCHAR(MAX) NULL,   -- JSON array of song UUIDs
    playback_position FLOAT NULL,
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE settings (
    key         NVARCHAR(255) PRIMARY KEY,
    value       NVARCHAR(MAX) NULL
);
CREATE TABLE agent_conversations (
    id          UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    user_id     UNIQUEIDENTIFIER NOT NULL REFERENCES users(id),
    title       NVARCHAR(255) NULL,
    created_at  DATETIME2 NOT NULL DEFAULT GETDATE(),
    updated_at  DATETIME2 NOT NULL DEFAULT GETDATE()
);
CREATE TABLE agent_conversation_messages (
    id              UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    conversation_id UNIQUEIDENTIFIER NOT NULL REFERENCES agent_conversations(id),
    role            NVARCHAR(50) NOT NULL,  -- 'user' | 'assistant'
    content         NVARCHAR(MAX) NOT NULL,
    created_at      DATETIME2 NOT NULL DEFAULT GETDATE()
);
"""


def get_schema() -> str:
    """Return the Koel DB schema string."""
    return _KOEL_SCHEMA.strip()


# ── System prompt ──────────────────────────────────────────────────────────────

def load_system_prompt(prompt_file: Optional[Path] = None) -> str:
    """Load system prompt and inject schema."""
    default_path = Path("prompts/system_prompt.txt")
    path = prompt_file or default_path

    if path.exists():
        template = path.read_text(encoding="utf-8")
        return template.replace("{schema}", get_schema())
    else:
        # Inline fallback
        logger.warning("System prompt file not found at %s, using inline fallback.", path)
        return (
            "Bạn là SQL expert cho hệ thống Koel music streaming (SQL Server).\n"
            "RULES:\n"
            "- Chỉ sinh câu lệnh SELECT\n"
            "- Không sinh INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER\n"
            "- Output: SQL thuần, không markdown fence, không giải thích\n\n"
            f"SCHEMA:\n{get_schema()}"
        )


# ── JSONL loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at line %d in %s: %s", i, path, e)
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def _override_system_prompt(sample: dict, system_prompt: str) -> dict:
    """Replace the system message with our canonical prompt (schema included)."""
    messages = sample.get("messages", [])
    if messages and messages[0].get("role") == "system":
        messages = [{"role": "system", "content": system_prompt}] + messages[1:]
    return {**sample, "messages": messages}


# ── HuggingFace Dataset builder ─────────────────────────────────────────────────

def build_dataset(
    jsonl_path: Path,
    tokenizer,
    system_prompt_file: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Build a HuggingFace Dataset from JSONL.

    Each row has a 'text' field containing the full conversation formatted
    with the tokenizer's chat template (including special tokens).

    Args:
        jsonl_path: Path to train.jsonl or val.jsonl
        tokenizer: Loaded tokenizer with apply_chat_template support
        system_prompt_file: Optional path to prompts/system_prompt.txt
        max_samples: Truncate dataset for quick debugging

    Returns:
        HuggingFace Dataset with columns: ['text', 'tier']
    """
    system_prompt = load_system_prompt(system_prompt_file)
    raw_samples = load_jsonl(jsonl_path)

    if max_samples:
        raw_samples = raw_samples[:max_samples]
        logger.info("Truncated to %d samples for debugging.", max_samples)

    texts = []
    tiers = []

    for sample in raw_samples:
        # Inject canonical system prompt
        sample = _override_system_prompt(sample, system_prompt)
        messages = sample["messages"]

        # Apply chat template — adds BOS/EOS and special tokens
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            tiers.append(sample.get("_tier", "unknown"))
        except Exception as e:
            logger.warning("Skipping sample due to template error: %s", e)
            continue

    logger.info("Built %d formatted samples.", len(texts))

    return Dataset.from_dict({"text": texts, "tier": tiers})


def get_tier_distribution(dataset: Dataset) -> dict[str, int]:
    """Return tier distribution for logging."""
    from collections import Counter
    return dict(Counter(dataset["tier"]))
