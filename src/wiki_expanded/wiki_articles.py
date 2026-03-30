"""Build a simple Wikipedia dataset with url, title, and text columns."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonlines

from .constants import FILE_NAMES


class WikiArticlesBuilder:
    """Build a simple dataset from processed dictionaries.

    The output rows have the schema: url, title, text.

    Args:
        processed_dir: Path to an exact processed run folder containing
            articles.sqlite3 and title_to_url.json.
        output_file: Path to the output JSONL file.
    """

    def __init__(self, processed_dir: Path, output_file: Path) -> None:
        """Initialize the builder."""
        self.processed_dir = processed_dir
        self.output_file = output_file

    def build(self) -> None:
        """Build and save the dataset to disk."""
        title_to_url = self._load_json(self.processed_dir / FILE_NAMES["title_to_url"])

        db_path = self.processed_dir / FILE_NAMES["articles_db"]
        if not db_path.exists():
            raise FileNotFoundError(f"Missing required SQLite database: {db_path}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT title, text FROM articles ORDER BY title")
            with jsonlines.open(self.output_file, mode="w") as writer:
                for title, text in cursor:
                    if title not in title_to_url:
                        continue
                    writer.write(
                        {"url": title_to_url[title], "title": title, "text": text}
                    )
        finally:
            conn.close()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load a required JSON dictionary from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {path}")
        return data

    @staticmethod
    def _remove_title_from_text(text: str) -> str:
        """Remove markdown title heading from text by splitting on double newline."""
        _, separator, remainder = text.partition("\n\n")
        if separator:
            return remainder
        return text
