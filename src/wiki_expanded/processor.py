"""Processes Wikipedia JSONL articles and builds dependency dictionaries."""

import datetime
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import jsonlines
from transformers import AutoTokenizer

from .constants import FILE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="processing.log",
)

logger = logging.getLogger(__name__)


class Processor:
    """Process a Wikipedia JSONL file and build dictionaries for the dataset.

    This writes:

    - a SQLite DB (``articles_db``) with article title and text
    - ``title_to_links.json`` mapping title -> list of outgoing links
    - ``link_to_freq.json`` mapping link -> number of articles it appears in
    - ``title_to_num_tokens.json`` mapping title -> number of tokens in the article
    """

    PROGRESS_LOG_INTERVAL = 10000

    def __init__(
        self,
        save_dir: Path,
        tokenizer_name: str = "google/gemma-7b",
        max_files: int | None = None,
    ) -> None:
        """Initialize the Processor."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_num_tokens: dict[str, int] = {}
        self.link_to_freq: Counter[str] = Counter()

        self.titles_seen: set[str] = set()
        self.articles_processed: int = 0
        self.max_files: int | None = max_files

        self.save_dir: Path = save_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process(self, jsonl_file: Path) -> None:
        """Build dictionaries used for the expanded Wikipedia dataset.

        Args:
            jsonl_file: Path to a JSONL file containing the Wikipedia
                articles to process.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_dir / date_str
        save_dir.mkdir(parents=True, exist_ok=True)

        db_path = save_dir / FILE_NAMES["articles_db"]
        conn = sqlite3.connect(db_path)
        try:
            self._init_db(conn)

            for i, article in enumerate(self._stream_jsonl(jsonl_file=jsonl_file)):
                if not i % self.PROGRESS_LOG_INTERVAL:
                    logger.info(f"Processed {i} articles")

                plain_text = article["plaintext"].strip()
                if not plain_text:
                    continue

                text = f"# {article['title']}\n\n{plain_text}"
                title = article["title"]

                # Check for duplicates BEFORE updating dictionaries to avoid
                # data corruption.
                assert title not in self.titles_seen, f"Duplicate title: {title}"

                links = self._get_links(article=article)
                num_tokens = self._count_tokens(text=text)

                self._update_dictionaries(
                    title=title, links=links, num_tokens=num_tokens
                )
                self._insert_article(conn=conn, title=title, text=text)

                self.titles_seen.add(title)
                self.articles_processed += 1

                if self.max_files and self.articles_processed >= self.max_files:
                    break

            self._save_to_disk(save_dir=save_dir)
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Done processing {self.articles_processed} articles.")

    @staticmethod
    def _init_db(conn: sqlite3.Connection) -> None:
        """Initialise the SQLite database for article storage."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                title TEXT PRIMARY KEY,
                text TEXT NOT NULL
            )
            """
        )
        conn.commit()

    @staticmethod
    def _insert_article(conn: sqlite3.Connection, title: str, text: str) -> None:
        """Insert or replace a single article in the SQLite database."""
        conn.execute(
            "INSERT OR REPLACE INTO articles (title, text) VALUES (?, ?)", (title, text)
        )

    def _update_dictionaries(
        self, title: str, links: list[str], num_tokens: int
    ) -> None:
        """Update dictionaries used to build the expanded dataset."""
        self.title_to_links[title] = links
        self.link_to_freq.update(links)
        self.title_to_num_tokens[title] = num_tokens

    @staticmethod
    def _get_links(article: dict[str, Any]) -> list[str]:
        """Get the links of a Wikipedia article."""
        links: set[str] = set()

        for section in article["sections"]:
            if "paragraphs" not in section:
                continue

            for paragraph in section["paragraphs"]:
                if "sentences" not in paragraph:
                    continue

                for sentence in paragraph["sentences"]:
                    if "links" not in sentence:
                        continue

                    for link in sentence["links"]:
                        if link["type"] == "internal":
                            links.add(link["page"])
        return list(links)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        input_ids = self.tokenizer(text, return_tensors="pt")
        tokens = input_ids["input_ids"].squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return len(tokens)

    def _save_to_disk(self, save_dir: Path) -> None:
        """Save the in-memory dictionaries to disk."""
        title_to_links_path = save_dir / FILE_NAMES["title_to_links"]
        self._dump(data=self.title_to_links, path=title_to_links_path)

        title_to_num_tokens_path = save_dir / FILE_NAMES["title_to_num_tokens"]
        self._dump(data=self.title_to_num_tokens, path=title_to_num_tokens_path)

        link_to_freq_path = save_dir / FILE_NAMES["link_to_freq"]
        self._dump(data=self.link_to_freq, path=link_to_freq_path)

    @staticmethod
    def _dump(data: dict, path: Path) -> None:
        """Dump a dictionary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def _stream_jsonl(jsonl_file: Path) -> Iterator[dict[str, Any]]:
        """Stream articles from a JSONL file one by one."""
        with jsonlines.open(jsonl_file, mode="r") as reader:
            for article in reader:
                yield article


def save(data: dict, filepath: Path = Path("tmp.json")) -> None:
    """Save a dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
