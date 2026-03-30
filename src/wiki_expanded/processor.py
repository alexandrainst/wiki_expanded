"""Processes Wikipedia JSONL articles and builds dependency dictionaries."""

import datetime
import html
import json
import logging
import re
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
    - ``title_to_url.json`` mapping title -> stable Wikipedia URL
    """

    PROGRESS_LOG_INTERVAL = 10000
    UNICODE_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})")

    def __init__(
        self,
        save_dir: Path,
        tokenizer_name: str = "google/gemma-7b",
        max_files: int | None = None,
    ) -> None:
        """Initialize the Processor."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_num_tokens: dict[str, int] = {}
        self.title_to_url: dict[str, str] = {}
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

                plain_text = self._build_plaintext_from_sections(article).strip()
                if not plain_text:
                    continue

                title = self._normalize_wiki_text(article["title"]).strip()
                text = f"# {title}\n\n{plain_text}"
                url = self._get_curid_url(article=article)

                # Check for duplicates BEFORE updating dictionaries to avoid
                # data corruption.
                assert title not in self.titles_seen, f"Duplicate title: {title}"

                links = self._get_links(article=article)
                num_tokens = self._count_tokens(text=text)

                self._update_dictionaries(
                    title=title, links=links, num_tokens=num_tokens, url=url
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
        self, title: str, links: list[str], num_tokens: int, url: str
    ) -> None:
        """Update dictionaries used to build the expanded dataset."""
        self.title_to_links[title] = links
        self.link_to_freq.update(links)
        self.title_to_num_tokens[title] = num_tokens
        self.title_to_url[title] = url

    def _get_curid_url(self, article: dict[str, Any]) -> str:
        """Build a stable Danish Wikipedia URL from pageID."""
        page_id = article.get("pageID")
        page_id_str = str(page_id).strip() if page_id is not None else ""

        if not page_id_str or not page_id_str.isdigit():
            title = article.get("title", "<missing title>")
            raise ValueError(f"Invalid pageID for article '{title}': {page_id!r}")

        return f"https://da.wikipedia.org/?curid={page_id_str}"

    @classmethod
    def _build_plaintext_from_sections(cls, article: dict[str, Any]) -> str:
        """Build plaintext from the article's sections, including section headers.

        Reconstructs the article body text from the structured sections data,
        inserting markdown headings for each section title. Depth 0 maps to ##,
        depth 1 to ###, etc. (since # is reserved for the article title).

        Args:
            article: A parsed Wikipedia article dict with a 'sections' key.

        Returns:
            The reconstructed plaintext string.
        """
        parts: list[str] = []
        sections = article.get("sections", [])

        for idx, section in enumerate(sections):
            section_parts: list[str] = []

            # Emit section heading (skip empty titles, i.e. the lead section)
            title = cls._normalize_wiki_text(section.get("title", "")).strip()
            if title:
                heading_level = section.get("depth", 0) + 2  # depth 0 -> ##
                heading_prefix = "#" * heading_level
                section_parts.append(f"{heading_prefix} {title}")

            # Collect paragraph text
            for paragraph in section.get("paragraphs", []):
                sentences = paragraph.get("sentences", [])
                sentence_texts = [
                    cls._normalize_wiki_text(s["text"])
                    for s in sentences
                    if s.get("text")
                ]
                if sentence_texts:
                    section_parts.append(" ".join(sentence_texts))

            # Collect list items
            for lst in section.get("lists", []):
                list_lines = []
                for item in lst:
                    item_text = cls._normalize_wiki_text(item.get("text", "")).strip()
                    if item_text:
                        list_lines.append(f" * {item_text}")
                if list_lines:
                    section_parts.append("\n".join(list_lines))

            # Drop heading-only sections unless followed by a deeper subsection
            has_only_heading = len(section_parts) == 1 and section_parts[0].startswith(
                "#"
            )
            if has_only_heading:
                current_depth = section.get("depth", 0)
                next_depth = (
                    sections[idx + 1].get("depth", 0) if idx + 1 < len(sections) else -1
                )
                if next_depth <= current_depth:
                    continue

            if section_parts:
                parts.append("\n\n".join(section_parts))

        # Strip trailing heading-only parts (e.g. empty "Noter", "Referencer")
        while parts and all(line.startswith("#") for line in parts[-1].splitlines()):
            parts.pop()

        return "\n\n".join(parts)

    @classmethod
    def _get_links(cls, article: dict[str, Any]) -> list[str]:
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
                            link_title = cls._normalize_wiki_text(link["page"]).strip()
                            if link_title:
                                links.add(link_title)
        return list(links)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        input_ids = self.tokenizer(text, return_tensors="pt")
        tokens = input_ids["input_ids"].squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return len(tokens)

    @classmethod
    def _decode_unicode_escapes(cls, value: str) -> str:
        r"""Decode only literal unicode escape sequences (e.g. \u002e)."""
        return cls.UNICODE_ESCAPE_PATTERN.sub(
            lambda match: chr(int(match.group(1), 16)), value
        )

    @classmethod
    def _normalize_wiki_text(cls, value: str) -> str:
        """Normalize escaped unicode and HTML entities in wiki text fields."""
        return html.unescape(cls._decode_unicode_escapes(value))

    def _save_to_disk(self, save_dir: Path) -> None:
        """Save the in-memory dictionaries to disk."""
        title_to_links_path = save_dir / FILE_NAMES["title_to_links"]
        self._dump(data=self.title_to_links, path=title_to_links_path)

        title_to_num_tokens_path = save_dir / FILE_NAMES["title_to_num_tokens"]
        self._dump(data=self.title_to_num_tokens, path=title_to_num_tokens_path)

        title_to_url_path = save_dir / FILE_NAMES["title_to_url"]
        self._dump(data=self.title_to_url, path=title_to_url_path)

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
