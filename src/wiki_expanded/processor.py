"""Processes Wikipedia JSONL articles and builds dependency dictionaries."""

import datetime
import json
import logging
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

    Builds the following dictionaries:
        - title_to_text: Article title -> Article text
        - title_to_links: Article title -> List of outgoing links
        - link_to_freq: Link -> Number of articles it appears in
        - title_to_tokens: Article title -> Token IDs
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
        self.title_to_text: dict[str, str] = {}
        self.title_to_tokens: dict[str, list[int]] = {}
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
        for i, article in enumerate(self._stream_jsonl(jsonl_file=jsonl_file)):
            if not i % self.PROGRESS_LOG_INTERVAL:
                logger.info(f"Processed {i} articles")

            plain_text = article["plaintext"].strip()
            if not plain_text:
                continue

            text = f"# {article['title']}\n\n{plain_text}"
            title = article["title"]

            # Check for duplicates BEFORE updating dictionaries to avoid data corruption
            assert title not in self.titles_seen, f"Duplicate title: {title}"

            links = self._get_links(article=article)
            tokens = self._tokenize_text(text=text)
            self._update_dictionaries(
                title=title, text=text, links=links, tokens=tokens
            )

            self.titles_seen.add(title)
            self.articles_processed += 1

            if self.max_files and self.articles_processed >= self.max_files:
                self._save_to_disk()
                return

        self._save_to_disk()
        logger.info(f"Done processing {self.articles_processed} articles.")

    def _update_dictionaries(
        self, title: str, text: str, links: list[str], tokens: list[int]
    ) -> None:
        """Update dictionaries used to build the expanded dataset."""
        self.title_to_links[title] = links
        self.link_to_freq.update(links)
        self.title_to_text[title] = text
        self.title_to_tokens[title] = tokens

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

    def _tokenize_text(self, text: str) -> list[int]:
        """Tokenize the text using the configured tokenizer.

        Args:
            text: The text to tokenize.

        Returns:
            A list of token IDs.
        """
        input_ids = self.tokenizer(text, return_tensors="pt")
        tokens = input_ids["input_ids"].squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return tokens

    def _save_to_disk(self) -> None:
        """Save the dictionaries to disk."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.save_dir / date_str
        save_dir.mkdir(parents=True, exist_ok=True)

        title_to_links_path = save_dir / FILE_NAMES["title_to_links"]
        self._dump(data=self.title_to_links, path=title_to_links_path)

        title_to_text_path = save_dir / FILE_NAMES["title_to_text"]
        self._dump(data=self.title_to_text, path=title_to_text_path)

        title_to_tokens_path = save_dir / FILE_NAMES["title_to_tokens"]
        self._dump(data=self.title_to_tokens, path=title_to_tokens_path)

        link_to_freq_path = save_dir / FILE_NAMES["link_to_freq"]
        self._dump(data=self.link_to_freq, path=link_to_freq_path)

    @staticmethod
    def _dump(data: dict, path: Path) -> None:
        """Dump a dictionary to a JSON file."""
        with open(path, "w") as f:
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
