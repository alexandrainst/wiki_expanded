"""Builds the Expanded Wikipedia dataset."""

import datetime
import json
import logging
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import jsonlines

from .constants import FILE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="dataset_builder.log",
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build the expanded Wikipedia dataset.

    Args:
        processed_dir (Path): The directory containing the processed dictionaries
            from running `src/scripts/process.py`.
        save_dir (Path): The path to save the expanded dataset.
        min_tokens (int): Minimum number of tokens required for a sample to be
            included in the dataset. Defaults to 0.
        max_tokens (int): Stop expanding links when the text reaches this many
            tokens. Defaults to sys.maxsize (effectively no limit).
        max_dataset_length (int, optional): The maximum number of samples in the
            expanded dataset. If None, use all.
        max_link_expansions_local (int, optional): The maximum number of links
            that can be expanded for a single sample. If None, no limit.
        max_link_expansions_global (int, optional): The maximum number of times
            a link can be expanded across all samples. If None, no limit.
        include_strategy (str, optional): The strategy to use for including the link
            text in the expanded text. If "prepend", the link text is prepended to the
            text. If "append", the link text is appended to the text.
    """

    PROGRESS_LOG_INTERVAL = 10000

    def __init__(
        self,
        processed_dir: Path,
        save_dir: Path,
        min_tokens: int = 0,
        max_tokens: int = sys.maxsize,
        max_dataset_length: int | None = None,
        max_link_expansions_local: int | None = None,
        max_link_expansions_global: int | None = None,
        include_strategy: str = "prepend",
    ) -> None:
        """Initialize the DatasetBuilder."""
        self.title_to_links: dict[str, list[str]] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.title_to_num_tokens: dict[str, int] = {}
        self.min_tokens: int = min_tokens
        self.max_tokens: int = max_tokens
        self.save_dir: Path = save_dir / datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.link_expansion_count: Counter[str] = Counter()
        self.files_expanded: int = 0
        self.dataset_length: int = 0
        self.max_dataset_length: int | None = max_dataset_length
        self.max_link_expansions_local: int | None = max_link_expansions_local
        self.max_link_expansions_global: int | None = max_link_expansions_global

        self.include_strategy: str = include_strategy

        self.links_considered_in_first_iteration_but_not_expanded: set[str] = set()
        self.all_links_considered_for_expansion: set[str] = set()

        self.dataset_path: Path = self.save_dir / FILE_NAMES["dataset"]
        self.link_expansion_path: Path = (
            self.save_dir / FILE_NAMES["link_expansion_count"]
        )

        self._articles_db_conn: sqlite3.Connection | None = None

        self._read_processed_data(processed_dir=processed_dir)

    def build_expanded_dataset(self) -> None:
        """Build the expanded Wikipedia dataset."""
        try:
            for iteration in range(2):
                self._init_dataset()
                self._build_dataset(iteration=iteration)

                if iteration == 0:
                    self.links_considered_in_first_iteration_but_not_expanded = (
                        self.all_links_considered_for_expansion
                        - set(self.link_expansion_count.keys())
                    )
                self._log_iteration_stats(iteration=iteration)

            self._save_link_expansion_count()
            logger.info(f"Done building dataset. Saved {self.dataset_length} samples.")
        finally:
            if self._articles_db_conn is not None:
                self._articles_db_conn.close()
                self._articles_db_conn = None

    def _init_dataset(self) -> None:
        """Initialise the dataset."""
        self.link_expansion_count = Counter()
        self.files_expanded = 0
        self.dataset_length = 0

        if not self.dataset_path.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_path.touch()

    def _build_dataset(self, iteration: int) -> None:
        """Build the dataset."""
        n_titles = self._get_n_titles()
        for i, (title, text) in enumerate(self._iter_titles_and_texts()):
            if not i % self.PROGRESS_LOG_INTERVAL:
                logger.info("Processed %d/%d titles", i, n_titles)

            sample = self._expand(title=title, text=text)
            if not self.min_tokens <= sample["n_tokens"] <= self.max_tokens:
                continue

            if iteration == 0:
                links = self.title_to_links[title]
                self.all_links_considered_for_expansion.update(links)

            self._track_expanded_links(links=sample["links_expanded"])
            if iteration == 1:
                self._append_sample(sample=sample)

            dataset_done = self.max_dataset_length and (
                self.dataset_length >= self.max_dataset_length
            )
            if dataset_done:
                break

    def _iter_titles_and_texts(self) -> Iterator[tuple[str, str]]:
        """Yield (title, text) pairs from either SQLite or in-memory dict."""
        if self._articles_db_conn is None:
            msg = "articles_db connection is not initialised."
            raise RuntimeError(msg)

        cursor = self._articles_db_conn.execute("SELECT title, text FROM articles")
        for title, text in cursor:
            yield str(title), str(text)

    def _get_n_titles(self) -> int:
        """Return the number of titles available for expansion."""
        if self._articles_db_conn is None:
            msg = "articles_db connection is not initialised."
            raise RuntimeError(msg)

        cursor = self._articles_db_conn.execute("SELECT COUNT(*) FROM articles")
        (count,) = cursor.fetchone()
        return int(count)

    def _find_links_not_expanded_in_iteration(self) -> set[str]:
        """Find links that were not expanded in the first iteration.

        These will be prioritized in the second iteration.
        """
        links = [
            link
            for link in self.link_to_freq
            if self._article_exists(link) and link not in self.link_expansion_count
        ]
        return set(links)

    def _track_expanded_links(self, links: list[str]) -> None:
        """Track the expanded links."""
        for link in links:
            self.link_expansion_count[link] += 1

    def _append_sample(self, sample: dict[str, Any]) -> None:
        """Append a sample to the dataset."""
        with jsonlines.open(self.dataset_path, mode="a") as writer:
            writer.write(sample)

        self.dataset_length += 1

    def _save_link_expansion_count(self) -> None:
        """Save the link expansion count to disk."""
        self._dump(dict(self.link_expansion_count), self.link_expansion_path)

    def _read_processed_data(self, processed_dir: Path) -> None:
        """Folder as: data/processed/2025-08-12-13-08-36."""
        date_str_dir = not processed_dir.name == "processed"
        if not date_str_dir:
            processed_dir = self._extract_latest_processed_dir(processed_dir)

        title_to_links_path = processed_dir / FILE_NAMES["title_to_links"]
        link_to_freq_path = processed_dir / FILE_NAMES["link_to_freq"]
        title_to_num_tokens_path = processed_dir / FILE_NAMES["title_to_num_tokens"]
        articles_db_path = processed_dir / FILE_NAMES["articles_db"]

        self.title_to_links = self._load_json(title_to_links_path)

        if not articles_db_path.exists():
            msg = f"articles_db not found at: {articles_db_path}"
            raise FileNotFoundError(msg)

        self._articles_db_conn = sqlite3.connect(articles_db_path)

        self.link_to_freq = Counter(self._load_json(link_to_freq_path))
        self.title_to_num_tokens = self._load_json(title_to_num_tokens_path)

    def _extract_latest_processed_dir(self, processed_dir: Path) -> Path:
        """Extract the latest folder with required dictionaries."""
        subdirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 0:
            msg = "No processed data found"
            raise ValueError(msg)
        if len(subdirs) == 1:
            return subdirs[0]
        return max(subdirs, key=lambda d: d.name)

    def _expand(self, text: str, title: str) -> dict[str, Any]:
        """Expand the text of a Wikipedia article.

        Expand links iteratively until the text reaches `max_tokens` tokens
        (if specified) or all available links are exhausted.

        Args:
            text: The text of the Wikipedia article.
            title: The title of the Wikipedia article.

        Returns:
            A sample for the expanded Wikipedia dataset.
        """
        links = self._get_links(title=title)

        current_tokens: int = self.title_to_num_tokens[title]
        n_links_expanded: int = 0
        links_expanded: list[str] = []
        while current_tokens < self.max_tokens and (
            self.max_link_expansions_local is None
            or n_links_expanded < self.max_link_expansions_local
        ):
            if not links:
                break
            link = links.pop()

            if not self._article_exists(link):
                continue

            n_links_expanded += 1
            link_article = self._get_article_text(link)
            links_expanded.append(link)
            current_tokens += self.title_to_num_tokens[link]
            text = self._include_link_text(link_article=link_article, text=text)

        sample = {
            "title": title,
            "n_tokens": current_tokens,
            "links_expanded": links_expanded,
            "n_links_expanded": n_links_expanded,
            "expanded_text": text,
        }
        return sample

    def _include_link_text(self, link_article: str, text: str) -> str:
        """Include the link text in the text."""
        if self.include_strategy == "append":
            text = f"{text}\n\n{link_article}"
        elif self.include_strategy == "prepend":
            text = f"{link_article}\n\n{text}"
        else:
            msg = f"Invalid include strategy: {self.include_strategy}"
            raise ValueError(msg)
        return text

    def _article_exists(self, title: str) -> bool:
        """Return True if we have text available for the given article."""
        if self._articles_db_conn is None:
            msg = "articles_db connection is not initialised."
            raise RuntimeError(msg)

        cursor = self._articles_db_conn.execute(
            "SELECT 1 FROM articles WHERE title = ? LIMIT 1", (title,)
        )
        return cursor.fetchone() is not None

    def _get_article_text(self, title: str) -> str:
        """Fetch article text from SQLite or the in-memory dict."""
        if self._articles_db_conn is None:
            msg = "articles_db connection is not initialised."
            raise RuntimeError(msg)

        cursor = self._articles_db_conn.execute(
            "SELECT text FROM articles WHERE title = ?", (title,)
        )
        row = cursor.fetchone()
        if row is None:
            msg = f"Missing article text for title: {title}"
            raise KeyError(msg)
        (text,) = row
        return str(text)

    def _get_links(self, title: str) -> list[str]:
        """Get the links present in the article."""
        links: list[str] = list(set(self.title_to_links[title]))
        links = [link for link in links if self._article_exists(link)]

        if self.max_link_expansions_global:
            links = [
                link
                for link in links
                if self.link_expansion_count[link] < self.max_link_expansions_global
            ]

        links = self._prioritize_links(links=links)

        return links

    def _prioritize_links(self, links: list[str]) -> list[str]:
        """Queue links by expansion count, frequency, and token length.

        Links with low expansion count, low frequency, and few tokens are prioritized.
        """
        if not links:
            return links

        # If a link was not expanded in the first iteration,
        # and has not yet been expanded, prioritize it.
        to_prioritize: list[bool] = [
            link in self.links_considered_in_first_iteration_but_not_expanded
            and self.link_expansion_count[link] == 0
            for link in links
        ]

        expansion_counts: list[int] = [
            self.link_expansion_count[link] for link in links
        ]
        frequencies: list[int] = [self.link_to_freq[link] for link in links]
        num_tokens: list[int] = [self.title_to_num_tokens[link] for link in links]
        items: list[tuple[str, bool, int, int, int]] = list(
            zip(links, to_prioritize, expansion_counts, frequencies, num_tokens)
        )

        # Sort by:
        #   (not_expanded_in_first_iteration descending,
        #    expansion_count ascending,
        #    frequency ascending,
        #    num_tokens descending)
        # This prioritizes:
        #   links not expanded in first iteration,
        #   less expanded links,
        #   less frequent links,
        #   fewer tokens
        items = sorted(
            items,
            key=lambda item: (not item[1], item[2], item[3], item[4]),
            reverse=True,
        )

        links = [item[0] for item in items]
        return links

    def _log_iteration_stats(self, iteration: int) -> None:
        """Log the iteration stats."""
        logger.info(
            f"Total number of unique expanded links (iteration {iteration}): "
            f"{len(self.link_expansion_count)}"
        )
        logger.info(
            f"Total number of links expanded (iteration {iteration}): "
            f"{sum(self.link_expansion_count.values())}"
        )
        logger.info(
            "Total number of links considered for expansion: "
            f"{len(self.all_links_considered_for_expansion)}"
        )

    @staticmethod
    def _dump(data: dict, path: Path) -> None:
        """Dump a Python object to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load a JSON file and return the decoded Python object."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
