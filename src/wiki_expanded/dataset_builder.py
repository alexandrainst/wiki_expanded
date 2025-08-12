"""Builds the Expanded Wikipedia dataset."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import jsonlines

from .constants import FILE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="dataset_builder.log",
)

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds the Wiki Expanded dataset.

    Args:
        processed_dir (Path): The directory containing the processed dictionaries
            from running `src/scripts/process.py`.
        dataset_dir_path (Path): The path to save the expanded dataset.
        max_words (int): The maximum number of words in the expanded text.
        max_dataset_length (int, optional): The maximum number of samples in the
            expanded dataset. If None, use all.
    """

    def __init__(
        self,
        processed_dir: Path,
        dataset_dir_path: Path,
        max_words: int,
        max_dataset_length: int | None = None,
    ) -> None:
        """Initialize the DatasetBuilder."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.link_to_freq: Counter[str] = Counter()

        self.dataset_dir_path: Path = dataset_dir_path
        self.dataset: list[dict[str, Any]] = []
        self.titles_in_dataset: set[str] = set()
        self.link_expansion_count: Counter[str] = Counter()
        self.files_expanded: int = 0
        self.max_words: int = max_words
        self.dataset_length: int = 0
        self.max_dataset_length: int | None = max_dataset_length
        self._read_processed_data(processed_dir=processed_dir)

    def build_expanded_dataset(self) -> None:
        """Build the Expanded Wikipedia dataset."""
        for title, text in self.title_to_text.items():
            if title in self.titles_in_dataset:
                continue
            self.titles_in_dataset.add(title)

            sample = self._expand(title=title, text=text)
            self.dataset.append(sample)

            if (
                self.max_dataset_length
                and self.dataset_length >= self.max_dataset_length
            ):
                self._save_to_disk()
                return

        self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Save the dataset and link expansion count to disk."""
        dataset_dir_path = self.dataset_dir_path.parent / FILE_NAMES["dataset"]
        dataset_dir_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(dataset_dir_path, mode="w") as writer:
            writer.write_all(self.dataset)

        link_expansion_path = (
            dataset_dir_path.parent / FILE_NAMES["link_expansion_count"]
        )
        with open(link_expansion_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.link_expansion_count), f, ensure_ascii=False, indent=2)

    def _read_processed_data(self, processed_dir: Path) -> None:
        """Folder as: data/processed/2025-08-12-13-08-36."""
        date_str_dir = not processed_dir.name == "processed"
        if not date_str_dir:
            processed_dir = self._extract_latest_processed_dir(processed_dir)

        title_to_links_path = processed_dir / FILE_NAMES["title_to_links"]
        title_to_text_path = processed_dir / FILE_NAMES["title_to_text"]
        link_to_freq_path = processed_dir / FILE_NAMES["link_to_freq"]

        with open(title_to_links_path, "r") as f:
            self.title_to_links = json.load(f)
        with open(title_to_text_path, "r") as f:
            self.title_to_text = json.load(f)
        with open(link_to_freq_path, "r") as f:
            self.link_to_freq = json.load(f)

    def _extract_latest_processed_dir(self, processed_dir: Path) -> Path:
        """Extract the latest folder with required dictionaries."""
        subdirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 0:
            raise ValueError("No processed data found")
        elif len(subdirs) == 1:
            return subdirs[0]
        else:
            return max(subdirs, key=lambda d: d.name)

    def _expand(self, text: str, title: str) -> dict[str, Any]:
        """Expand the text of a Wikipedia article.

        Expand links iteratively until the text is at most `max_words` words long.

        Args:
            text: The text of the Wikipedia article.
            title: The title of the Wikipedia article.

        Returns:
            A sample for the Wiki Expanded dataset.
        """
        links = self._get_links(title=title)
        n_words = len(text.split())
        n_links_expanded = 0
        while len(text.split()) < self.max_words:
            link = links.pop(0)
            self.link_expansion_count[link] += 1

            link_article = f"\n\n{self.title_to_text[link]}"
            n_words += len(link_article.split())
            text += link_article

        sample = {
            "title": title,
            "expanded_text": text,
            "n_words": n_words,
            "n_links_expanded": n_links_expanded,
        }
        return sample

    def _get_links(self, title: str) -> list[str]:
        """Get the links present in the article.

        Args:
            title: The title of the Wikipedia article.

        Returns:
            A list of links present in the article.
        """
        links = self.title_to_links[title]
        links = [link for link in links if link in self.title_to_text]
        links = sorted(links, key=lambda link: self.link_to_freq[link], reverse=True)
        return links
