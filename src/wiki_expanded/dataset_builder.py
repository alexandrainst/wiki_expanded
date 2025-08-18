"""Builds the Expanded Wikipedia dataset."""

import datetime
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
    """Build the expanded Wikipedia dataset.

    Args:
        processed_dir (Path): The directory containing the processed dictionaries
            from running `src/scripts/process.py`.
        save_dir (Path): The path to save the expanded dataset.
        num_tokens_threshold (int): Stop expanding links when the text is at least
            this many tokens long.
        max_dataset_length (int, optional): The maximum number of samples in the
            expanded dataset. If None, use all.
        max_link_expansions (int, optional): The maximum number of times a link can be
            expanded. If None, no limit.
        include_strategy (str, optional): The strategy to use for including the link
            text in the expanded text. If "prepend", the link text is prepended to the
            text. If "append", the link text is appended to the text.
        ignore_short_samples (bool, optional): Ignore samples with fewer than
            `num_tokens_threshold` tokens.
    """

    PROGRESS_LOG_INTERVAL = 10000

    def __init__(
        self,
        processed_dir: Path,
        save_dir: Path,
        num_tokens_threshold: int,
        max_dataset_length: int | None = None,
        max_link_expansions: int | None = None,
        include_strategy: str = "prepend",
        ignore_short_samples: bool = False,
    ) -> None:
        """Initialize the DatasetBuilder."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.title_to_tokens: dict[str, list[int]] = {}
        self.title_to_num_tokens: dict[str, int] = {}
        self.num_tokens_threshold: int = num_tokens_threshold
        self.save_dir: Path = save_dir / datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.link_expansion_count: Counter[str] = Counter()
        self.files_expanded: int = 0
        self.dataset_length: int = 0
        self.max_dataset_length: int | None = max_dataset_length
        self.max_link_expansions: int | None = max_link_expansions
        self.include_strategy: str = include_strategy
        self.ignore_short_samples: bool = ignore_short_samples

        self.links_considered_in_first_iteration_but_not_expanded: set[str] = set()
        self.all_links_considered_for_expansion: set[str] = set()

        self.dataset_path: Path = self.save_dir / FILE_NAMES["dataset"]
        self.link_expansion_path: Path = (
            self.save_dir / FILE_NAMES["link_expansion_count"]
        )
        self._read_processed_data(processed_dir=processed_dir)

    def build_expanded_dataset(self) -> None:
        """Build the expanded Wikipedia dataset."""
        for iteration in range(2):
            self._init_dataset()
            self._build_dataset(iteration=iteration)

            if iteration == 0:
                self.links_considered_in_first_iteration_but_not_expanded = (
                    self.all_links_considered_for_expansion
                    - set(self.link_expansion_count.keys())
                )

            logger.info(
                f"Total number of unique expanded links "
                f"(iteration {iteration}): "
                f"{len(self.link_expansion_count)}"
            )
            logger.info(
                f"Total number of links expanded "
                f"(iteration {iteration}): "
                f"{sum(self.link_expansion_count.values())}"
            )
            logger.info(
                f"Total number of links considered for expansion: "
                f"{len(self.all_links_considered_for_expansion)}"
            )

        self._save_link_expansion_count()
        logger.info(f"Done building dataset. Saved {self.dataset_length} samples.")

    def _init_dataset(self) -> None:
        self.link_expansion_count = Counter()
        self.files_expanded = 0
        self.dataset_length = 0

        if not self.dataset_path.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_path.touch()

    def _build_dataset(self, iteration: int) -> None:
        n_titles = len(self.title_to_text)
        for i, (title, text) in enumerate(self.title_to_text.items()):
            if not i % self.PROGRESS_LOG_INTERVAL:
                logger.info(f"Processed {i}/{n_titles} titles")

            sample = self._expand(title=title, text=text)
            if (
                self.ignore_short_samples
                and sample["n_tokens"] < self.num_tokens_threshold
            ):
                continue

            if iteration == 0:
                links = self.title_to_links[title]
                self.all_links_considered_for_expansion.update(links)

            self._track_expanded_links(links=sample["links_expanded"])
            if iteration == 1:
                self._append_sample(sample=sample)

            self.dataset_length += 1
            dataset_done = self.max_dataset_length and (
                self.dataset_length >= self.max_dataset_length
            )
            if dataset_done:
                break

    def _find_links_not_expanded_in_iteration(self) -> set[str]:
        """Find links that were not expanded in the first iteration.

        These will be prioritized in the second iteration.
        """
        links = [
            link
            for link in self.link_to_freq
            if link in self.title_to_text and link not in self.link_expansion_count
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

    def _save_link_expansion_count(self) -> None:
        """Save the link expansion count to disk."""
        with open(self.link_expansion_path, "w", encoding="utf-8") as f:
            json.dump(dict(self.link_expansion_count), f, ensure_ascii=False, indent=2)

    def _read_processed_data(self, processed_dir: Path) -> None:
        """Folder as: data/processed/2025-08-12-13-08-36."""
        date_str_dir = not processed_dir.name == "processed"
        if not date_str_dir:
            processed_dir = self._extract_latest_processed_dir(processed_dir)

        title_to_links_path = processed_dir / FILE_NAMES["title_to_links"]
        title_to_text_path = processed_dir / FILE_NAMES["title_to_text"]
        link_to_freq_path = processed_dir / FILE_NAMES["link_to_freq"]
        title_to_tokens_path = processed_dir / FILE_NAMES["title_to_tokens"]
        with open(title_to_links_path, "r") as f:
            self.title_to_links = json.load(f)
        with open(title_to_text_path, "r") as f:
            self.title_to_text = json.load(f)
        with open(link_to_freq_path, "r") as f:
            self.link_to_freq = json.load(f)
        with open(title_to_tokens_path, "r") as f:
            self.title_to_tokens = json.load(f)

        for title, tokens in self.title_to_tokens.items():
            self.title_to_num_tokens[title] = len(tokens)

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

        Expand links iteratively until the text is at least
        `num_tokens_threshold` tokens long.

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
        while current_tokens < self.num_tokens_threshold:
            if not links:
                break
            link = links.pop()

            n_links_expanded += 1
            link_article = self.title_to_text[link]
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
        """Include the link text in the text.

        Args:
            link_article: The text of the link article.
            text: The text of the Wikipedia article.

        Returns:
            The text of the Wikipedia article with the link text included.
        """
        if self.include_strategy == "append":
            text = f"{text}\n\n{link_article}"
        elif self.include_strategy == "prepend":
            text = f"{link_article}\n\n{text}"
        else:
            raise ValueError(f"Invalid include strategy: {self.include_strategy}")
        return text

    def _get_links(self, title: str) -> list[str]:
        """Get the links present in the article.

        Args:
            title: The title of the Wikipedia article.

        Returns:
            A list of links present in the article.
        """
        links: list[str] = self.title_to_links[title]
        links = list(set([link for link in links if link in self.title_to_text]))

        if self.max_link_expansions:
            links = [
                link
                for link in links
                if self.link_expansion_count[link] < self.max_link_expansions
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
