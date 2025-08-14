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
        link_priority_strategy (str, optional): The strategy to use for prioritizing
            links. If "length", prioritize links with the least number of tokens. If
            "frequency", prioritize links that appear in the least number of articles.
            If "length_mix_frequency", prioritize based on a combination of link length
            and frequency.
        penalty_multiplier (float, optional): The multiplier for the penalty for links
            that have already been expanded.
    """

    def __init__(
        self,
        processed_dir: Path,
        save_dir: Path,
        num_tokens_threshold: int,
        max_dataset_length: int | None = None,
        max_link_expansions: int | None = None,
        include_strategy: str = "prepend",
        ignore_short_samples: bool = False,
        link_priority_strategy: str = "length",
        penalty_multiplier: float = 0.0,
    ) -> None:
        """Initialize the DatasetBuilder."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.title_to_tokens: dict[str, list[str]] = {}
        self.title_to_num_tokens: dict[str, int] = {}
        self.num_tokens_threshold: int = num_tokens_threshold
        self.save_dir: Path = save_dir / datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        self.titles_in_dataset: set[str] = set()
        self.link_expansion_count: Counter[str] = Counter()
        self.files_expanded: int = 0
        self.dataset_length: int = 0
        self.max_dataset_length: int | None = max_dataset_length
        self.max_link_expansions: int | None = max_link_expansions
        self.include_strategy: str = include_strategy
        self.ignore_short_samples: bool = ignore_short_samples
        self.link_priority_strategy: str = link_priority_strategy
        self.penalty_multiplier: float = penalty_multiplier

        self.dataset_path: Path = self.save_dir / FILE_NAMES["dataset"]
        self.link_expansion_path: Path = (
            self.save_dir / FILE_NAMES["link_expansion_count"]
        )
        self._read_processed_data(processed_dir=processed_dir)

    def build_expanded_dataset(self) -> None:
        """Build the expanded Wikipedia dataset."""
        n_titles = len(self.title_to_text)
        for i, (title, text) in enumerate(self.title_to_text.items()):
            if not i % 1000:
                logger.info(f"Processed {i}/{n_titles} titles")

            sample = self._expand(title=title, text=text)
            if (
                self.ignore_short_samples
                and sample["n_tokens"] < self.num_tokens_threshold
            ):
                continue

            self._track_expanded_links(links=sample["links_expanded"])
            self._append_sample(sample=sample)

            dataset_done = (
                self.max_dataset_length
                and self.dataset_length >= self.max_dataset_length
            )
            if dataset_done:
                break

        self._save_link_expansion_count()

    def _track_expanded_links(self, links: list[str]) -> None:
        """Track the expanded links."""
        for link in links:
            self.link_expansion_count[link] += 1

    def _append_sample(self, sample: dict[str, Any]) -> None:
        """Append a sample to the dataset."""
        if not self.dataset_path.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_path.touch()

        with jsonlines.open(self.dataset_path, mode="a") as writer:
            writer.write(sample)

        self.dataset_length += 1

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

        n_words = len(text.split())
        n_tokens = self.title_to_num_tokens[title]
        n_links_expanded: int = 0
        links_expanded: list[str] = []
        while len(text.split()) < self.num_tokens_threshold:
            if not links:
                break
            link = links.pop()
            if (
                self.max_link_expansions
                and self.link_expansion_count[link] >= self.max_link_expansions
            ):
                continue

            n_links_expanded += 1
            link_article = self.title_to_text[link]
            links_expanded.append(link)
            n_words += len(link_article.split())
            n_tokens += self.title_to_num_tokens[link]
            text = self._include_link_text(link_article=link_article, text=text)

        sample = {
            "title": title,
            "n_words": n_words,
            "n_tokens": n_tokens,
            "links_expanded": links_expanded,
            "n_links_expanded": n_links_expanded,
            "expanded_text": text,
        }
        return sample

    def _include_link_text(self, link_article: str, text: str) -> str:
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
        links = self.title_to_links[title]
        links = list(set([link for link in links if link in self.title_to_text]))

        if self.link_priority_strategy == "length":
            links = sorted(
                links, key=lambda link: self.title_to_num_tokens[link], reverse=True
            )

        elif self.link_priority_strategy == "frequency":
            # Penalize links that have already been expanded
            # The penalty will be at most `penalty_multiplier * max_link_expansions`
            links = sorted(
                links,
                key=lambda link: self.link_to_freq[link]
                + self.penalty_multiplier * self.link_expansion_count[link],
                reverse=True,
            )
        elif self.link_priority_strategy == "length_mix_frequency":

            def _to_bucket(link: str) -> int:
                """Assign link to bucket based on its frequency and expansion penalty.

                Links that appear more frequently and have been expanded more times
                will be placed into higher-numbered buckets.

                Example of `items` after sorting wr.t. the tuple (bucket, num_tokens):
                    ('jorden', 14, 13156)
                    ('solen', 13, 14181)
                    ('stjernebillede', 4, 717)
                    ('stjerne', 3, 14476)
                    ('helium', 3, 9416)
                    ('ur', 1, 2761)
                    ('galakse', 1, 904)
                    ('parsec', 1, 570)
                    ('ionisering', 1, 70)
                    ('størrelsesklasse', 0, 1293)
                    ('absolut størrelsesklasse', 0, 958)
                    ('matematisk formel', 0, 210)
                    ('henrietta swan leavitt', 0, 151)
                    ('cepheus', 0, 133)
                    ('lysstyrke', 0, 92)

                In this case `lysstyrke` will be expanded first.

                Args:
                    link: The link title.

                Returns:
                    An integer representing the bucket for the link.
                """
                freq = self.link_to_freq[link]
                penalty = self.penalty_multiplier * self.link_expansion_count[link]
                value = freq + penalty
                bucket = (
                    int(value // self.penalty_multiplier)
                    if self.penalty_multiplier > 0
                    else 0
                )
                return bucket

            buckets = [_to_bucket(link) for link in links]
            num_tokens = [self.title_to_num_tokens[link] for link in links]
            items = list(zip(links, buckets, num_tokens))

            items = sorted(items, key=lambda items: (items[1], items[2]), reverse=True)

            links = [item[0] for item in items]
        else:
            raise ValueError(
                f"Invalid link priority strategy: {self.link_priority_strategy}"
            )

        return links
