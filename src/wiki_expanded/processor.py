"""Processes raw Wikipedia articles to build dependencies dictionaries."""

import datetime
import html
import json
import logging
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

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
    """Processes raw Wikipedia article files to build five dictionaries.

    Dictionaries built that will be used to build the expanded Wikipedia dataset:
    1. title_to_text: Maps each article title to its text.
    2. title_to_links: Maps each article title to the links found in the article.
    3. link_to_freq: Maps each link to the number of articles it appears in.
    4. original_title: Maps each lowercase title to the original title.
    5. title_to_tokens: Maps each article title to its tokenized content.

    Args:
        text_dir (Path): Directory containing raw article text files.
        save_dir (Path): Directory to save the processed dictionaries.
        max_files (int, optional): Maximum number of files to process.
            If None, process all files.
        tokenizer_name (str, optional): Name of the tokenizer to use.
            Defaults to "google/gemma-7b".
        capitalize_titles_and_links (bool): Wether to capitalize titles
            and links or not.
    """

    PROGRESS_LOG_INTERVAL = 10000

    _LINK_PATTERN = re.compile(r'<a href="([^"]+)">.+?</a>')
    _CENTURY_PATTERN = re.compile(r"^\d+\.\s*århundrede f\.Kr\.$", re.IGNORECASE)
    _YEAR_ERNE_PATTERN = re.compile(r"^\d{4}'erne")
    _YEAR_BEFORE_CR_PATTERN = re.compile(r"^\d+\s*f\.Kr\.$")
    _DATE_PATTERN = re.compile(r"^\d{1,2}\.\s+\w+$")
    _ONLY_NUMBERS_PATTERN = re.compile(r"^\d+$")

    def __init__(
        self,
        text_dir: Path,
        save_dir: Path,
        max_files: int | None = None,
        tokenizer_name: str = "google/gemma-7b",
        capitalize_titles_and_links: bool = True,
        redirects_path: Path = Path("data/raw/redirects.json"),  # Add parameter
    ) -> None:
        """Initialize the Processor."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.title_to_tokens: dict[str, list[int]] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.original_title: dict[str, str] = {}
        self.redirects: dict[str, str] = json.load(open(redirects_path, "r"))
        self.titles_seen: set[str] = set()
        self.articles_processed: int = 0
        self.text_dir: Path = text_dir
        self.save_dir: Path = save_dir
        self.max_files: int | None = max_files
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.capitalize_titles_and_links: bool = capitalize_titles_and_links

    def process(self) -> None:
        """Build the five dictionaries."""
        for path in self.text_dir.glob("*/wiki_*"):
            articles = self._read_jsonl(path=path)
            for article in articles:
                title, text = (
                    self._make_human_readable(string=article["title"]),
                    self._make_human_readable(string=article["text"]),
                )

                self.articles_processed += 1
                if not self.articles_processed % self.PROGRESS_LOG_INTERVAL:
                    logger.info(f"Processed {self.articles_processed} articles.")

                original_title = title
                title = (
                    self._capitalize(title)
                    if self.capitalize_titles_and_links
                    else title
                )

                title = self.redirects[title] if title in self.redirects else title

                if (
                    self._missing_data(title=title, text=text)
                    or self._ignore_title(title=title)
                    or self._ignore_text(text=text)
                ):
                    continue

                text_processed = self._process_text(text=text)
                if not text_processed:
                    continue

                if title in self.titles_seen:
                    # There are title duplicates (given case insensitivity)
                    # Prioritize the longer text
                    # See for example https://da.wikipedia.org/wiki/Usa
                    old_text_processed = self.title_to_text[title]
                    override: bool = len(text_processed) > len(old_text_processed)
                    if not override:
                        continue

                    logger.info(
                        f"Overriding '{self.original_title[title]}' with '{title}'.\n"
                        f"Previous text head: {old_text_processed[:100]!r}\n"
                        f"New text head:      {text_processed[:100]!r}"
                    )
                    # Before overriding, decrement the link counts from
                    # previous article/title
                    for link in self.title_to_links[title]:
                        self.link_to_freq[link] -= 1

                self.titles_seen.add(title)

                self.original_title[title] = original_title

                links = self._extract_links(text=text)
                self.title_to_links[title] = links
                self.link_to_freq.update(links)

                self.title_to_text[title] = text_processed

                tokens = self._tokenize_text(text=text_processed)
                self.title_to_tokens[title] = tokens

                if self.max_files and self.articles_processed >= self.max_files:
                    self._save_to_disk()
                    return

        self._save_to_disk()
        logger.info(f"Done processing {self.articles_processed} articles.")

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
        title_to_text_path = save_dir / FILE_NAMES["title_to_text"]
        title_to_tokens_path = save_dir / FILE_NAMES["title_to_tokens"]
        link_to_freq_path = save_dir / FILE_NAMES["link_to_freq"]
        original_title_path = save_dir / FILE_NAMES["original_title"]

        with open(title_to_links_path, "w") as f:
            json.dump(self.title_to_links, f)
        with open(title_to_text_path, "w") as f:
            json.dump(self.title_to_text, f)
        with open(title_to_tokens_path, "w") as f:
            json.dump(self.title_to_tokens, f)
        with open(link_to_freq_path, "w") as f:
            json.dump(self.link_to_freq, f)
        with open(original_title_path, "w") as f:
            json.dump(self.original_title, f)

    def _extract_links(self, text: str) -> list[str]:
        """Extract links from a Wikipedia article.

        Args:
            text: The text of the Wikipedia article.

        Returns:
            A list of links found in the article.
        """
        text = html.unescape(text)
        links = list(set(self._LINK_PATTERN.findall(text)))
        decoded_links = [urllib.parse.unquote(link) for link in links]

        links = [
            self._make_human_readable(string=link)
            for link in decoded_links
            if not self._ignore_title(title=link)
        ]

        links = [
            self.redirects[link] if link in self.redirects else link for link in links
        ]

        if self.capitalize_titles_and_links:
            links = [self._capitalize(link) for link in links]
        return links

    @staticmethod
    def _capitalize(string: str) -> str:
        return string[0].upper() + string[1:]

    def _ignore_title(self, title: str) -> bool:
        # Ignore links that are too short. Can be articles about a single letter
        if len(title) <= 1:
            return True

        # Ignore links that are only numbers
        if self._ONLY_NUMBERS_PATTERN.match(title):
            return True

        # Ignore links are "18. århundrede f.Kr."
        if self._CENTURY_PATTERN.match(title):
            return True

        # Ignore link as "1950'erne"
        if self._YEAR_ERNE_PATTERN.match(title):
            return True

        # Ignore links as "39 f.Kr."
        if self._YEAR_BEFORE_CR_PATTERN.match(title):
            return True

        # Ignore date links as "10. september"
        if self._DATE_PATTERN.match(title):
            return True

        return False

    def _ignore_text(self, text: str) -> bool:
        """Return True if text should be ignored based on unwanted starting lines."""
        unwanted_starts = [
            "For alternative betydninger, se",
            "Dette er en , som omdirigerer fra en fejlskrivning",
            "omdirigeringer med denne skabelon",
            "omdirigeres hertil. For",
            "#REDIRECT",
            "refererer til flere ting:",
            "har flere betydninger:",
            "Opslagsordet har også en anden betydning",
            "kan have flere betydninger:",
            "kan betyde:",
            "Der er for få eller ingen i denne artikel",
            "kan henvise til flere artikler:",
            "henfører primært til:",
        ]
        first_line = text.split("\n", 1)[0]
        return any(unwanted in first_line for unwanted in unwanted_starts)

    @staticmethod
    def _process_text(text: str) -> str:
        """Process the text of a Wikipedia article.

        Args:
            text: The text of the Wikipedia article.

        Returns:
            The processed text of the Wikipedia article.
        """

        def _remove_redundant_startings(lines: list[str]) -> list[str]:
            if not lines:
                return lines
            unwanted_starts = ["Ikke at forveksle med"]
            _pop_it: bool = any(unwanted in lines[0] for unwanted in unwanted_starts)
            if _pop_it:
                lines.pop(0)
            return lines

        def _remove_html_links(text: str) -> str:
            text = re.sub(r'<a href="[^"]*">(.*?)</a>', r"\1", text)
            return text

        def _remove_css_styling(lines: list[str]) -> list[str]:
            if not lines:
                return lines
            if "<templatestyles" in lines[-1]:
                lines.pop()
            return lines

        def _remove_trailing_header(lines: list[str]) -> list[str]:
            if not lines:
                return lines
            last_section = lines[-1]
            is_header = len(last_section.split(" ")) <= 2
            if is_header:
                lines.pop()
            return lines

        text = _remove_html_links(text=text)
        lines = text.split("\n")
        lines = _remove_redundant_startings(lines=lines)
        lines = _remove_css_styling(lines=lines)
        lines = _remove_trailing_header(lines=lines)

        text = "\n".join(lines)
        return text

    @staticmethod
    def _make_human_readable(string: str) -> str:
        """Make a string human readable.

        Args:
            string: The string to make human readable.

        Returns:
            The human readable string.
        """
        string = html.unescape(string)
        string = urllib.parse.unquote(string)
        return string

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file."""
        with jsonlines.open(path, mode="r") as reader:
            return list(reader)

    def _missing_data(self, title: str, text: str) -> bool:
        """Check if the title and text are missing."""
        return not title or not text
