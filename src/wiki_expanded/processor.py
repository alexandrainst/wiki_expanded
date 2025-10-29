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
        capitalize_titles_and_links (bool): Whether to capitalize titles
            and links or not.
        redirects_path (Path): Path to the title redirects file.
    """

    PROGRESS_LOG_INTERVAL = 10000

    _HTML_LINK_PATTERN = re.compile(r'<a href="[^"]*">(.*?)</a>')
    _SQUARE_BRACKET_LINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")

    _EMPTY_SQUARE_BRACKET_PATTERN = re.compile(r"\n\[\]")

    _TEMPLATESTYLES_PATTERN = re.compile(r"<templatestyles.+?>", re.IGNORECASE)
    _SPOILER_PATTERN = re.compile(
        r" - \"Handling, afslutning og/eller plot afsløres i det følgende.\"?",
        re.IGNORECASE,
    )
    _REMOVE_PATTERNS = [_TEMPLATESTYLES_PATTERN, _SPOILER_PATTERN]

    # Remove files, images, etc.
    # Also removes nested files as "[[Fil:Gettysburg national cemetery img
    # 4164.jpg|thumb|right|[[Gettysburg National Cemetery]]]]"
    _FILE_PATTERN = re.compile(
        r"\[\[(Fil|Billede|Image):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", re.IGNORECASE
    )
    _IMAGE_LABEL = re.compile(r"\{\{Image label.+?\}\}")

    _CENTURY_PATTERN = re.compile(r"^\d+\.\s*århundrede f\.Kr\.$", re.IGNORECASE)
    _YEAR_ERNE_PATTERN = re.compile(r"^\d{4}'erne")
    _YEAR_BEFORE_CR_PATTERN = re.compile(r"^\d+\s*f\.Kr\.$")
    _DATE_PATTERN = re.compile(r"^\d{1,2}\.\s+\w+$")
    _ONLY_NUMBERS_PATTERN = re.compile(r"^\d+$")
    _TITLE_IGNORE_SUB_STRINGS = ["(flertydig)"]

    # Ignore texts that start with these strings
    _UNWANTED_TEXTS = [
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
        "henviser til flere artikler:",
        "henfører primært til:",
        "er navnet på flere personer:",
        "Der er flere personer med",
        "henvise til flere personer:(flertydig).",
    ]

    # Pop initial lines if they start with these strings
    _UNWANTED_FIRST_LINES = [
        "Ikke at forveksle med",
        "Denne artikel handler om",
        "Denne artikel omhandler",
        "Uddybende artikel:",
        "Uddybende artikler:",
        "Hovedartikel:",
        "For alternative betydninger, se",
        "Denne artikel bør gennemlæses af en person",
        "Se også:",
        "Tekst mangler,",
        "Indledende runde",
    ]

    _SECTIONS_TO_IGNORE = [
        "Noter\n",
        "Se også\n",
        "Litteratur\n",
        "Referencer\n",
        "Kilder\n",
        "Kilde\n",
        "Eksterne henvisninger\n",
        "Ekstern henvisning\n",
        "Yderligere læsning\n",
        "Kilder og henvisninger\n",
    ]

    def __init__(
        self,
        text_dir: Path,
        save_dir: Path,
        max_files: int | None = None,
        tokenizer_name: str = "google/gemma-7b",
        capitalize_titles_and_links: bool = True,
        redirects_path: Path = Path("data/raw/redirects.json"),
    ) -> None:
        """Initialize the Processor."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.title_to_tokens: dict[str, list[int]] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.original_title: dict[str, str] = {}

        with open(redirects_path, "r") as f:
            self.redirects: dict[str, str] = json.load(f)

        self.titles_seen: set[str] = set()
        self.articles_processed: int = 0
        self.max_files: int | None = max_files

        self.text_dir: Path = text_dir
        self.save_dir: Path = save_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.capitalize_titles_and_links: bool = capitalize_titles_and_links

    def process(self, path: Path | None = None) -> None:
        """Build the five dictionaries.

        Args:
            path: Path to a specific wiki file to process. If None, process all files.
        """
        paths: list[Path] = [path] if path else list(self.text_dir.glob("*/wiki_*"))
        for path in paths:
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

                text_processed: str = self._remove_noise(text=text)
                if not text_processed:
                    continue

                text_processed = self._to_markdown(text=text_processed, title=title)

                if title in self.titles_seen:
                    # For duplicates, prioritize the longer text
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
        # Extract HTML-style links
        html_links = self._HTML_LINK_PATTERN.findall(text)
        # Extract square bracket links and remove the brackets
        square_bracket_links = self._SQUARE_BRACKET_LINK_PATTERN.findall(text)
        # Combine both types of links and ignore duplicates
        links = list(set(html_links) | set(square_bracket_links))
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

        links = list(set(links))
        return links

    @staticmethod
    def _capitalize(string: str) -> str:
        """Capitalize the first letter of a string."""
        if not string:
            return string

        return string[0].upper() + string[1:]

    def _ignore_title(self, title: str) -> bool:
        """Ignore titles based on various criteria."""
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

        if any(
            ignore_substring in title
            for ignore_substring in self._TITLE_IGNORE_SUB_STRINGS
        ):
            return True

        return False

    def _ignore_text(self, text: str) -> bool:
        """Return True if text should be ignored based on unwanted starting lines."""
        lines = text.split("\n", 3)[:3]
        return any(
            any(unwanted in line for unwanted in self._UNWANTED_TEXTS) for line in lines
        )

    def _remove_noise(self, text: str) -> str:
        """Clean the text of a Wikipedia article.

        Args:
            text: The text of the Wikipedia article.

        Returns:
            The cleaned text of the Wikipedia article.
        """

        def _remove_empty_square_brackets(text: str) -> str:
            return self._EMPTY_SQUARE_BRACKET_PATTERN.sub("", text)

        def _remove_files(text: str) -> str:
            return self._FILE_PATTERN.sub("", text)

        def _remove_image_labels(text: str) -> str:
            return self._IMAGE_LABEL.sub("", text)

        def _remove_redundant_startings(lines: list[str]) -> list[str]:
            if not lines:
                return lines

            for i in range(len(lines)):
                if not _ignore_line(line=lines[i]):
                    break

            return lines[i:]

        def _remove_redundant_subsection_startings(lines: list[str]) -> list[str]:
            ignore_indices = []
            for i in range(len(lines) - 1):
                line = lines[i]
                next_line = lines[i + 1]
                if line.startswith("##"):
                    if _ignore_line(line=next_line):
                        ignore_indices.append(i + 1)

            return [line for i, line in enumerate(lines) if i not in ignore_indices]

        def _ignore_line(line: str) -> bool:
            _empty_line: bool = not line.strip()
            _has_unwanted_starting: bool = any(
                unwanted in line.strip() for unwanted in self._UNWANTED_FIRST_LINES
            )
            return _empty_line or _has_unwanted_starting

        def _remove_html_links(text: str) -> str:
            text = self._HTML_LINK_PATTERN.sub(r"\1", text)
            return text

        def _remove_double_square_bracket_links(text: str) -> str:
            return self._SQUARE_BRACKET_LINK_PATTERN.sub(r"\1", text)

        def _remove_css_styling(text: str) -> str:
            """Remove CSS styling from the text.

            Remove `<templatestyles src="Reflist/styles.css" />`.
            """
            for pattern in self._REMOVE_PATTERNS:
                text = pattern.sub("", text)
            return text

        def _remove_empty_trailing_sections(lines: list[str]) -> list[str]:
            while lines and (
                not lines[-1].strip() or lines[-1].strip().startswith("##")
            ):
                lines.pop()
            return lines

        def _ignore_redundant_trailing_sections(text: str) -> str:
            sections = re.split(r"\n## ", text)

            if len(sections) == 1:
                return text

            for i in range(1, len(sections)):
                if any(
                    sections[i].startswith(section_to_ignore)
                    for section_to_ignore in self._SECTIONS_TO_IGNORE
                ):
                    break

            return sections[0] + "\n## " + "\n## ".join(sections[1:i])

        def _ignore_short_lines(lines: list[str]) -> list[str]:
            return [line for line in lines if len(line.strip()) > 1]

        def _remove_empty_sections(lines: list[str]) -> list[str]:
            while True:
                lines_copy = lines.copy()
                ignore_indices = []
                for i in range(len(lines) - 1):
                    line = lines[i]
                    next_line = lines[i + 1]
                    line_hash_tag_count = line.split(" ")[0].count("#")
                    next_line_hash_tag_count = next_line.split(" ")[0].count("#")
                    if not (line_hash_tag_count and next_line_hash_tag_count):
                        continue

                    if next_line_hash_tag_count <= line_hash_tag_count:
                        ignore_indices.append(i)
                lines = [
                    line for i, line in enumerate(lines) if i not in ignore_indices
                ]
                if lines_copy == lines:
                    break
            return lines

        text = _ignore_redundant_trailing_sections(text=text)
        text = _remove_empty_square_brackets(text=text)
        text = _remove_css_styling(text=text)
        text = _remove_files(text=text)
        text = _remove_image_labels(text=text)
        text = _remove_html_links(text=text)
        text = _remove_double_square_bracket_links(text=text)

        lines = text.split("\n")
        lines = _ignore_short_lines(lines=lines)
        lines = _remove_redundant_subsection_startings(lines=lines)
        lines = _remove_redundant_startings(lines=lines)
        lines = _remove_empty_trailing_sections(lines=lines)
        lines = _remove_empty_sections(lines=lines)

        text = "\n".join(lines)
        return text

    def _to_markdown(self, text: str, title: str) -> str:
        """Convert the text to markdown.

        Include the title as a markdown header
        Change wiki markup sections `==` to markdown headers `##`
        """
        text = re.sub(r"\n+", "\n", text)
        text = self._add_newlines_to_headers(text=text)
        text = f"# {title}\n\n{text}"
        # Replace 3 or more consecutive newlines with just two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _add_newlines_to_headers(self, text: str) -> str:
        """Add newlines to headers."""
        lines = text.split("\n")
        new_lines = []
        for i in range(len(lines)):
            if lines[i].startswith("#"):
                line = f"\n{lines[i]}\n"
                new_lines.append(line)
            else:
                new_lines.append(lines[i])
        return "\n".join(new_lines)

    def _make_human_readable(self, string: str) -> str:
        """Make a string human readable.

        Args:
            string: The string to make human readable.

        Returns:
            The human readable string.
        """
        string = html.unescape(string)
        string = urllib.parse.unquote(string)
        string = self.fix_malformed_html_entities(text=string)
        return string

    def fix_malformed_html_entities(self, text: str) -> str:
        """Fix malformed HTML entities."""
        text = text.replace("amp;", "")
        return text

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file."""
        with jsonlines.open(path, mode="r") as reader:
            return list(reader)

    def _missing_data(self, title: str, text: str) -> bool:
        """Check if the title and text are missing."""
        return not title or not text
