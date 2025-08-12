"""Processes raw Wikipedia articles to build dependencies dictionaries."""

import datetime
import html
import json
import logging
import re
import urllib.parse
from collections import Counter
from pathlib import Path

from .constants import FILE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="processing.log",
)

logger = logging.getLogger(__name__)


class Processor:
    """Processes raw Wikipedia article files to build four dictionaries.

    Dictionaries built that will be used to build the Wiki Expanded dataset:
    1. title_to_text: Maps each article title to its text.
    2. title_to_links: Maps each article title to the links found in the article.
    3. link_to_freq: Maps each link to the number of articles it appears in.
    4. titles_lowered_to_original: Maps each lowercase title to the original title.

    Args:
        text_dir (Path): Directory containing raw article text files.
        save_dir (Path): Directory to save the processed dictionaries.
        max_files (int, optional): Maximum number of files to process.
            If None, process all files.
    """

    def __init__(
        self, text_dir: Path, save_dir: Path, max_files: int | None = None
    ) -> None:
        """Initialize the Processor."""
        self.title_to_links: dict[str, list[str]] = {}
        self.title_to_text: dict[str, str] = {}
        self.link_to_freq: Counter[str] = Counter()
        self.titles_lowered_to_original: dict[str, str] = {}
        self.titles_seen: set[str] = set()
        self.files_processed: int = 0
        self.text_dir: Path = text_dir
        self.save_dir: Path = save_dir
        self.max_files: int | None = max_files

    def process(self) -> None:
        """Build the four dictionaries."""
        for path in self.text_dir.glob("*/wiki_*"):
            ids = self._get_all_ids(path=path)

            for id in ids:
                title, text = self._read_article(path=path, id=id)
                self.files_processed += 1
                if not self.files_processed % 10000:
                    logger.info(f"Processed {self.files_processed} files")

                if not title or title.lower() in self.titles_seen:
                    continue

                self.titles_lowered_to_original[title.lower()] = title
                title = title.lower()

                self.titles_seen.add(title)
                links = self._extract_links(text=text)
                self.link_to_freq.update(links)
                self.title_to_links[title] = links
                text_processed = self._process_text(text=text)
                self.title_to_text[title] = text_processed
                if self.max_files and self.files_processed >= self.max_files:
                    self._save_to_disk(save_dir=self.save_dir)
                    return

        self._save_to_disk(save_dir=self.save_dir)

    def _save_to_disk(self, save_dir: Path) -> None:
        """Save the dictionaries to disk."""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = save_dir / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        title_to_links_path = save_dir / FILE_NAMES["title_to_links"]
        title_to_text_path = save_dir / FILE_NAMES["title_to_text"]
        link_to_freq_path = save_dir / FILE_NAMES["link_to_freq"]
        titles_lowered_to_original_path = (
            save_dir / FILE_NAMES["titles_lowered_to_original"]
        )

        with open(title_to_links_path, "w") as f:
            json.dump(self.title_to_links, f)
        with open(title_to_text_path, "w") as f:
            json.dump(self.title_to_text, f)
        with open(link_to_freq_path, "w") as f:
            json.dump(self.link_to_freq, f)
        with open(titles_lowered_to_original_path, "w") as f:
            json.dump(self.titles_lowered_to_original, f)

    @staticmethod
    def _extract_links(text: str) -> list[str]:
        """Extract links from a Wikipedia article.

        Args:
            text: The text of the Wikipedia article.

        Returns:
            A list of links found in the article.
        """
        text = html.unescape(text)
        links = re.findall(r'<a href="([^"]+)">.+?</a>', text)
        decoded_links = [urllib.parse.unquote(link) for link in links]

        def _ignore_link(link_title: str) -> bool:
            # Ignore links that are too short. Can be articles about a single letter
            if len(link_title) <= 1:
                return True

            # Ignore year patterns
            if re.match(r"^\d{4}$", link_title):
                return True

            # Ignore date patterns like "10. september"
            if re.match(r"^\d{1,2}\.\s+\w+$", link_title):
                return True

            return False

        return [
            link.lower() for link in decoded_links if not _ignore_link(link_title=link)
        ]

    @staticmethod
    def _get_all_ids(path: Path) -> list[int]:
        """Get all the ids of the articles in the Wikipedia index file.

        Args:
            path: The path to the Wikipedia index file.

        Returns:
            A list of ids of the articles in the Wikipedia index file.
        """
        with open(path, "r", encoding="utf-8") as f:
            wiki_index_text = f.read()
        matches = re.findall(r'<doc id="(\d+)"', wiki_index_text)
        return [int(match) for match in matches]

    @staticmethod
    def _read_article(path: Path, id: int) -> tuple[str, str]:
        """Extract a Wikipedia article.

        Args:
            path: The path to the Wikipedia index file.
            id: The id of the article to extract.

        Returns:
            A tuple containing the title and text of the article.
        """
        with open(path, "r", encoding="utf-8") as f:
            wiki_index_text = f.read()

        pattern = rf'<doc id="{id}"[^>]*>(.*?)</doc>'
        match = re.search(pattern, wiki_index_text, re.DOTALL)
        if match:
            text = match.group(1).strip()
            no_content = "\n\n" not in text
            if no_content:
                return "", ""

            title, text = text.split("\n\n", 1)

            def _make_text_human_readable(text: str) -> str:
                text = html.unescape(text)
                text = urllib.parse.unquote(text)
                return text

            text = _make_text_human_readable(text=text)
            return title, text
        else:
            raise ValueError(f"Article with id {id} not found in {path}")

    @staticmethod
    def _process_text(text: str) -> str:
        """Process the text of a Wikipedia article.

        Args:
            text: The text of the Wikipedia article.

        Returns:
            The processed text of the Wikipedia article.
        """

        def _remove_html_links(text: str) -> str:
            text = re.sub(r'<a href="[^"]*">(.*?)</a>', r"\1", text)
            return text

        def _remove_css_styling(lines: list[str]) -> list[str]:
            if lines[-1] == '<templatestyles src="Reflist/styles.css" />':
                lines.pop()
            return lines

        def _remove_trailing_header(lines: list[str]) -> list[str]:
            last_section = lines[-1]
            is_header = len(last_section.split(" ")) <= 2
            if is_header:
                lines.pop()
            return lines

        text = _remove_html_links(text=text)
        lines = text.split("\n")
        lines = _remove_css_styling(lines=lines)
        lines = _remove_trailing_header(lines=lines)

        text = "\n".join(lines)
        return text
