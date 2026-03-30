"""Build a simple Wikipedia dataset with url, title, and text columns."""

import json
from pathlib import Path
from typing import Any

import jsonlines

from .constants import FILE_NAMES


class WikiArticlesBuilder:
    """Build a simple dataset from processed dictionaries.

    The output rows have the schema: url, title, text.

    Args:
        processed_dir: Path to an exact processed run folder containing
            title_to_text.json and title_to_url.json.
        output_file: Path to the output JSONL file.
    """

    def __init__(self, processed_dir: Path, output_file: Path) -> None:
        """Initialize the builder."""
        self.processed_dir = processed_dir
        self.output_file = output_file

    def build(self) -> None:
        """Build and save the dataset to disk."""
        title_to_text = self._load_dictionary(file_key="title_to_text")
        title_to_url = self._load_dictionary(file_key="title_to_url")
        self._validate_keys_match(
            title_to_text=title_to_text, title_to_url=title_to_url
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(self.output_file, mode="w") as writer:
            for title in sorted(title_to_text):
                writer.write(
                    {
                        "url": title_to_url[title],
                        "title": title,
                        "text": title_to_text[title],
                    }
                )

    def _load_dictionary(self, file_key: str) -> dict[str, Any]:
        """Load a required dictionary from the processed folder."""
        file_name = FILE_NAMES[file_key]
        file_path = self.processed_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required file in processed folder: {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary in {file_path}")
        return data

    def _validate_keys_match(
        self, title_to_text: dict[str, Any], title_to_url: dict[str, Any]
    ) -> None:
        """Validate that required dictionaries share exactly the same titles."""
        titles_text = set(title_to_text.keys())
        titles_url = set(title_to_url.keys())
        if titles_text != titles_url:
            missing_in_url = sorted(titles_text - titles_url)
            missing_in_text = sorted(titles_url - titles_text)
            raise ValueError(
                "Title key mismatch between title_to_text and title_to_url. "
                f"Missing in title_to_url: {missing_in_url[:5]}; "
                f"missing in title_to_text: {missing_in_text[:5]}"
            )

    @staticmethod
    def _remove_title_from_text(text: str) -> str:
        """Remove markdown title heading from text by splitting on double newline."""
        _, separator, remainder = text.partition("\n\n")
        if separator:
            return remainder
        return text
