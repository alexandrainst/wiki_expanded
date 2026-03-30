"""Build a simple Wikipedia dataset with url, title, and text columns.

Usage:
>>> python src/scripts/build_wiki_articles.py \
...   --processed-dir="data/processed/da" \
...   --output-file="dataset.jsonl"
"""

from pathlib import Path

import click

from wiki_expanded.constants import FILE_NAMES
from wiki_expanded.wiki_articles import WikiArticlesBuilder


def _resolve_processed_dir(processed_dir: Path) -> Path:
    """Resolve processed directory to the folder containing dictionary files."""
    required_files = {FILE_NAMES["title_to_text"], FILE_NAMES["title_to_url"]}
    if required_files.issubset({path.name for path in processed_dir.iterdir()}):
        return processed_dir

    subdirs = [path for path in processed_dir.iterdir() if path.is_dir()]
    if not subdirs:
        raise ValueError(f"No processed run folders found under: {processed_dir}")

    latest_subdir = max(subdirs, key=lambda path: path.name)
    latest_files = {path.name for path in latest_subdir.iterdir()}
    if not required_files.issubset(latest_files):
        raise ValueError(
            "Processed folder does not contain required files "
            f"{sorted(required_files)}: {latest_subdir}"
        )
    return latest_subdir


@click.command()
@click.option(
    "--processed-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a processed run folder or its parent directory.",
)
@click.option(
    "--output-file",
    default=Path("dataset.jsonl"),
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    show_default=True,
    help="Path to write the output JSONL dataset.",
)
def main(processed_dir: Path, output_file: Path) -> None:
    """Build url-title-text dataset from processed dictionaries."""
    resolved_processed_dir = _resolve_processed_dir(processed_dir=processed_dir)
    builder = WikiArticlesBuilder(
        processed_dir=resolved_processed_dir, output_file=output_file
    )
    builder.build()


if __name__ == "__main__":
    main()
