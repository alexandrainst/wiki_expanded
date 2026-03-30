"""Build a simple Wikipedia dataset with url, title, and text columns.

Usage:
>>> python src/scripts/build_wiki_articles.py \
...   --processed-dir="data/processed/da/2026-03-27-13-16-29" \
...   --output-file="dataset.jsonl"
"""

from pathlib import Path

import click

from wiki_expanded.wiki_articles import WikiArticlesBuilder


@click.command()
@click.option(
    "--processed-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "Path to a processed run folder containing "
        "articles.sqlite3 and title_to_url.json."
    ),
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
    builder = WikiArticlesBuilder(processed_dir=processed_dir, output_file=output_file)
    builder.build()


if __name__ == "__main__":
    main()
