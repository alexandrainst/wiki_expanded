"""Build the expanded Wikipedia dataset.

Usage:
>>> python src/scripts/build_dataset.py \
        --processed-dir="data/processed/da" \
        --min-tokens=32768 \
        --max-tokens=65536 \
        --max-link-expansions-global=10 \
        --include-strategy="prepend"
"""

import sys
from pathlib import Path

import click

from wiki_expanded.dataset_builder import DatasetBuilder


@click.command()
@click.option(
    "--processed-dir",
    default="data/processed",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing processed data.",
)
@click.option(
    "--min-tokens",
    default=0,
    type=int,
    help=(
        "Minimum number of tokens required for a sample to be included in the dataset."
    ),
)
@click.option(
    "--max-tokens",
    default=None,
    type=int,
    help=(
        "Stop expanding links when the text reaches this many tokens. "
        "If not specified, no upper limit is applied."
    ),
)
@click.option(
    "--save-dir",
    default="data/final",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Path to save the expanded dataset (JSONL).",
)
@click.option(
    "--max-dataset-length",
    default=None,
    type=int,
    help="Maximum number of samples in the expanded dataset. If None, use all.",
)
@click.option(
    "--max-link-expansions-local",
    default=None,
    type=int,
    help=(
        "Maximum number of links that can be expanded for a single sample. "
        "If None, no limit."
    ),
)
@click.option(
    "--max-link-expansions-global",
    default=None,
    type=int,
    help=(
        "Maximum number of times a link can be expanded across all samples. "
        "If None, no limit."
    ),
)
@click.option(
    "--include-strategy",
    default="prepend",
    type=click.Choice(["prepend", "append"], case_sensitive=False),
    help="Strategy for how to include link text.",
)
def main(
    processed_dir: Path,
    min_tokens: int,
    max_tokens: int | None,
    save_dir: Path,
    max_dataset_length: int | None,
    max_link_expansions_local: int | None = None,
    max_link_expansions_global: int | None = None,
    include_strategy: str = "prepend",
) -> None:
    """Build the expanded Wikipedia dataset."""
    if not processed_dir.exists():
        raise FileNotFoundError(
            (
                f"Processed directory '{processed_dir}' not found. "
                "Run: `python src/scripts/process.py`"
            )
        )

    # Convert None to sys.maxsize for no upper limit
    if max_tokens is None:
        max_tokens = sys.maxsize

    builder = DatasetBuilder(
        processed_dir=processed_dir,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        save_dir=save_dir,
        max_dataset_length=max_dataset_length,
        max_link_expansions_local=max_link_expansions_local,
        max_link_expansions_global=max_link_expansions_global,
        include_strategy=include_strategy,
    )
    builder.build_expanded_dataset()


if __name__ == "__main__":
    main()
