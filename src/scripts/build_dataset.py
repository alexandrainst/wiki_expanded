"""Build the expanded Wikipedia dataset."""

from pathlib import Path

import click

from wiki_expanded.dataset_builder import DatasetBuilder


@click.command()
@click.option(
    "--processed-dir",
    default="data/processed",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing processed data.",
)
@click.option(
    "--num-tokens-threshold",
    default=10000,
    type=int,
    help="Stop expanding links when the text is at least this many tokens long.",
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
    "--max-link-expansions",
    default=None,
    type=int,
    help="Maximum number of times a link can be expanded. If None, no limit.",
)
@click.option(
    "--include-strategy",
    default="prepend",
    type=click.Choice(["prepend", "append"], case_sensitive=False),
    help="Strategy for how to include link text.",
)
@click.option(
    "--ignore-short-samples",
    default=False,
    type=bool,
    help="Ignore samples with fewer than `num_tokens_threshold` tokens.",
)
def main(
    processed_dir: Path,
    num_tokens_threshold: int,
    save_dir: Path,
    max_dataset_length: int | None,
    max_link_expansions: int | None = None,
    include_strategy: str = "prepend",
    ignore_short_samples: bool = False,
) -> None:
    """Build the expanded Wikipedia dataset."""
    builder = DatasetBuilder(
        processed_dir=processed_dir,
        num_tokens_threshold=num_tokens_threshold,
        save_dir=save_dir,
        max_dataset_length=max_dataset_length,
        max_link_expansions=max_link_expansions,
        include_strategy=include_strategy,
        ignore_short_samples=ignore_short_samples,
    )
    builder.build_expanded_dataset()


if __name__ == "__main__":
    main()
