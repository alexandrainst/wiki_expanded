"""Build 3 dictionaries that will be used to to construct the Wiki Expanded dataset.

1. `data/processed/title_to_text.json`: Maps each title to the text of the article.
2. `data/processed/title_to_links.json`: Maps each title to the links in the article.
3. `data/processed/link_to_freq.json`: Maps each link to the number of articles it
    appears in.
"""

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
    "--max-words",
    default=10000,
    type=int,
    help="Maximum number of words in expanded text.",
)
@click.option(
    "--dataset-dir-path",
    default="data/final/dataset.jsonl",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to save the expanded dataset (JSONL).",
)
@click.option(
    "--max-dataset-length",
    default=None,
    type=int,
    help="Maximum number of samples in the expanded dataset. If None, use all.",
)
def main(
    processed_dir: Path,
    max_words: int,
    dataset_dir_path: Path,
    max_dataset_length: int | None,
) -> None:
    """Build the Expanded Wikipedia dataset."""
    builder = DatasetBuilder(
        processed_dir=processed_dir,
        max_words=max_words,
        dataset_dir_path=dataset_dir_path,
        max_dataset_length=max_dataset_length,
    )
    builder.build_expanded_dataset()


if __name__ == "__main__":
    main()
