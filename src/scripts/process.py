"""Build JSON files that will be used to construct the expanded Wikipedia dataset.

Builds the following JSON files:
    - title_to_text.json: Article title -> Article text
    - title_to_links.json: Article title -> List of outgoing links
    - link_to_freq.json: Link -> Number of articles it appears in
    - title_to_num_tokens.json: Article title -> Number of tokens in the article

Usage:
>>> python src/scripts/process.py --jsonl-file="dawiki_pages.jsonl"
"""

from pathlib import Path

import click

from wiki_expanded.processor import Processor


@click.command()
@click.option(
    "--jsonl-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the JSONL file with the Wikipedia articles to process.",
)
@click.option(
    "--save-dir",
    default="data/processed",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save the processed dictionaries.",
)
@click.option(
    "--tokenizer-name",
    default="google/gemma-7b",
    type=str,
    help="Name of the tokenizer to use.",
)
@click.option(
    "--max-files",
    default=None,
    type=int,
    help="Maximum number of files to process. If None, process all files.",
)
def main(
    jsonl_file: Path,
    save_dir: Path,
    tokenizer_name: str = "google/gemma-7b",
    max_files: int | None = None,
) -> None:
    """Process the Wikipedia JSONL file."""
    processor = Processor(
        save_dir=save_dir, max_files=max_files, tokenizer_name=tokenizer_name
    )
    processor.process(jsonl_file=jsonl_file)


if __name__ == "__main__":
    main()
