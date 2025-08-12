"""Build 3 dictionaries that will be used to to construct the Wiki Expanded dataset.

1. `data/processed/title_to_text.json`: Maps each title to the text of the article.
2. `data/processed/title_to_links.json`: Maps each title to the links in the article.
3. `data/processed/link_to_freq.json`: Maps each link to the number of articles
    it appears in.
4. `data/processed/titles_lowered_to_original.json`: Maps each lowercase title to
    the original title.
"""

from pathlib import Path

import click

from wiki_expanded.processor import Processor


@click.command()
@click.option(
    "--text-dir",
    default="data/raw/text",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing raw article text files.",
)
@click.option(
    "--save-dir",
    default="data/processed",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to save the processed dictionaries.",
)
@click.option(
    "--max-files",
    default=None,
    type=int,
    help="Maximum number of files to process. If None, process all files.",
)
def main(text_dir: Path, save_dir: Path, max_files: int | None) -> None:
    """Process the raw articles."""
    processor = Processor(text_dir=text_dir, save_dir=save_dir, max_files=max_files)
    processor.process()


if __name__ == "__main__":
    main()
