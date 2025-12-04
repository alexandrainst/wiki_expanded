"""Build 5 dictionaries that will be used to construct the expanded Wikipedia dataset.

1. `data/processed/title_to_text.json`: Maps each article title to its text.
2. `data/processed/title_to_links.json`: Maps each article title to
    the links found in the article.
3. `data/processed/link_to_freq.json`: Maps each link to the number
    of articles it appears in.
4. `data/processed/title_lower_to_original.json`: Maps each lowercase
    title to the original title.
5. `data/processed/title_to_tokens.json`: Maps each article title to
    its tokenized content.
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
@click.option(
    "--tokenizer-name",
    default="google/gemma-7b",
    type=str,
    help="Name of the tokenizer to use.",
)
@click.option(
    "--capitalize-titles-and-links/--no-capitalize-titles-and-links",
    default=True,
    help="Whether to capitalize titles and links (default: True).",
)
@click.option(
    "--jsonl-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to the JSONL file with the Wikipedia articles to process.",
)
@click.option(
    "--language",
    default="da",
    type=click.Choice(["da", "en"]),
    help="Language code for the Wikipedia articles (da=Danish, en=English).",
)
def main(
    text_dir: Path,
    save_dir: Path,
    jsonl_file: Path,
    max_files: int | None,
    tokenizer_name: str = "google/gemma-7b",
    capitalize_titles_and_links: bool = True,
    language: str = "da",
) -> None:
    """Process the raw articles."""
    processor = Processor(
        text_dir=text_dir,
        save_dir=save_dir,
        max_files=max_files,
        tokenizer_name=tokenizer_name,
        capitalize_titles_and_links=capitalize_titles_and_links,
        language=language,
    )
    processor.process(jsonl_file=jsonl_file)


if __name__ == "__main__":
    main()
