"""Convert JSONL files to Parquet format with automatic ID generation."""

import json
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd


def convert_jsonl_to_parquet(input_path: Path, output_path: Path) -> None:
    """Convert JSONL file to Parquet format."""
    data = read_jsonl(file_path=input_path)

    data = add_id_column(data=data)

    df = pd.DataFrame(data)
    df = rename_columns(df=df)
    df = keep_only_relevant_columns(df=df)

    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))

    return data


def add_id_column(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ID column."""
    for i, record in enumerate(data):
        record["id"] = f"wiki_expanded_{i}"

    return data


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns."""
    # expanded_text -> text

    df = df.rename(columns={"expanded_text": "text"})

    return df


def keep_only_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns."""
    keep_columns = ["id", "text"]
    df = df[keep_columns]
    return df


@click.command(help="Convert JSONL files to Parquet format.")
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def main(input_file: Path) -> None:
    """CLI interface for JSONL to Parquet conversion."""
    input_path = input_file
    output_path = input_path.with_suffix(".parquet")

    convert_jsonl_to_parquet(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    exit(main())
