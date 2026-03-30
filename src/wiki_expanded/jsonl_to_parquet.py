"""Convert JSONL files to Parquet format with automatic ID generation.

This module is designed to work with very large JSONL files by streaming input
and writing Parquet row groups incrementally, avoiding loading the full dataset
into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import click
import pyarrow as pa
import pyarrow.parquet as pq


def convert_jsonl_to_parquet(
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int = 50_000,
    compression: str = "snappy",
    id_prefix: str = "wiki_expanded_",
) -> None:
    """Convert a JSONL file to Parquet format without loading it into memory.

    Args:
        input_path: Input JSONL file path.
        output_path: Output Parquet file path.
        batch_size: Number of rows per Parquet write (row group).
        compression: Parquet compression codec (e.g. "snappy", "zstd", "gzip").
        id_prefix: Prefix for generated IDs (ID = f"{id_prefix}{row_index}").
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    texts: list[str | None] = []
    writer: pq.ParquetWriter | None = None

    def flush() -> None:
        nonlocal ids, texts, writer
        if not ids:
            return

        table = pa.table({"id": ids, "text": texts})
        if writer is None:
            writer = pq.ParquetWriter(
                where=str(output_path), schema=table.schema, compression=compression
            )
        writer.write_table(table)
        ids = []
        texts = []

    try:
        for row_idx, record in enumerate(iter_jsonl(input_path)):
            ids.append(f"{id_prefix}{row_idx}")
            texts.append(extract_text(record))

            if len(ids) >= batch_size:
                flush()

        flush()
    finally:
        if writer is not None:
            writer.close()


def iter_jsonl(file_path: Path) -> Iterable[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, line by line.

    Empty lines are skipped.

    Args:
        file_path: Path to a JSONL file.

    Yields:
        Parsed JSON objects (dicts).
    """
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def extract_text(record: dict[str, Any]) -> str | None:
    """Extract the output text column from an input JSON record.

    The historical column name is "expanded_text". If input already uses "text",
    that is also accepted.

    Args:
        record: Parsed JSON record.

    Returns:
        The extracted text, or None if not present.
    """
    text = record.get("expanded_text")
    if text is None:
        text = record.get("text")

    if text is None:
        return None

    return str(text)


@click.command(help="Convert JSONL files to Parquet format (streaming).")
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output Parquet file path (defaults to INPUT_FILE.parquet).",
)
@click.option(
    "--batch-size",
    type=int,
    default=50_000,
    show_default=True,
    help="Rows per Parquet write (controls memory usage).",
)
@click.option(
    "--compression",
    type=str,
    default="snappy",
    show_default=True,
    help='Parquet compression codec (e.g. "snappy", "zstd", "gzip").',
)
@click.option(
    "--id-prefix",
    type=str,
    default="wiki_expanded_",
    show_default=True,
    help="Prefix for generated IDs.",
)
def main(
    input_file: Path,
    output_file: Path | None,
    batch_size: int,
    compression: str,
    id_prefix: str,
) -> None:
    """CLI interface for JSONL to Parquet conversion."""
    input_path = input_file
    output_path = output_file or input_path.with_suffix(".parquet")

    convert_jsonl_to_parquet(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        compression=compression,
        id_prefix=id_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
