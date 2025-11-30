"""Download and inspect IoT anomaly datasets for the FedIoT workspace."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 1024 * 1024


def derive_output_name(url: str) -> str:
    """Infer a filename from the download URL."""

    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    return candidate or "dataset.bin"


def build_output_path(path_value: Optional[str], inferred_name: str) -> Path:
    """Resolve the download target path relative to the project."""

    if path_value is not None:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = DEFAULT_RAW_DIR / inferred_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> Path:
    """Stream the remote file to disk with basic progress logging."""

    print(f"Downloading {url}\n -> {destination}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_bytes = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                total_bytes += len(chunk)
    size_mb = total_bytes / (1024 * 1024)
    print(f"Completed download ({size_mb:.2f} MB)")
    return destination


def _strip_all_suffixes(path: Path) -> str:
    """Remove all suffixes from a path name."""

    name = path.name
    for suffix in path.suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _ensure_within_directory(directory: Path, target: Path) -> None:
    """Validate that target stays inside directory."""

    directory_resolved = directory.resolve()
    target_resolved = target.resolve()
    if not target_resolved.is_relative_to(directory_resolved):
        raise ValueError(f"Unsafe path detected in archive: {target}")


def decompress_archive(archive_path: Path) -> List[Path]:
    """Extract supported archives and return extracted file paths."""

    base_name = _strip_all_suffixes(archive_path)
    target_dir = archive_path.parent / base_name
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Path] = []
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zipped:
            for member in zipped.infolist():
                destination = target_dir / member.filename
                # Guard against path traversal when extracting user-provided archives.
                _ensure_within_directory(target_dir, destination)
                zipped.extract(member, target_dir)
                if not member.is_dir():
                    extracted_files.append(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tarred:
            for member in tarred.getmembers():
                destination = target_dir / member.name
                # Guard against path traversal when extracting user-provided archives.
                _ensure_within_directory(target_dir, destination)
            tarred.extractall(target_dir)
            for member in tarred.getmembers():
                if member.isfile():
                    extracted_files.append(target_dir / member.name)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    print(f"Extracted {len(extracted_files)} files to {target_dir}")
    return extracted_files


def summarize_csv(
    file_path: Path,
    label_column: Optional[str],
    separator: str,
    encoding: str,
    head_rows: int,
) -> None:
    """Print dataset columns, sample rows, and basic label statistics."""

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found at {file_path}")

    print(f"--- Dataset summary: {file_path} ---")
    header_frame = pd.read_csv(
        file_path,
        sep=separator,
        encoding=encoding,
        nrows=0,
    )
    columns = list(header_frame.columns)
    print(f"Columns ({len(columns)}): {columns}")

    sample_frame = pd.read_csv(
        file_path,
        sep=separator,
        encoding=encoding,
        nrows=head_rows,
    )
    print(f"Sample rows (top {len(sample_frame)}):")
    print(sample_frame.to_string(index=False))

    if label_column:
        counter: Counter = Counter()
        total_rows = 0
        for chunk in pd.read_csv(
            file_path,
            sep=separator,
            encoding=encoding,
            usecols=[label_column],
            chunksize=50000,
        ):
            counter.update(chunk[label_column].dropna())
            total_rows += len(chunk)
        print(f"Rows counted: {total_rows}")
        print(f"Label distribution for '{label_column}':")
        for value, count in counter.most_common():
            print(f"  {value!r}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IoT datasets and generate quick summaries.",
    )
    parser.add_argument(
        "--url",
        help="HTTP(S) URL to a CSV or archive to download.",
    )
    parser.add_argument(
        "--output",
        help="Destination path for the downloaded file (defaults to data/raw/).",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Extract the downloaded archive if it is zip/tar.*.",
    )
    parser.add_argument(
        "--summarize",
        help="Path to a CSV file to inspect (relative paths resolve from project root).",
    )
    parser.add_argument(
        "--label-column",
        help="Optional label column to report value counts for.",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV delimiter (default: ',').",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8).",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=5,
        help="Number of sample rows to display during summary.",
    )
    args = parser.parse_args()

    if not args.url and not args.summarize:
        parser.error("provide --url to download or --summarize to inspect an existing file")

    downloaded_path: Optional[Path] = None
    if args.url:
        inferred_name = derive_output_name(args.url)
        target_path = build_output_path(args.output, inferred_name)
        downloaded_path = download_file(args.url, target_path)
        if args.decompress:
            try:
                extracted_files = decompress_archive(downloaded_path)
            except ValueError as exc:
                print(f"Decompression skipped: {exc}")
            else:
                preview = [
                    str(path.relative_to(PROJECT_ROOT))
                    for path in extracted_files[:10]
                ]
                if preview:
                    print("Sample extracted files:")
                    for item in preview:
                        print(f"  {item}")
                else:
                    print("Archive extracted but no files were detected.")

    if args.summarize:
        summary_path = Path(args.summarize).expanduser()
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
        summarize_csv(
            file_path=summary_path,
            label_column=args.label_column,
            separator=args.separator,
            encoding=args.encoding,
            head_rows=args.head_rows,
        )
    elif downloaded_path and downloaded_path.suffix.lower() == ".csv":
        summarize_csv(
            file_path=downloaded_path,
            label_column=args.label_column,
            separator=args.separator,
            encoding=args.encoding,
            head_rows=args.head_rows,
        )


if __name__ == "__main__":
    main()
