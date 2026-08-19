"""
Download the source data for the proof of concept.

Three files are needed:

    cp_di.tsv.gz              Cell Painting consensus profiles
    l1000_di.tsv.gz           L1000 consensus signatures
    repurposing_samples.txt   Drug Repurposing Hub structures (SMILES, InChIKey)

Usage:
python download_data.py                 # downloads into ./data
python download_data.py --data-dir DIR  # downloads into DIR
python download_data.py --force         # re-download even if present

Total download is roughly 49 MB.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

COMPLEMENTARITY_REPO = (
    "https://raw.githubusercontent.com/broadinstitute/"
    "lincs-profiling-complementarity/master/1.Data-exploration/Consensus"
)
CELL_PAINTING_REPO = (
    "https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master"
)

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class SourceFile:
    """One file to fetch, with enough information to sanity-check the result."""

    local_name: str
    url: str
    description: str
    approximate_megabytes: float
    is_gzipped: bool


# Note the inconsistent naming upstream: the L1000 file uses "doseindependent"
# while the Cell Painting file uses "dose_independent". This is not a typo here.
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile(
        local_name="cp_di.tsv.gz",
        url=(
            f"{COMPLEMENTARITY_REPO}/cell_painting/moa_sizes_consensus_datasets/"
            "cell_painting_moa_analytical_set_profiles_dose_independent.tsv.gz"
        ),
        description="Cell Painting consensus profiles (morphology)",
        approximate_megabytes=21.9,
        is_gzipped=True,
    ),
    SourceFile(
        local_name="l1000_di.tsv.gz",
        url=(
            f"{COMPLEMENTARITY_REPO}/L1000/moa_sizes_consensus_datasets/"
            "l1000_moa_analytical_set_profiles_doseindependent.tsv.gz"
        ),
        description="L1000 consensus signatures (transcriptomics)",
        approximate_megabytes=26.0,
        is_gzipped=True,
    ),
    SourceFile(
        local_name="repurposing_samples.txt",
        url=f"{CELL_PAINTING_REPO}/metadata/moa/clue/repurposing_samples_20200324.txt",
        description="Drug Repurposing Hub structures (SMILES, InChIKey)",
        approximate_megabytes=2.4,
        is_gzipped=False,
    ),
)


def _format_size(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB"


def _download_to_path(url: str, destination: Path) -> None:
    """Fetch a URL to disk, writing to a temporary file first.

    Writing to a temporary path and renaming on success means an interrupted
    download never leaves a truncated file that a later run would treat as valid.
    """
    temporary_path = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary_path.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} fetching {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error
    temporary_path.rename(destination)


def _verify_file(source: SourceFile, path: Path) -> None:
    """Check the downloaded file is readable and looks like the expected table.

    A truncated or error-page download would otherwise fail much later with a
    confusing pandas error, so it is worth catching here.
    """
    if path.stat().st_size < 1000:
        raise RuntimeError(f"{path.name} is suspiciously small; download likely failed.")

    opener = gzip.open if source.is_gzipped else open
    try:
        with opener(path, "rt", errors="replace") as handle:
            first_lines = [handle.readline() for _ in range(12)]
    except OSError as error:
        raise RuntimeError(f"{path.name} could not be read: {error}") from error

    if not any("\t" in line for line in first_lines):
        raise RuntimeError(f"{path.name} does not look like a tab-separated table.")


def download_source_files(data_directory: Path, force: bool = False) -> None:
    """Download every source file that is not already present."""
    data_directory.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_directory}\n")

    for source in SOURCE_FILES:
        destination = data_directory / source.local_name

        if destination.exists() and not force:
            print(
                f"  [skip]     {source.local_name:24s} "
                f"already present ({_format_size(destination.stat().st_size)})"
            )
            continue

        print(
            f"  [download] {source.local_name:24s} "
            f"{source.description} (~{source.approximate_megabytes:.1f} MB)"
        )
        _download_to_path(source.url, destination)
        _verify_file(source, destination)
        print(f"             saved {_format_size(destination.stat().st_size)}")

    print("\nAll source files present. Next: python src/run_similarity_analysis.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help="Where to save the files (default: ./data next to this script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they are already present",
    )
    arguments = parser.parse_args()

    try:
        download_source_files(arguments.data_dir, force=arguments.force)
    except RuntimeError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
