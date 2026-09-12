"""
Inspect the structure of the character recognition dataset.
"""

import csv
import sys
from pathlib import Path


def inspect_dataset(csv_path):
    """
    Inspect the structure of the character recognition dataset.

    The function checks whether the CSV file exists and displays
    the number of columns, row size, selected column names, and
    sample values from the first data row.

    Parameters
    ----------
    csv_path : str
        Path to the original character dataset CSV file.
    """
    path = Path(csv_path)

    if not path.exists():
        print("ERROR: Dataset file not found.")
        return

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        header = next(reader)
        first_row = next(reader)

        print("=" * 60)
        print("DATASET STRUCTURE")
        print("=" * 60)

        print("\nNumber of columns:", len(header))
        print("Number of values in one row:", len(first_row))

        print("\nFIRST 10 COLUMNS:")
        for i in range(min(10, len(header))):
            print(i, "->", header[i])

        print("\nLAST 10 COLUMNS:")
        start = max(0, len(header) - 10)

        for i in range(start, len(header)):
            print(i, "->", header[i])

        print("\nLAST 10 VALUES OF FIRST ROW:")
        for i in range(max(0, len(first_row) - 10), len(first_row)):
            print(i, "->", first_row[i])

        print("\nFirst row length:", len(first_row))

        print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print(
            r'python src\inspect_dataset.py "FULL_PATH_TO_CSV"'
        )
        sys.exit(1)

    inspect_dataset(sys.argv[1])