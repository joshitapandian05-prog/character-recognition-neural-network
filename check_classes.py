"""
Check the class labels present in the character dataset.
"""

import csv
from collections import Counter

CSV_PATH = r"C:\Users\JOSHITA\OneDrive\Desktop\digit_char_dataset.csv"


def check_classes(csv_path):
    """
    Count the number of samples belonging to each class.

    Parameters
    ----------
    csv_path : str
        Path to the original character dataset CSV file.

    Notes
    -----
    The class counts are used to verify the availability of
    sufficient samples before creating the balanced dataset.
    """
    class_counts = Counter()

    print("Reading class labels...")
    print("This may take a little time because the CSV is large.")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        # Skip header
        header = next(reader)

        # Class is the last column
        class_index = len(header) - 1

        for row in reader:
            if row:
                class_label = row[class_index]
                class_counts[class_label] += 1

    print("\n" + "=" * 50)
    print("CLASS INFORMATION")
    print("=" * 50)

    print("Number of unique classes:", len(class_counts))

    print("\nClasses and sample counts:")

    for label in sorted(class_counts, key=lambda x: int(float(x))):
        print(f"Class {label}: {class_counts[label]} samples")

    print("=" * 50)


if __name__ == "__main__":
    check_classes(CSV_PATH)