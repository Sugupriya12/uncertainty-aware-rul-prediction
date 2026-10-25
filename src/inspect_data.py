
from pathlib import Path
import pandas as pd

# Project root and dataset location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Challenge_Data"

FILES = ["train.txt", "test.txt", "final_test.txt"]

# PHM08 dataset: 26 columns
COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def inspect_file(filename):
    path = DATA_DIR / filename

    if not path.exists():
        print(f"\nMissing file: {path}")
        return

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    print(f"\n{'=' * 45}")
    print(f"File: {filename}")
    print(f"Shape: {df.shape}")
    print(f"Unique engines: {df['engine_id'].nunique()}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nCycles per engine:")
    print(df.groupby("engine_id")["cycle"].agg(["min", "max", "count"]).head())


if __name__ == "__main__":
    for filename in FILES:
        inspect_file(filename)
