"""
report.py — build the batch inspection table and export it as CSV.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd

COLUMNS = ["filename", "status", "defect_count", "primary_defect", "max_confidence", "all_defects"]


def make_row(filename: str, result: dict) -> dict:
    """Convert one inspect() result into a flat table row."""
    return {
        "filename": filename,
        "status": result["status"],
        "defect_count": result["defect_count"],
        "primary_defect": result["primary_class"],
        "max_confidence": result["max_confidence"],
        "all_defects": "; ".join(
            f"{d['class']}({d['confidence']:.2f})" for d in result["detections"]
        ) or "-",
    }


def build_report(rows: list) -> pd.DataFrame:
    """List of rows -> DataFrame with a fixed column order."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def summary_line(df: pd.DataFrame) -> str:
    total = len(df)
    if total == 0:
        return "No images processed."
    fails = int((df["status"] == "FAIL").sum())
    return f"Inspected {total} image(s): {total - fails} PASS, {fails} FAIL."


def save_csv(df: pd.DataFrame, directory: str | None = None) -> str:
    """Write the report to a timestamped CSV and return its path."""
    directory = directory or tempfile.gettempdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"inspection_report_{stamp}.csv")
    df.to_csv(path, index=False)
    return path