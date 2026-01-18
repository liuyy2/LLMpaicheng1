#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge results_per_episode.csv from two result dirs into one output dir.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List


def read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def append_rows(in_path: str, writer: csv.writer, skip_header: bool) -> int:
    count = 0
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge results_per_episode.csv from two result directories."
    )
    parser.add_argument("--llm", required=True, help="LLM results directory")
    parser.add_argument("--baseline", required=True, help="Baseline results directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    llm_csv = os.path.join(args.llm, "results_per_episode.csv")
    base_csv = os.path.join(args.baseline, "results_per_episode.csv")
    out_csv = os.path.join(args.output, "results_per_episode.csv")

    if not os.path.exists(llm_csv):
        print(f"Missing: {llm_csv}", file=sys.stderr)
        return 1
    if not os.path.exists(base_csv):
        print(f"Missing: {base_csv}", file=sys.stderr)
        return 1

    llm_header = read_header(llm_csv)
    base_header = read_header(base_csv)
    if llm_header != base_header:
        print("Header mismatch between CSV files.", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(llm_header)
        llm_rows = append_rows(llm_csv, writer, skip_header=True)
        base_rows = append_rows(base_csv, writer, skip_header=True)

    print(f"Output: {out_csv}")
    print(f"LLM rows: {llm_rows}")
    print(f"Baseline rows: {base_rows}")
    print(f"Total rows: {llm_rows + base_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
