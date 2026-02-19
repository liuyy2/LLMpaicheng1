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


def append_rows(in_path: str, writer: csv.writer, skip_header: bool,
                out_header: List[str] | None = None) -> int:
    """Write rows from in_path to writer.

    If *out_header* is given, columns are aligned to it: missing columns in the
    source file are filled with an empty string so the output is always
    consistent with the target header.
    """
    count = 0
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        src_header = next(reader)               # always read the header
        if out_header is not None and src_header != out_header:
            # build index map: out_col -> position in src row (or None)
            idx_map = [
                src_header.index(col) if col in src_header else None
                for col in out_header
            ]
            for row in reader:
                if not row:
                    continue
                aligned = [
                    row[i] if i is not None else ""
                    for i in idx_map
                ]
                writer.writerow(aligned)
                count += 1
        else:
            if not skip_header:
                writer.writerow(src_header)
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

    # Build merged header: union, preserving order (LLM header first, then any
    # extra baseline-only columns appended at the end).
    merged_header = list(llm_header)
    for col in base_header:
        if col not in merged_header:
            merged_header.append(col)

    if llm_header != base_header:
        extra_llm = [c for c in llm_header if c not in base_header]
        extra_base = [c for c in base_header if c not in llm_header]
        print(
            f"Note: header mismatch resolved automatically.\n"
            f"  Columns only in LLM file   : {extra_llm}\n"
            f"  Columns only in baseline   : {extra_base}\n"
            f"  Missing columns will be filled with empty string.",
            file=sys.stderr,
        )

    os.makedirs(args.output, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(merged_header)
        llm_rows = append_rows(llm_csv, writer, skip_header=True,
                               out_header=merged_header)
        base_rows = append_rows(base_csv, writer, skip_header=True,
                                out_header=merged_header)

    print(f"Output: {out_csv}")
    print(f"LLM rows: {llm_rows}")
    print(f"Baseline rows: {base_rows}")
    print(f"Total rows: {llm_rows + base_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
