"""Example: ETL-style collection of rotated tables.

Run with a PDF that mixes text orientations, e.g. one produced by the
synthetic fixture generator in ``tests/rotation_fixtures/synth.py``:

    python examples/collect_rotated_tables.py mixed.pdf

Every output record keeps the page number, the detected rotation, the page
bounding box and the logical row data, so downstream consumers can check a
row against the original page.
"""

import json
import sys
from typing import Any, Dict, List

import pdfplumber

TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "rotation_mode": "auto",
}


def collect(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.find_tables(TABLE_SETTINGS)):
                for row_index, row in enumerate(table.extract()):
                    records.append(
                        {
                            "page": page_number,
                            "table": table_index,
                            "rotation": table.rotation,
                            "bbox": list(table.bbox),
                            "row": row_index,
                            "cells": row,
                        }
                    )
    return records


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: collect_rotated_tables.py <path-to-pdf>", file=sys.stderr)
        return 2
    records = collect(argv[1])
    json.dump(records, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
