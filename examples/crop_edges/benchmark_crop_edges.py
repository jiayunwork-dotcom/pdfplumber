"""Measure the cost of deriving edges across many cropped views.

Run from the repository root:

    python examples/crop_edges/benchmark_crop_edges.py

The script builds pages with growing numbers of stroked paths and, for each,
opens several cropped views, reads their edges, and runs ``TableFinder``.  It
reports elapsed time and peak traced memory, together with the average number
of page objects each view actually holds after cropping.

Two regimes show where the cost comes from:

* ``full-width``: every path spans the whole page, so every crop intersects
  every path.  Each view therefore holds (nearly) the whole page, and total
  work is proportional to ``paths x views`` — the irreducible worst case.
* ``localized``: paths occupy small, scattered strips.  A crop intersects only
  a handful of them, and total work tracks the number of *intersecting* paths,
  not the page's total.

In neither regime does the implementation copy the full page's path list into
each view: a view stores only the objects the existing crop already returns,
and derives its edges from that subset.

The numbers depend on the machine; this is evidence, not a test.
"""

import io
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pdf_fixtures import PageContent, build_pdf  # noqa: E402

import pdfplumber  # noqa: E402


def full_width_page(n_paths):
    content = PageContent()
    for i in range(n_paths):
        y = 5 + (i * 1.7) % 190
        content.line(5, y, 195, y + 0.3)
    return content


def localized_page(n_paths):
    content = PageContent()
    for i in range(n_paths):
        y = 5 + (i * 1.8) % 190
        x = 5 + (i * 13) % 150
        content.line(x, y, x + 20, y)
    return content


def measure(page_factory, n_paths, n_views, box):
    pdf = pdfplumber.open(io.BytesIO(build_pdf([page_factory(n_paths)])))
    page = pdf.pages[0]
    _ = page.edges

    views = [page.crop(box(k)) for k in range(n_views)]
    held = [sum(len(v) for v in view.objects.values()) for view in views]

    tracemalloc.start()
    t0 = time.perf_counter()
    for view in views:
        _ = view.edges
        view.debug_tablefinder()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "paths": n_paths,
        "views": n_views,
        "page_objects": sum(len(v) for v in page.objects.values()),
        "avg_view_objects": sum(held) / len(held),
        "max_view_objects": max(held),
        "elapsed_ms": elapsed * 1000.0,
        "peak_kb": peak / 1024.0,
    }


def main():
    regimes = {
        "full-width": (
            full_width_page,
            lambda k: (10 + k, 10, 190 - k, 190),
        ),
        "localized": (
            localized_page,
            lambda k: (
                10 + (k * 7) % 160,
                10 + (k * 11) % 160,
                30 + (k * 7) % 160,
                30 + (k * 11) % 160,
            ),
        ),
    }

    header = (
        "regime      paths  views  page_obj  avg_view_obj  max_view_obj  "
        "elapsed_ms  peak_kb"
    )
    print(header)
    print("-" * len(header))
    for name, (factory, box) in regimes.items():
        for n in (200, 400, 800):
            r = measure(factory, n, 20, box)
            print(
                f"{name:11s} {r['paths']:5d}  {r['views']:5d}  "
                f"{r['page_objects']:8d}  {r['avg_view_objects']:12.1f}  "
                f"{r['max_view_objects']:12d}  {r['elapsed_ms']:10.1f}  "
                f"{r['peak_kb']:7.1f}"
            )


if __name__ == "__main__":
    main()
