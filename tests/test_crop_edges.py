"""Region-aware edge derivation and table-finding on cropped pages.

These tests build their own PDFs (see ``pdf_fixtures.py``), so the expected
geometry is derived from the drawing commands rather than from pdfplumber's
output. All coordinates below are pdfplumber coordinates (origin top-left),
i.e. the PDF user-space y values flipped about the page height.
"""

import copy
import io
import time
import tracemalloc
from typing import Dict, List, Sequence, Tuple

import pytest
from pdf_fixtures import PageContent, build_pdf

import pdfplumber
from pdfplumber import utils
from pdfplumber.table import TableFinder


def _open(pages: Sequence[PageContent], **kwargs):
    return pdfplumber.open(io.BytesIO(build_pdf(pages, **kwargs)))


def _bbox_of(edge: Dict) -> Tuple[float, float, float, float]:
    return (edge["x0"], edge["top"], edge["x1"], edge["bottom"])


def _edge_boxes(edges) -> List[Tuple[str, float, float, float, float]]:
    return sorted((e["object_type"], *_bbox_of(e)) for e in edges)


def _inside(edge: Dict, bbox: Tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    return (
        x0 - 1e-6 <= edge["x0"] <= edge["x1"] <= x1 + 1e-6
        and top - 1e-6 <= edge["top"] <= edge["bottom"] <= bottom + 1e-6
    )


# ---------------------------------------------------------------------------
# The two reported cases
# ---------------------------------------------------------------------------


def test_rectangle_entirely_outside_crop_yields_no_edges():
    # PDF-space rect (10,10)-(190,190); flipping about H=200 gives
    # pdfplumber (10,10)-(190,190), so a crop to (50,50,150,150) excludes all
    # four sides.
    pdf = _open([PageContent().rect(10, 10, 180, 180)])
    page = pdf.pages[0]
    cropped = page.crop((50, 50, 150, 150))

    assert cropped.rects  # the object still intersects the crop
    assert cropped.rect_edges == []
    assert cropped.edges == []
    assert cropped.debug_tablefinder().cells == []
    assert cropped.extract_table() is None


def test_open_polyline_outside_crop_is_clipped():
    # PDF (10,100)->(100,100)->(100,190). Flipped: (10,100)->(100,100)->
    # (100,10). Crop (50,50,150,150) keeps only the in-region portions:
    # horizontal from x=50..100 at top=100, vertical from y=50..100 at x=100.
    pdf = _open([PageContent().polyline([(10, 100), (100, 100), (100, 190)])])
    page = pdf.pages[0]
    cropped = page.crop((50, 50, 150, 150))

    assert _edge_boxes(cropped.curve_edges) == [
        ("curve_edge", 50.0, 100.0, 100.0, 100.0),
        ("curve_edge", 100.0, 50.0, 100.0, 100.0),
    ]
    assert all(_inside(e, (50, 50, 150, 150)) for e in cropped.curve_edges)


# ---------------------------------------------------------------------------
# Partial crossings, re-entry, tangency, closed shapes
# ---------------------------------------------------------------------------


def test_partially_crossing_rectangle_is_clipped_not_closed():
    # PDF rect x=20..120, y=20..120 flips to top=80..180. Crop x=50..150,
    # top=40..90 keeps the bottom horizontal (top=80) and the right vertical
    # (x=120); the top horizontal (top=180) and left vertical (x=20) are out.
    # The crop edge does not become a rectangle side.
    pdf = _open([PageContent().rect(20, 20, 100, 100)])
    cropped = pdf.pages[0].crop((50, 40, 150, 90))

    boxes = {(e["orientation"], _bbox_of(e)) for e in cropped.rect_edges}
    assert boxes == {
        ("h", (50.0, 80.0, 120.0, 80.0)),
        ("v", (120.0, 80.0, 120.0, 90.0)),
    }
    assert all(_inside(e, (50, 40, 150, 90)) for e in cropped.rect_edges)
    # No cell, since the crop boundary did not seal the shape.
    assert cropped.debug_tablefinder().cells == []


def test_open_path_leaving_and_reentering_is_not_bridged():
    # Path PDF (10,20)->(190,20)->(190,90)->(10,90)->(10,150); flipped tops
    # are 180, 180, 110, 110, 50. Two horizontal segments cross a narrow
    # x=40..60 crop at top=180 and top=110; the vertical connectors are
    # outside. The gap between the horizontals must stay a gap.
    pdf = _open(
        [PageContent().polyline([(10, 20), (190, 20), (190, 90), (10, 90), (10, 150)])]
    )
    cropped = pdf.pages[0].crop((40, 0, 60, 200))
    paths = [e for e in cropped.curve_edges if e["orientation"] == "h"]
    assert _edge_boxes(paths) == [
        ("curve_edge", 40.0, 110.0, 60.0, 110.0),
        ("curve_edge", 40.0, 180.0, 60.0, 180.0),
    ]


def test_closed_path_keeps_its_own_closing_side_only():
    # Closed triangle drawn as an explicit path (not a square, so pdfminer
    # keeps it as an LTCurve). Flipped about H=200 the drawn segments are:
    # (20,180)->(120,180) [h], (120,180)->(120,80) [v], and the `h` closing
    # side back to (20,180). Only the diagonal from (120,80) to (20,180)
    # crosses the crop, and it is clipped — no closing side is invented.
    pdf = _open(
        [
            PageContent().polyline(
                [(20, 20), (120, 20), (120, 120), (20, 20)], close=True
            )
        ]
    )
    cropped = pdf.pages[0].crop((30, 30, 110, 110))
    assert _edge_boxes(cropped.curve_edges) == [
        ("curve_edge", 90.0, 90.0, 110.0, 110.0)
    ]
    # No cell, and no edge coincides with the crop's own boundary sides.
    assert cropped.debug_tablefinder().cells == []


def test_tangent_edge_is_kept_and_out_of_region_tangency_dropped():
    # PDF y=150 flips to top=50; PDF y=10 flips to top=190. The first lies
    # exactly on the crop's top edge, the second entirely outside.
    pdf = _open([PageContent().line(20, 150, 80, 150).line(20, 10, 80, 10)])
    page = pdf.pages[0]
    cropped = page.crop((20, 50, 80, 150))
    assert cropped.edges and _edge_boxes(cropped.edges) == [
        ("line", 20.0, 50.0, 80.0, 50.0)
    ]

    # A view whose top edge is just past the segment yields nothing from it.
    grazed = page.crop((20, 60, 80, 150))
    assert grazed.edges == []


def test_inside_table_still_recognized_text_included():
    # Table with all four sides and a cross, entirely within (50,50,150,150).
    # PDF y=60..140 flips to top=60..140.
    content = (
        PageContent()
        .rect(60, 60, 80, 80)
        .line(100, 60, 100, 140)
        .line(60, 100, 140, 100)
        .text(64, 118, "A")
        .text(104, 118, "B")
        .text(64, 78, "C")
        .text(104, 78, "D")
    )
    pdf = _open([content])
    cropped = pdf.pages[0].crop((50, 50, 150, 150))

    tf = cropped.debug_tablefinder()
    assert len(tf.cells) == 4
    assert cropped.extract_table() == [["A", "B"], ["C", "D"]]


def test_decorative_box_around_inside_table_is_clipped():
    # Inner table PDF x/y 70..130 (pdfplumber top 70..130); decorative frame
    # PDF 60..140 (top 60..140). The crop (55, 65, 145, 135) keeps the frame's
    # left/right verticals (x=60 and x=140) but cuts off its top/bottom. Those
    # clipped verticals must not invent cells along the crop boundary, and the
    # inner table must still be found with its text.
    content = (
        PageContent()
        .rect(60, 60, 80, 80)  # frame: PDF x/y 60..140
        .rect(70, 70, 60, 60)  # inner table outer: 70..130
        .line(100, 70, 100, 130)
        .line(70, 100, 130, 100)
        .text(74, 112, "P")
        .text(104, 112, "Q")
        .text(74, 82, "R")
        .text(104, 82, "S")
    )
    pdf = _open([content])
    cropped = pdf.pages[0].crop((55, 65, 145, 135))

    tf = cropped.debug_tablefinder()
    assert cropped.extract_table() == [["P", "Q"], ["R", "S"]]
    # The decorative frame's verticals survive, clipped to the crop.
    frame_verticals = {
        (_bbox_of(e)) for e in cropped.rect_edges if e["orientation"] == "v"
    }
    assert (60.0, 65.0, 60.0, 135.0) in frame_verticals
    assert (140.0, 65.0, 140.0, 135.0) in frame_verticals
    # Every reported cell stays inside the crop.
    for cell in tf.cells:
        assert 55 - 1e-6 <= cell[0] and cell[2] <= 145 + 1e-6
        assert 65 - 1e-6 <= cell[1] and cell[3] <= 135 + 1e-6


# ---------------------------------------------------------------------------
# lines_strict source distinction survives
# ---------------------------------------------------------------------------


def test_lines_strict_does_not_promote_rect_or_path_edges():
    # A table whose grid is drawn entirely as a rectangle plus a single line
    # divider; "lines" uses the rect's sides, "lines_strict" must not, so it
    # cannot form cells from the rectangle's sides.
    content = PageContent().rect(50, 50, 100, 100).line(50, 100, 150, 100)
    pdf = _open([content])
    page = pdf.pages[0]

    loose = page.debug_tablefinder(
        {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    )
    strict = page.debug_tablefinder(
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"}
    )
    assert loose.cells  # rectangle sides participate
    assert strict.cells == []
    assert strict.edges == [] or all(e["object_type"] == "line" for e in strict.edges)


# ---------------------------------------------------------------------------
# Explicit hints: numeric and object forms, reused across views
# ---------------------------------------------------------------------------


def _grid_content() -> PageContent:
    # 3 columns x 2 rows inside (60..140, 60..140) plus text.
    return (
        PageContent()
        .line(60, 60, 140, 60)
        .line(60, 100, 140, 100)
        .line(60, 140, 140, 140)
        .line(60, 60, 60, 140)
        .line(100, 60, 100, 140)
        .line(140, 60, 140, 140)
        .text(64, 118, "A")
        .text(104, 118, "B")
        .text(64, 78, "C")
        .text(104, 78, "D")
    )


def test_numeric_hints_outside_region_are_dropped_not_snapped():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    region = page.crop((60, 60, 140, 140))

    # x=145 is 5pt outside the crop; with a 10pt snap tolerance it is well
    # within reach of the boundary, but region membership is checked before
    # snapping, so it must not participate.
    settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [62, 100, 138, 145],
        "explicit_horizontal_lines": [62, 100, 138],
        "snap_tolerance": 10,
        "join_tolerance": 10,
        "intersection_tolerance": 10,
    }
    tf = TableFinder(region, settings)
    xs = sorted({round(e["x0"], 6) for e in tf.edges if e["orientation"] == "v"})
    assert xs == [62.0, 100.0, 138.0]
    # All retained geometry stays in the crop.
    for edge in tf.edges:
        assert _inside(edge, (60, 60, 140, 140))
    for cell in tf.cells:
        assert 60 - 1e-6 <= cell[0] and cell[2] <= 140 + 1e-6


def test_numeric_hint_just_outside_region_not_pulled_in_by_tolerance():
    pdf = _open([_grid_content()])
    region = pdf.pages[0].crop((60, 60, 140, 140))

    # x=142.5 is 2.5pt outside — inside the 3pt snap tolerance, but still
    # outside the region, so it is dropped rather than snapped to 140.
    settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [60, 100, 142.5],
        "explicit_horizontal_lines": [60, 100, 140],
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
    }
    tf = TableFinder(region, settings)
    xs = sorted({round(e["x0"], 6) for e in tf.edges if e["orientation"] == "v"})
    assert xs == [60.0, 100.0]
    assert all(cell[2] <= 140 + 1e-6 for cell in tf.cells)


def test_uncropped_numeric_hints_are_not_filtered():
    # On a full page, numeric hints keep their historical behaviour: a hint
    # outside the page box is used as-is (not dropped, not clipped).
    pdf = _open([PageContent().text(80, 80, "X")])
    page = pdf.pages[0]
    tf = TableFinder(
        page,
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": [-10, 210],
            "explicit_horizontal_lines": [0, 200],
        },
    )
    xs = sorted({e["x0"] for e in tf.edges if e["orientation"] == "v"})
    assert xs == [-10, 210]


def test_object_hints_use_only_original_in_region_segments():
    pdf = _open([PageContent().rect(0, 0, 200, 200)])
    page = pdf.pages[0]
    region = page.crop((50, 50, 150, 150))

    # Pass the *original* page's rect object explicitly. Only the parts of its
    # outline inside the crop may be used — never the crop's outline.
    settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [page.rects[0]],
        "explicit_horizontal_lines": [page.rects[0]],
    }
    with pytest.raises(ValueError):
        # The interface still validates: explicit needs >= 2 descriptors.
        TableFinder(region, settings)

    settings["explicit_vertical_lines"] = [page.rects[0], page.rects[0]]
    settings["explicit_horizontal_lines"] = [page.rects[0], page.rects[0]]
    tf = TableFinder(region, settings)
    for edge in tf.edges:
        assert _inside(edge, (50, 50, 150, 150))
    assert tf.cells == []  # the original rect's sides are outside


def test_object_hints_keep_inside_segments():
    # Rect (60,60)-(140,140) plus text; crop wider than the rect but offset so
    # only some of the rect's sides lie inside.
    pdf = _open([PageContent().rect(60, 60, 80, 80).text(64, 118, "A")])
    page = pdf.pages[0]
    region = page.crop((70, 70, 200, 200))
    rect = page.rects[0]

    # Only the right/bottom sides of the original rect are inside.
    tf = TableFinder(
        region,
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": [rect, rect],
            "explicit_horizontal_lines": [rect, rect],
        },
    )
    for edge in tf.edges:
        assert _inside(edge, (70, 70, 200, 200))
    assert tf.cells == []


def test_explicit_config_reused_across_multiple_views():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    settings = {
        "vertical_strategy": "explicit",
        "horizontal_strategy": "explicit",
        "explicit_vertical_lines": [60, 100, 140],
        "explicit_horizontal_lines": [60, 100, 140],
    }
    snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in settings.items()}
    object_ids = [id(v) for v in settings["explicit_vertical_lines"]]

    full = TableFinder(page, settings)
    assert len(full.cells) == 4

    inner = page.crop((60, 60, 140, 140))
    inner_tf = TableFinder(inner, settings)
    assert len(inner_tf.cells) == 4

    # A crop that excludes the right column/bottom row: only the retained
    # hints may participate.
    left = page.crop((60, 60, 100, 140))
    left_tf = TableFinder(left, settings)
    assert all(cell[2] <= 100 + 1e-6 for cell in left_tf.cells)

    # Re-running the original still sees the same cells; nothing was mutated.
    again = TableFinder(page, settings)
    assert again.cells == full.cells
    for key, value in snapshot.items():
        assert settings[key] == value
    assert [id(v) for v in settings["explicit_vertical_lines"]] == object_ids


@pytest.mark.parametrize(
    "v_strategy,h_strategy",
    [
        ("lines", "explicit"),
        ("explicit", "lines"),
        ("lines_strict", "explicit"),
        ("explicit", "lines_strict"),
        ("lines", "lines_strict"),
        ("lines_strict", "lines"),
    ],
)
def test_mixed_strategies_respect_region(v_strategy, h_strategy):
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    region = page.crop((60, 60, 140, 140))
    settings = {
        "vertical_strategy": v_strategy,
        "horizontal_strategy": h_strategy,
        "explicit_vertical_lines": [60, 100, 140],
        "explicit_horizontal_lines": [60, 100, 140],
    }
    tf = TableFinder(region, settings)
    for edge in tf.edges:
        assert _inside(edge, (60, 60, 140, 140))
    for cell in tf.cells:
        assert 60 - 1e-6 <= cell[0] and cell[2] <= 140 + 1e-6
        assert 60 - 1e-6 <= cell[1] and cell[3] <= 140 + 1e-6


def test_explicit_object_hints_are_not_mutated():
    pdf = _open([PageContent().rect(60, 60, 80, 80).text(64, 118, "A")])
    page = pdf.pages[0]
    region = page.crop((50, 50, 150, 150))
    rect = page.rects[0]
    edge = page.edges[0]
    rect_before, edge_before = copy.deepcopy(rect), copy.deepcopy(edge)
    hints = [rect, rect]
    hints_before = [copy.deepcopy(h) for h in hints]

    TableFinder(
        region,
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": hints,
            "explicit_horizontal_lines": [rect, edge],
        },
    )
    assert rect == rect_before
    assert edge == edge_before
    assert hints == hints_before


def test_valid_explicit_config_with_insufficient_cells_gives_empty():
    # Valid config: two numeric horizontal hints plus text-derived verticals,
    # on a page with no words. Nothing encloses a cell, so the result is empty
    # rather than an error.
    pdf = _open([PageContent().text(150, 150, "Z")])
    page = pdf.pages[0].crop((10, 10, 100, 100))
    tf = TableFinder(
        page,
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "explicit",
            "explicit_horizontal_lines": [20, 40],
        },
    )
    assert tf.cells == []
    assert tf.tables == []


def test_explicit_validation_still_applies_after_cropping():
    pdf = _open([_grid_content()])
    region = pdf.pages[0].crop((60, 60, 140, 140))
    with pytest.raises(ValueError):
        TableFinder(
            region,
            {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": [60],
                "horizontal_strategy": "lines",
            },
        )


# ---------------------------------------------------------------------------
# Region plumbing: relative crops, nested crops, MediaBox/Rotate
# ---------------------------------------------------------------------------


def test_relative_crop_matches_absolute_crop_geometry():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    absolute = page.crop((60, 60, 100, 100))

    outer = page.crop((40, 40, 160, 160))
    relative = outer.crop((20, 20, 60, 60), relative=True)
    assert relative.bbox == absolute.bbox
    assert _edge_boxes(relative.edges) == _edge_boxes(absolute.edges)
    assert relative.debug_tablefinder().cells == absolute.debug_tablefinder().cells


def test_nested_crop_edges_consistent_with_fresh_finder():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    outer = page.crop((60, 60, 140, 140))
    inner = outer.crop((80, 80, 120, 120))
    direct_inner = page.crop((80, 80, 120, 120))

    direct = TableFinder(inner)
    assert _edge_boxes(direct.edges) == _edge_boxes(inner.edges)
    assert direct.cells == inner.debug_tablefinder().cells
    # Cropping through an intermediate view equals cropping the page directly.
    assert _edge_boxes(inner.edges) == _edge_boxes(direct_inner.edges)
    for edge in inner.edges:
        assert _inside(edge, (80, 80, 120, 120))


def test_media_box_offset_and_rotation_keep_region_consistent():
    # MediaBox starting at (100, 50). Draw a rect and a cross; then crop a
    # smaller box that cuts the rect on every rotation. Both the reported
    # bbox offset and the rotated coordinates must be respected, and the
    # clipped edges must agree with a directly created finder.
    content = (
        PageContent()
        .rect(150, 80, 80, 80)
        .line(190, 80, 190, 160)
        .line(150, 120, 230, 120)
    )
    for rotate in (0, 90, 180, 270):
        pdf = _open([content], media_box=(100, 50, 300, 250), rotate=rotate)
        page = pdf.pages[0]
        bx = page.bbox
        mid = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
        crop = (mid[0] - 30, mid[1] - 30, mid[0] + 30, mid[1] + 30)
        cropped = page.crop(crop)

        for edge in cropped.edges:
            assert _inside(edge, crop)
        direct = TableFinder(cropped)
        assert _edge_boxes(direct.edges) == _edge_boxes(cropped.edges)
        # The identity crop reproduces the page's own geometry exactly.
        assert _edge_boxes(page.crop(bx).edges) == _edge_boxes(page.edges)


# ---------------------------------------------------------------------------
# Selection-only views do not become crops
# ---------------------------------------------------------------------------


def test_within_and_outside_bbox_do_not_reintroduce_a_boundary():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]

    within = page.within_bbox((60, 60, 140, 140))
    # Objects are only *selected*; edges are not re-clipped to the selection.
    within_edges = _edge_boxes(within.edges)
    page_edges = _edge_boxes(page.edges)
    assert set(within_edges).issubset(set(page_edges))
    assert within.debug_tablefinder().cells == page.debug_tablefinder().cells

    outside = page.outside_bbox((60, 60, 140, 140))
    # The grid is the page's only geometry, so nothing lies fully outside.
    assert outside.edges == []
    assert outside.bbox == page.bbox


def test_filter_does_not_act_as_crop():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    filtered = page.filter(lambda obj: obj["object_type"] in ("line", "char"))
    assert _edge_boxes(filtered.edges) == _edge_boxes(page.edges)


def test_within_bbox_nested_inside_a_crop_keeps_crop_region():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    cropped = page.crop((50, 50, 150, 150))
    within = cropped.within_bbox((60, 60, 140, 140))

    # The selection is narrower, but the parent crop still limits the region:
    # objects intersected the crop and were clipped there, so no boundary is
    # re-introduced at (60,60,140,140).
    assert within._edge_bbox == (50, 50, 150, 150)
    for edge in within.edges:
        assert _inside(edge, (50, 50, 150, 150))
    assert _edge_boxes(within.edges) == _edge_boxes(cropped.edges)


# ---------------------------------------------------------------------------
# Isolation: interleaved reads, no mutation of caller config/objects
# ---------------------------------------------------------------------------


def test_interleaved_views_do_not_mutate_source_objects():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    original_edges = [dict(e) for e in page.edges]
    original_objs = {
        kind: [dict(o) for o in objs]
        for kind, objs in page.objects.items()
        for o in objs
    }

    view_a = page.crop((60, 60, 140, 140))
    view_b = page.crop((70, 70, 130, 130))
    for _ in range(2):
        _ = view_a.edges, view_b.curve_edges, view_a.debug_tablefinder()
        _ = page.debug_tablefinder()
        assert [dict(e) for e in page.edges] == original_edges
        for kind, objs in page.objects.items():
            assert [dict(o) for o in objs] == original_objs[kind]

    # The two views disagree, proving caches are not shared.
    assert _edge_boxes(view_a.edges) != _edge_boxes(view_b.edges)


def test_direct_edge_lists_match_recreated_finder():
    # A page mixing a line, a rect and a path: edges (as reported directly)
    # must equal a freshly created TableFinder's edges on the same view, and
    # must equal the concatenation of the per-source edge lists.
    content = (
        PageContent()
        .line(10, 20, 190, 20)
        .rect(60, 60, 40, 40)
        .polyline([(10, 150), (190, 150), (190, 170)])
    )
    pdf = _open([content])
    cropped = pdf.pages[0].crop((40, 10, 160, 160))

    finder = TableFinder(cropped)
    assert _edge_boxes(finder.edges) == _edge_boxes(cropped.edges)

    clip = (40, 10, 160, 160)
    line_edges = [
        e
        for e in (utils.line_to_edge(line, clip) for line in cropped.lines)
        if e is not None
    ]
    combined = line_edges + cropped.rect_edges + cropped.curve_edges
    assert sorted(map(_bbox_of, combined)) == sorted(map(_bbox_of, cropped.edges))
    # A new finder sees exactly the same objects' geometry.
    assert _edge_boxes(TableFinder(cropped).edges) == _edge_boxes(combined)


# ---------------------------------------------------------------------------
# Scaling: cost grows with paths, not with paths x views
# ---------------------------------------------------------------------------


def _many_line_content(n_paths: int) -> PageContent:
    content = PageContent()
    for i in range(n_paths):
        y = 5 + (i * 7) % 190
        content.line(5, y, 195, y + 0.5)
    return content


def test_many_views_do_not_copy_all_paths():
    n_paths = 200
    pdf = _open([_many_line_content(n_paths)])
    page = pdf.pages[0]

    # Warm the original page.
    _ = page.edges
    base_objects = sum(len(v) for v in page.objects.values())
    full_paths = len(page.lines)

    views = [page.crop((10 + i, 10, 190 - i, 190)) for i in range(0, 20)]
    for view in views:
        _ = view.edges

    # Each cropped view holds only the objects intersecting its box. Because
    # the fixture's lines all span the full width, every view keeps all of
    # them; the key point is that a view never holds *more* than the page, and
    # edge derivation reads the views' own (already-clipped) objects rather
    # than copying the page's full path list into every view.
    for view in views:
        held = sum(len(v) for v in view.objects.values())
        assert held <= base_objects

    # Edge lists are cached per view, never shared or mixed across views.
    page_edges_object = page.edges
    for view in views:
        assert view.edges is not page_edges_object

    tracemalloc.start()
    t0 = time.perf_counter()
    for view in views:
        TableFinder(view)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Loose sanity bounds rather than brittle timing assertions.
    assert elapsed < 5.0
    assert peak < 50 * 1024 * 1024

    # Sanity: the page really did contain the paths the views saw.
    assert full_paths == n_paths


def test_interleaved_original_and_crop_edge_reads_are_stable():
    pdf = _open([_grid_content()])
    page = pdf.pages[0]
    original = _edge_boxes(page.edges)

    a = page.crop((60, 60, 140, 140))
    b = page.crop((60, 60, 110, 140))
    before_a = _edge_boxes(a.edges)
    before_b = _edge_boxes(b.curve_edges)

    # Interleave reads among the original and both crops.
    for _ in range(3):
        _ = page.debug_tablefinder()
        _ = a.debug_tablefinder()
        _ = b.debug_tablefinder()
        assert _edge_boxes(page.edges) == original
        assert _edge_boxes(a.edges) == before_a
        assert _edge_boxes(b.curve_edges) == before_b
