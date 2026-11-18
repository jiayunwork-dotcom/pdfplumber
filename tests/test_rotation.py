"""Tests for ``rotation_mode="auto"`` text extraction and table discovery.

All PDFs used here are synthesised by ``tests/rotation_fixtures/synth.py``.
Expected character boxes, cell contents and logical row/column order come
from that generator's logical matrices and placement transforms; no expected
value is produced by the new extraction logic.
"""

import io
import math
import os
import sys

import pytest

import pdfplumber
from pdfplumber.utils.rotation import (
    UnsupportedRotationError,
    from_local_box,
    matrix_rotation,
    to_local_box,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rotation_fixtures"))

import synth  # noqa: E402

AUTO_TABLE = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "rotation_mode": "auto",
}

MATRIX = [
    ["Description", "Qty", "Amount"],
    ["Widget assembly", "12", "-3.5"],
    ["Bolt", "", "128.75"],
    ["Nut and washer", "7", "0"],
    ["Gearbox", "25", "-40.25"],
]


def _open(pdf):
    return pdfplumber.open(io.BytesIO(pdf.build()))


def _single(matrix, rotation, ox, oy, **kwargs):
    pdf = synth.RawPDF()
    table = synth.SyntheticTable(matrix, rotation, ox, oy, **kwargs)
    pdf.add_page(table.draw(pdf))
    return pdf, table


def _normalize(matrix):
    """Drop rows/columns that are entirely empty in the generated matrix."""

    rows = [row for row in matrix if any(cell for cell in row)]
    if not rows:
        return []
    width = len(rows[0])
    keep = [column for column in range(width) if any(row[column] for row in rows)]
    return [[row[column] for column in keep] for row in rows]


def _strip_blank_rows(table):
    return [row for row in table if any(cell for cell in row)]


def _assert_char_grid(page, synthetic):
    """Check page bboxes of every char, in generated order."""

    expected = synthetic.expected_char_boxes()
    actual = [
        (char["text"], char["x0"], char["top"], char["x1"], char["bottom"])
        for char in page.chars
    ]
    assert len(actual) == len(expected)
    for want, got in zip(expected, actual):
        assert got[0] == want[0]
        for a, b in zip(want[1:], got[1:]):
            assert abs(a - b) < 1e-6, (want, got)


class TestRotationGeometry:
    def test_matrix_rotations(self):
        for angle in (0, 90, 180, 270):
            theta = math.radians(angle)
            matrix = (
                math.cos(theta),
                math.sin(theta),
                -math.sin(theta),
                math.cos(theta),
                10,
                20,
            )
            assert matrix_rotation(matrix) == angle

    def test_float_noise_is_accepted(self):
        # cos(pi/2) is ~6.1e-17, not exactly zero.
        assert matrix_rotation((6.12e-17, 1.0, -1.0, 6.12e-17, 0, 0)) == 90

    @pytest.mark.parametrize(
        "matrix",
        [
            (1, 0.2, 0.0, 1, 0, 0),  # shear
            (1, 0, 0.3, 1, 0, 0),  # shear
            (-1, 0, 0, 1, 0, 0),  # mirror
            (1, 0, 0, -1, 0, 0),  # mirror
            (0.7, 0.7, -0.7, 0.7, 0, 0),  # 45 degrees
            (1, 0, 0, 0, 0, 0),  # degenerate
        ],
    )
    def test_unsupported_matrices(self, matrix):
        with pytest.raises(UnsupportedRotationError):
            matrix_rotation(matrix)

    def test_local_box_roundtrip(self):
        boxes = [(10, 20, 50, 60), (0, 0, 100, 200), (-30, 5, 80, 90)]
        for rotation in (0, 90, 180, 270):
            for box in boxes:
                assert from_local_box(to_local_box(box, rotation), rotation) == box


class TestAutoWords:
    @pytest.mark.parametrize(
        "rotation,origin",
        [(0, (80, 700)), (90, (80, 300)), (180, (500, 300)), (270, (500, 700))],
    )
    def test_words_have_expected_rotation_and_text(self, rotation, origin):
        pdf, synthetic = _single(MATRIX, rotation, *origin)
        with _open(pdf) as doc:
            page = doc.pages[0]
            _assert_char_grid(page, synthetic)
            words = page.extract_words(rotation_mode="auto", return_chars=True)
            assert words
            assert {word["rotation"] for word in words} == {rotation}
            # Letters inside each word read in the table's direction.
            for word in words:
                assert word["direction"] == "ltr"
                assert "".join(char["text"] for char in word["chars"]) == word["text"]
            joined = {(word["text"], word["rotation"]) for word in words}
            for expected in ("Description", "Widget", "assembly", "-40.25"):
                assert (expected, rotation) in joined

    def test_opposite_orientations_are_never_merged(self):
        pdf = synth.RawPDF()
        # Two words a hair apart but rotated 0 and 180 degrees.
        pdf.add_page(
            [
                pdf.text(100, 700, "abc", 10, 0),
                pdf.text(100, 690, "xyz", 10, 180),
            ]
        )
        with _open(pdf) as doc:
            page = doc.pages[0]
            words = page.extract_words(rotation_mode="auto")
            by_rotation = {word["rotation"]: word["text"] for word in words}
            assert by_rotation[0] == "abc"
            assert by_rotation[180] == "xyz"

    def test_return_chars_preserves_identity_and_bboxes(self):
        pdf, synthetic = _single(MATRIX, 90, 80, 300)
        with _open(pdf) as doc:
            page = doc.pages[0]
            before = [dict(char) for char in page.chars]
            words = page.extract_words(rotation_mode="auto", return_chars=True)
            page_ids = {id(char) for char in page.chars}
            for word in words:
                for char in word["chars"]:
                    assert id(char) in page_ids
            assert [dict(char) for char in page.chars] == before

    def test_legacy_default_is_unchanged(self):
        pdf, _synthetic = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            legacy = page.extract_words()
            assert all("rotation" not in word for word in legacy)
            explicit = page.extract_words(rotation_mode="legacy")
            assert legacy == explicit

    def test_invalid_rotation_mode(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            with pytest.raises(ValueError):
                doc.pages[0].extract_words(rotation_mode="sometimes")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"line_dir": "btt"},
            {"char_dir": "rtl"},
            {"line_dir_rotated": "ltr"},
            {"char_dir_rotated": "btt"},
            {"use_text_flow": True},
        ],
    )
    def test_unsupported_auto_text_settings(self, kwargs):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            with pytest.raises(ValueError):
                doc.pages[0].extract_words(rotation_mode="auto", **kwargs)

    def test_unsupported_transform_is_rejected(self):
        pdf = synth.RawPDF()
        pdf.add_page(
            [
                "BT /F1 10 Tf 1 0.2 0 1 100 700 Tm (Sheared) Tj ET\n",
                "BT /F1 10 Tf -1 0 0 1 300 700 Tm (Mirror) Tj ET\n",
            ]
        )
        with _open(pdf) as doc:
            page = doc.pages[0]
            with pytest.raises(ValueError):
                page.extract_words(rotation_mode="auto")
            # Legacy extraction is unaffected.
            assert page.extract_words()

    def test_repeated_calls_are_isolated(self):
        pdf, synthetic = _single(MATRIX, 180, 500, 300)
        with _open(pdf) as doc:
            page = doc.pages[0]
            first = page.extract_words(rotation_mode="auto", return_chars=True)
            legacy = page.extract_words()
            second = page.extract_words(rotation_mode="auto", return_chars=True)
            assert [word["text"] for word in first] == [word["text"] for word in second]
            assert page.extract_words() == legacy


class TestAutoTables:
    @pytest.mark.parametrize(
        "rotation,origin",
        [(0, (80, 700)), (90, (80, 300)), (180, (500, 300)), (270, (500, 700))],
    )
    def test_single_direction(self, rotation, origin):
        pdf, synthetic = _single(MATRIX, rotation, *origin)
        with _open(pdf) as doc:
            page = doc.pages[0]
            tables = page.find_tables(AUTO_TABLE)
            assert len(tables) == 1
            table = tables[0]
            assert table.rotation == rotation
            assert len(table.rows) == len(MATRIX)
            assert len(table.columns) == len(MATRIX[0])
            assert table.extract() == _normalize(MATRIX)
            _assert_char_grid(page, synthetic)

    def test_mixed_0_and_90(self):
        first = [
            ["Code", "Qty", "Amt"],
            ["AA", "1", "2"],
            ["BB", "3", "4"],
            ["CC", "5", "6"],
        ]
        second = [
            ["Name", "Val", "Tag"],
            ["XX", "7", "8"],
            ["YY", "9", "10"],
            ["ZZ", "11", "12"],
        ]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(first, 0, 60, 700).draw(pdf)
        ops += synth.SyntheticTable(second, 90, 500, 700).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(AUTO_TABLE)
            assert len(tables) == 2
            by_rotation = {table.rotation: table.extract() for table in tables}
            assert by_rotation[0] == first
            assert by_rotation[90] == second

    def test_mixed_90_and_270(self):
        first = [
            ["Code", "Qty", "Amt"],
            ["AA", "1", "2"],
            ["BB", "3", "4"],
            ["CC", "5", "6"],
        ]
        second = [
            ["Name", "Val", "Tag"],
            ["XX", "7", "8"],
            ["YY", "9", "10"],
            ["ZZ", "11", "12"],
        ]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(first, 90, 520, 350).draw(pdf)
        ops += synth.SyntheticTable(second, 270, 100, 600).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(AUTO_TABLE)
            assert len(tables) == 2
            by_rotation = {table.rotation: table.extract() for table in tables}
            assert by_rotation[90] == first
            assert by_rotation[270] == second

    def test_side_by_side_same_direction(self):
        first = [
            ["Code", "Qty", "Amt"],
            ["AA", "1", "2"],
            ["BB", "3", "4"],
            ["CC", "5", "6"],
        ]
        second = [
            ["Name", "Val", "Tag"],
            ["XX", "7", "8"],
            ["YY", "9", "10"],
            ["ZZ", "11", "12"],
        ]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(first, 0, 60, 700, column_pitch=80).draw(pdf)
        # Gap = two column pitches, i.e. twice normal column spacing.
        ops += synth.SyntheticTable(
            second, 0, 60 + 3 * 80 + 200, 700, column_pitch=80
        ).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(AUTO_TABLE)
            assert len(tables) == 2
            extracted = sorted((table.extract() for table in tables), key=str)
            assert extracted == sorted([first, second], key=str)

    def test_paragraph_and_header_not_absorbed(self):
        pdf = synth.RawPDF()
        synthetic = synth.SyntheticTable(MATRIX, 0, 80, 700)
        ops = synthetic.draw(pdf)
        ops += synth.paragraph(
            pdf,
            [
                "This paragraph is beside the table and should",
                "not be absorbed, no matter how many words it has",
                "or how consistently it is left aligned.",
            ],
            420,
            700,
        )
        ops.append(pdf.text(80, 800, "A plain page header spanning the top", 10, 0))
        pdf.add_page(ops)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(AUTO_TABLE)
            assert len(tables) == 1
            assert tables[0].extract() == _normalize(MATRIX)

    def test_multiline_cells(self):
        matrix = [
            ["Name", "Qty", "Note"],
            ["Widget assembly", "12", "first line\nsecond line"],
            ["Bolt", "3", "plain"],
            ["Nut", "7", "other"],
        ]
        pdf, _synthetic = _single(matrix, 0, 80, 700)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(AUTO_TABLE)
            assert len(tables) == 1
            assert tables[0].extract() == matrix

    def test_rows_and_columns_use_logical_order(self):
        for rotation, origin in [
            (0, (80, 700)),
            (90, (80, 300)),
            (180, (500, 300)),
            (270, (500, 700)),
        ]:
            pdf, _ = _single(MATRIX, rotation, *origin)
            with _open(pdf) as doc:
                table = doc.pages[0].find_tables(AUTO_TABLE)[0]
                # Every row/column has one cell per logical column/row.
                for row in table.rows:
                    assert len(row.cells) == len(MATRIX[0])
                for column in table.columns:
                    assert len(column.cells) == len(MATRIX)
                # Bounding box order is logical even when page coordinates
                # decrease (rotation 180) or swap (90/270).
                first_row_top = min(cell[1] for cell in table.rows[0].cells)
                last_row_top = min(cell[1] for cell in table.rows[-1].cells)
                if rotation == 0:
                    assert first_row_top < last_row_top
                elif rotation == 180:
                    assert first_row_top > last_row_top

    def test_bbox_cells_use_page_coordinates(self):
        pdf, _ = _single(MATRIX, 90, 80, 300)
        with _open(pdf) as doc:
            page = doc.pages[0]
            table = page.find_tables(AUTO_TABLE)[0]
            # All page cells lie inside the table's page bbox.
            x0, top, x1, bottom = table.bbox
            for cell in table.cells:
                assert x0 - 1e-6 <= cell[0] and cell[2] <= x1 + 1e-6
                assert top - 1e-6 <= cell[1] and cell[3] <= bottom + 1e-6

    def test_direct_table_extract_needs_no_rotation_argument(self):
        pdf, _ = _single(MATRIX, 270, 500, 700)
        with _open(pdf) as doc:
            table = doc.pages[0].find_tables(AUTO_TABLE)[0]
            assert table.extract() == _normalize(MATRIX)

    def test_find_table_and_extract_table_largest(self):
        big = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
        small = [["x", "y", "z"], ["1", "2", "3"]]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(small, 0, 60, 700).draw(pdf)
        ops += synth.SyntheticTable(big, 0, 60, 550).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            page = doc.pages[0]
            assert page.find_table(AUTO_TABLE).extract() == big
            assert page.extract_table(AUTO_TABLE) == big
            assert len(page.extract_tables(AUTO_TABLE)) == 2

    def test_tie_break_is_stable(self):
        matrix = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(matrix, 0, 60, 700).draw(pdf)
        ops += synth.SyntheticTable(matrix, 0, 60, 500).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            page = doc.pages[0]
            tables = page.find_tables(AUTO_TABLE)
            expected = sorted(
                tables, key=lambda t: (-len(t.cells), t.bbox[1], t.bbox[0])
            )[0]
            assert page.find_table(AUTO_TABLE).bbox == expected.bbox
            # Repeated calls return the same selection.
            assert page.find_table(AUTO_TABLE).bbox == expected.bbox

    def test_auto_table_requires_text_strategies(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            for strategies in [
                {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                {"vertical_strategy": "text", "horizontal_strategy": "lines_strict"},
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "explicit",
                },
                {"vertical_strategy": "explicit", "horizontal_strategy": "text"},
            ]:
                settings = dict(strategies, rotation_mode="auto")
                with pytest.raises(ValueError):
                    page.find_tables(settings)

    def test_auto_table_rejects_unsupported_text_settings(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            for extra in [
                {"text_line_dir": "btt"},
                {"text_char_dir": "rtl"},
                {"text_use_text_flow": True},
            ]:
                settings = dict(AUTO_TABLE, **extra)
                with pytest.raises(ValueError):
                    page.find_tables(settings)

    def test_auto_table_rejects_legacy_text_rotation(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            with pytest.raises(ValueError):
                doc.pages[0].find_tables(dict(AUTO_TABLE, text_rotation_mode="legacy"))

    def test_text_prefix_can_enable_auto(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            tables = doc.pages[0].find_tables(
                {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "text_rotation_mode": "auto",
                }
            )
            assert len(tables) == 1
            assert tables[0].extract() == _normalize(MATRIX)

    def test_invalid_rotation_mode_in_table_settings(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            with pytest.raises(ValueError):
                doc.pages[0].find_tables(dict(AUTO_TABLE, rotation_mode="nope"))


class TestCroppingAndPageRotation:
    def test_crop_preserves_table(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            table = page.find_tables(AUTO_TABLE)[0]
            cropped = page.crop(table.bbox)
            tables = cropped.find_tables(AUTO_TABLE)
            assert len(tables) == 1
            assert tables[0].rotation == table.rotation
            assert tables[0].extract() == table.extract()

    def test_relative_crop(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            table = page.find_tables(AUTO_TABLE)[0]
            x0, top, x1, bottom = table.bbox
            relative = (
                x0 - page.bbox[0],
                top - page.bbox[1],
                x1 - page.bbox[0],
                bottom - page.bbox[1],
            )
            cropped = page.crop(relative, relative=True)
            assert cropped.extract_table(AUTO_TABLE) == table.extract()

    def test_crop_does_not_mutate_table_or_page(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            table = page.find_tables(AUTO_TABLE)[0]
            before_rotation = table.rotation
            before_extract = table.extract()
            before_chars = [dict(char) for char in page.chars]
            page.crop(table.bbox)
            assert table.rotation == before_rotation
            assert table.extract() == before_extract
            assert [dict(char) for char in page.chars] == before_chars

    @pytest.mark.parametrize("page_rotate", [90, 180, 270])
    def test_page_rotate(self, page_rotate):
        pdf = synth.RawPDF()
        # A visually unrotated table on a page with /Rotate set: the word
        # rotation must account for the page rotation.
        ops = synth.SyntheticTable(MATRIX, 0, 100, 650).draw(pdf)
        pdf.add_page(ops, rotate=page_rotate)
        with _open(pdf) as doc:
            page = doc.pages[0]
            assert page.rotation == page_rotate
            tables = page.find_tables(AUTO_TABLE)
            assert len(tables) == 1
            assert tables[0].extract() == _normalize(MATRIX)


class TestAmbiguity:
    def _ambiguous_pdf(self):
        first = [["A1", "A2", "A3"], ["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        second = [["B1", "B2", "B3"], ["j", "k", "l"], ["m", "n", "o"], ["p", "q", "r"]]
        pdf = synth.RawPDF()
        # Same rotation, overlapping bounding boxes and partially shared
        # columns: the competing tables cannot be separated unambiguously.
        ops = synth.SyntheticTable(
            first, 0, 100, 700, column_pitch=90, row_pitch=24
        ).draw(pdf)
        ops += synth.SyntheticTable(
            second, 0, 250, 660, column_pitch=90, row_pitch=24
        ).draw(pdf)
        pdf.add_page(ops)
        return pdf

    def test_overlap_raises_named_error(self):
        from pdfplumber.rotation_table import AmbiguousTableError

        with _open(self._ambiguous_pdf()) as doc:
            page = doc.pages[0]
            with pytest.raises(AmbiguousTableError) as excinfo:
                page.find_tables(AUTO_TABLE)
            message = str(excinfo.value)
            assert "Ambiguous" in message
            assert excinfo.value.tables

    def test_no_arbitrary_winner(self):
        from pdfplumber.rotation_table import AmbiguousTableError

        for _ in range(3):
            with _open(self._ambiguous_pdf()) as doc:
                with pytest.raises(AmbiguousTableError):
                    doc.pages[0].find_tables(AUTO_TABLE)


class TestNoSharedCharacters:
    def test_tables_do_not_share_chars(self):
        first = [["Code", "Qty", "Amt"], ["AA", "1", "2"], ["BB", "3", "4"]]
        second = [["Name", "Val", "Tag"], ["XX", "7", "8"], ["YY", "9", "10"]]
        pdf = synth.RawPDF()
        ops = synth.SyntheticTable(first, 0, 60, 700).draw(pdf)
        ops += synth.SyntheticTable(second, 90, 500, 700).draw(pdf)
        pdf.add_page(ops)
        with _open(pdf) as doc:
            page = doc.pages[0]
            tables = page.find_tables(AUTO_TABLE)
            seen = set()
            for table in tables:
                ids = table.claimed_char_ids()
                assert not (ids & seen)
                seen |= ids


class TestScaling:
    """Scaling check: fixed table size, growing table count on one page.

    The algorithm's work grows with the number of characters, lines, column
    cores, components and candidate tables -- never with page pixel area.
    These assertions bound the *candidate* scale directly from ``stats`` and
    check that found tables are exactly the generated ones at each size.
    """

    @staticmethod
    def _build(count):
        side = 10
        cell = 420.0
        pitch = 70.0
        width = side * cell + 100
        height = side * cell + 100
        pdf = synth.RawPDF(width=width, height=height)
        ops = []
        matrices = []
        for index in range(count):
            row = index // side
            column = index % side
            matrix = [
                [f"T{index}", "Qty", "Amt"],
                ["alpha", str(index), "1.5"],
                ["beta", "-2", "3.25"],
            ]
            ox = 80 + column * cell
            oy = height - 80 - row * cell
            ops += synth.SyntheticTable(
                matrix, 0, ox, oy, column_pitch=pitch, row_pitch=30
            ).draw(pdf)
            matrices.append(matrix)
        pdf.add_page(ops)
        return pdf, matrices, width * height

    def test_count_growth(self):
        import time
        import tracemalloc

        measurements = []
        for count in (10, 40, 100):
            pdf, matrices, page_area = self._build(count)
            with _open(pdf) as doc:
                page = doc.pages[0]
                tracemalloc.start()
                started = time.perf_counter()
                finder = page.debug_tablefinder(AUTO_TABLE)
                elapsed = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                found = [table.extract() for table in finder.tables]
                assert len(found) == count
                assert sorted(found, key=str) == sorted(matrices, key=str)

                # Diagnostics must scale with content, not page area.
                assert finder.stats["words"] == count * 9
                assert finder.stats["candidates"] == count
                # Column cores may be shared by vertically stacked tables
                # that align on the same x positions, so the count is at most
                # three per table and at least three in total.
                assert 3 <= finder.stats["column_cores"] <= count * 3
                assert finder.stats["lines"] <= count * 3

                measurements.append(
                    (count, page_area, peak, finder.stats["candidates"], elapsed)
                )
                print(
                    f"tables={count} correct={len(found)} "
                    f"seconds={elapsed:.3f} peak_bytes={peak} "
                    f"words={finder.stats['words']} "
                    f"lines={finder.stats['lines']} "
                    f"cores={finder.stats['column_cores']} "
                    f"candidates={finder.stats['candidates']}"
                )

        # Same page size throughout.  Peak memory should grow roughly with
        # the number of candidate tables, not with page area or with the
        # square of the table count.
        small_peak = measurements[0][2] / measurements[0][3]
        large_peak = measurements[-1][2] / measurements[-1][3]
        assert large_peak < 4 * small_peak
        assert measurements[-1][0] == 100
        count_ratio = measurements[-1][0] / measurements[0][0]
        peak_ratio = measurements[-1][2] / measurements[0][2]
        assert peak_ratio < 2 * count_ratio


class TestAutoTextAndDebug:
    def test_extract_text_auto(self):
        matrix = [
            ["First", "Second"],
            ["alpha", "beta"],
            ["gamma", "delta"],
        ]
        pdf, _ = _single(matrix, 0, 80, 700)
        with _open(pdf) as doc:
            text = doc.pages[0].extract_text(rotation_mode="auto")
            assert text.splitlines() == [
                "First Second",
                "alpha beta",
                "gamma delta",
            ]

    def test_extract_text_auto_mixed(self):
        pdf = synth.RawPDF()
        ops = [pdf.text(80, 800, "upright", 10, 0)]
        ops.append(pdf.text(200, 700, "sideways", 10, 90))
        pdf.add_page(ops)
        with _open(pdf) as doc:
            text = doc.pages[0].extract_text(rotation_mode="auto")
            assert "upright" in text
            assert "sideways" in text
            # The upright group is emitted first, and each group reads in its
            # own direction (never reversed).
            assert text.index("upright") < text.index("sideways")
            assert "sideways"[::-1] not in text

    def test_extract_text_legacy_invalid_raises(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            with pytest.raises(ValueError):
                doc.pages[0].extract_text(rotation_mode="many")

    def test_debug_tablefinder_parity(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as doc:
            finder = doc.pages[0].debug_tablefinder(AUTO_TABLE)
            assert finder.tables
            assert finder.stats["candidates"] == len(finder.tables)
            assert finder.edges == []
            assert finder.intersections == {}
            assert len(finder.cells) == len(finder.tables[0].cells)


class TestIndependentWordBoxes:
    @staticmethod
    def _cell_word_boxes(synthetic, row_index, column_index):
        """Independent expectation: token bboxes from per-char transform."""

        cell = synthetic.matrix[row_index][column_index]
        if not cell:
            return []
        boxes = []
        for line_index, line in enumerate(cell.split("\n")):
            u = column_index * synthetic.column_pitch + synth.CELL_PAD
            v = (
                row_index * synthetic.row_pitch
                + synth.FIRST_BASELINE
                + line_index * synth.INTRA_CELL_LINE_GAP
            )
            boxes.append(
                synth.char_bboxes(
                    line, synthetic.rotation, u, v, synthetic.ox, synthetic.oy
                )
            )
        expected = []
        for line_boxes in boxes:
            token_start = None
            acc = []
            for index, box in enumerate(line_boxes):
                if box[0] == " ":
                    if token_start is not None:
                        expected.append(_box_of(acc))
                        token_start = None
                        acc = []
                    continue
                if token_start is None:
                    token_start = index
                acc.append(box)
            if token_start is not None:
                expected.append(_box_of(acc))
        return expected

    @pytest.mark.parametrize(
        "rotation,origin",
        [
            (0, (80, 700)),
            (90, (80, 300)),
            (180, (500, 300)),
            (270, (500, 700)),
        ],
    )
    def test_word_boxes_match_generator(self, rotation, origin):
        pdf, synthetic = _single(MATRIX, rotation, *origin)
        expected = []
        for row_index, row in enumerate(synthetic.matrix):
            for column_index in range(len(row)):
                expected.extend(
                    self._cell_word_boxes(synthetic, row_index, column_index)
                )
        with _open(pdf) as doc:
            words = doc.pages[0].extract_words(rotation_mode="auto", return_chars=True)
        actual = [(w["text"], w["x0"], w["top"], w["x1"], w["bottom"]) for w in words]
        assert len(actual) == len(expected)
        for want, got in zip(expected, actual):
            assert got[0] == want[0]
            for a, b in zip(want[1:], got[1:]):
                assert abs(a - b) < 1e-6, (want, got)


def _box_of(boxes):
    text = "".join(box[0] for box in boxes)
    return (
        text,
        min(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        max(box[4] for box in boxes),
    )


class TestIsolation:
    def test_interleaved_crop_and_original_reads(self):
        pdf, synthetic = _single(MATRIX, 90, 80, 300)
        with _open(pdf) as doc:
            page = doc.pages[0]
            original_legacy = page.extract_words()
            original_chars = [dict(char) for char in page.chars]
            original_edges = [dict(edge) for edge in page.edges]

            table = page.find_tables(AUTO_TABLE)[0]
            cropped = page.crop(table.bbox)
            cropped_legacy = cropped.extract_words()
            cropped_auto = cropped.extract_words(rotation_mode="auto")
            page.find_tables(AUTO_TABLE)
            cropped.find_tables(AUTO_TABLE)

            assert page.extract_words() == original_legacy
            assert [dict(char) for char in page.chars] == original_chars
            assert [dict(edge) for edge in page.edges] == original_edges
            assert cropped.extract_words() == cropped_legacy
            assert cropped.extract_words(rotation_mode="auto") == cropped_auto

    def test_original_legacy_result_matches_fresh_page(self):
        pdf, _ = _single(MATRIX, 0, 80, 700)
        with _open(pdf) as pdf_doc:
            page = pdf_doc.pages[0]
            legacy_first = page.extract_words()
            page.find_tables(AUTO_TABLE)
            page.extract_words(rotation_mode="auto", return_chars=True)
            assert page.extract_words() == legacy_first


class TestCellCharUniqueness:
    def test_single_table_cells_do_not_share_chars(self):
        matrix = [
            ["A", "B", "C"],
            ["alpha", "beta", "gamma"],
            ["one", "", "three"],
            ["four", "five", "six"],
        ]
        pdf, _ = _single(matrix, 0, 80, 700)
        with _open(pdf) as doc:
            page = doc.pages[0]
            table = page.find_tables(AUTO_TABLE)[0]
            us, vs = table._auto_local_grid()
            selected = table._auto_assign_chars()
            ids = [id(char) for cell in selected for char in cell]
            assert len(ids) == len(set(ids))
            # Every character inside the grid is assigned exactly once.
            inside = 0
            for char in page.chars:
                u0, v0, u1, v1 = to_local_box(
                    (char["x0"], char["top"], char["x1"], char["bottom"]),
                    table.rotation,
                )
                if us[0] <= (u0 + u1) / 2 < us[-1] and vs[0] <= (v0 + v1) / 2 < vs[-1]:
                    inside += 1
            assert len(ids) == inside
