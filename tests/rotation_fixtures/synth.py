"""Independent synthetic-PDF generator for the rotation tests.

This module deliberately does **not** import ``pdfplumber``.  It writes a
minimal PDF by hand, using only Adobe core-font metrics (via ``pdfminer``,
which is a runtime dependency) to compute glyph advance widths.  The tests
compute their expectations from the logical matrices and placement
transforms recorded here, never by calling the new extraction logic.
"""

import math
from typing import List, Optional, Sequence, Tuple

from pdfminer.fontmetrics import FONT_METRICS

# Font metrics: Helvetica glyph widths are expressed in 1/1000 em and the
# pdfminer glyph box is (0, descent, advance, descent + size) in text space.
_FONT = "Helvetica"
_PROPS, _WIDTHS = FONT_METRICS[_FONT]
DESCENT = _PROPS["Descent"] / 1000.0
ASCENT = _PROPS["Ascent"] / 1000.0

PAGE_WIDTH = 595.2756
PAGE_HEIGHT = 841.8898

FONT_SIZE = 10.0
COLUMN_PITCH = 90.0
ROW_PITCH = 34.0
CELL_PAD = 8.0
FIRST_BASELINE = 12.0
INTRA_CELL_LINE_GAP = 10.0


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _num(value: float) -> str:
    out = f"{value:.6f}".rstrip("0").rstrip(".")
    return out if out not in ("", "-0") else "0"


def char_advance(ch: str, size: float = FONT_SIZE) -> float:
    return _WIDTHS.get(ch, 500) / 1000.0 * size


def text_width(text: str, size: float = FONT_SIZE) -> float:
    return sum(char_advance(ch, size) for ch in text)


def place(
    rotation: int, u: float, v: float, ox: float, oy: float
) -> Tuple[float, float]:
    """Map a logical cell position to PDF user space.

    ``u`` runs along the baseline (reading direction), ``v`` runs visually
    downward (towards later lines).  ``oy`` is the PDF-space y (upward) of
    the table's first baseline at ``u = v = 0``.
    """

    rotation %= 360
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # u is the glyph baseline direction; v is "visually down" = -normal.
    return (ox + u * cos_t + v * sin_t, oy + u * sin_t - v * cos_t)


def char_bboxes(
    text: str,
    rotation: int,
    u: float,
    v: float,
    ox: float,
    oy: float,
    size: float = FONT_SIZE,
) -> List[Tuple[str, float, float, float, float]]:
    """Independent expected char boxes in pdfplumber page coordinates.

    Returns ``(char, x0, top, x1, bottom)``.  Page coordinates have the
    origin at the top-left, so ``top = PAGE_HEIGHT - pdf_y``.
    """

    boxes = []
    cursor = u
    # In text space (y up from the baseline) pdfminer uses
    # (0, descent, advance, descent + size).  In this module's visual "v"
    # coordinates (v down) the box spans [-descent - size, -descent].
    glyph_v_top = -(size + DESCENT * size)
    glyph_v_bottom = -DESCENT * size
    for ch in text:
        advance = char_advance(ch, size)
        corners = []
        for du, dv in (
            (0.0, glyph_v_top),
            (advance, glyph_v_top),
            (0.0, glyph_v_bottom),
            (advance, glyph_v_bottom),
        ):
            pdf_x, pdf_y = place(rotation, cursor + du, v + dv, ox, oy)
            corners.append((pdf_x, PAGE_HEIGHT - pdf_y))
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        boxes.append((ch, min(xs), min(ys), max(xs), max(ys)))
        cursor += advance
    return boxes


class RawPDF:
    """A minimal, dependency-light PDF writer."""

    def __init__(self, width: float = PAGE_WIDTH, height: float = PAGE_HEIGHT):
        self.width = width
        self.height = height
        self.pages: List[Tuple[str, int]] = []

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = FONT_SIZE,
        rotation: int = 0,
    ) -> str:
        theta = math.radians(rotation)
        a, b = math.cos(theta), math.sin(theta)
        c, d = -math.sin(theta), math.cos(theta)
        return (
            f"BT /F1 {_num(size)} Tf "
            f"{_num(a)} {_num(b)} {_num(c)} {_num(d)} {_num(x)} {_num(y)} Tm "
            f"({_escape(text)}) Tj ET\n"
        )

    def add_page(self, ops: Sequence[str], rotate: int = 0) -> None:
        self.pages.append(("".join(ops), rotate))

    def build(self) -> bytes:
        objects: List[Optional[bytes]] = [None, None]
        offsets: List[int] = [0]

        def add(obj: bytes) -> int:
            objects.append(obj)
            return len(objects)

        font = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"
        )
        kids = []
        for ops, rotate in self.pages:
            content = ops.encode("latin-1")
            content_obj = add(
                b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream"
            )
            rotate_part = f" /Rotate {rotate}" if rotate else ""
            kids.append(
                add(
                    (
                        f"<< /Type /Page /Parent 2 0 R "
                        f"/MediaBox [0 0 {_num(self.width)} {_num(self.height)}]"
                        f"{rotate_part} /Resources << /Font << /F1 {font} 0 R >> >> "
                        f"/Contents {content_obj} 0 R >>"
                    ).encode("latin-1")
                )
            )
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = (
            f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
            f"/Count {len(kids)} >>"
        ).encode("latin-1")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for index, obj in enumerate(objects, 1):
            offsets.append(len(out))
            assert obj is not None
            out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        total = len(objects) + 1
        out += f"xref\n0 {total}\n".encode() + b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            out += f"{offset:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {total} /Root 1 0 R >>\n" f"startxref\n{xref}\n%%EOF\n"
        ).encode()
        return bytes(out)


class SyntheticTable:
    """One placed table: its logical matrix and its page-coordinate cells."""

    def __init__(
        self,
        matrix: List[List[str]],
        rotation: int,
        ox: float,
        oy: float,
        column_pitch: float = COLUMN_PITCH,
        row_pitch: float = ROW_PITCH,
        size: float = FONT_SIZE,
    ):
        self.matrix = matrix
        self.rotation = rotation
        self.ox = ox
        self.oy = oy
        self.column_pitch = column_pitch
        self.row_pitch = row_pitch
        self.size = size

    def draw(self, pdf: RawPDF) -> List[str]:
        ops: List[str] = []
        for row_index, row in enumerate(self.matrix):
            for column_index, cell in enumerate(row):
                if not cell:
                    continue
                for line_index, line in enumerate(cell.split("\n")):
                    u = column_index * self.column_pitch + CELL_PAD
                    v = (
                        row_index * self.row_pitch
                        + FIRST_BASELINE
                        + line_index * INTRA_CELL_LINE_GAP
                    )
                    x, y = place(self.rotation, u, v, self.ox, self.oy)
                    ops.append(pdf.text(x, y, line, self.size, self.rotation))
        return ops

    def expected_char_boxes(self) -> List[Tuple[str, float, float, float, float]]:
        boxes = []
        for row_index, row in enumerate(self.matrix):
            for column_index, cell in enumerate(row):
                if not cell:
                    continue
                for line_index, line in enumerate(cell.split("\n")):
                    u = column_index * self.column_pitch + CELL_PAD
                    v = (
                        row_index * self.row_pitch
                        + FIRST_BASELINE
                        + line_index * INTRA_CELL_LINE_GAP
                    )
                    boxes.extend(
                        char_bboxes(
                            line, self.rotation, u, v, self.ox, self.oy, self.size
                        )
                    )
        return boxes

    def expected_cell_char_boxes(
        self,
    ) -> List[List[List[Tuple[str, float, float, float, float]]]]:
        """Expected chars grouped by logical cell, mirroring ``matrix``."""
        result = []
        for row_index, row in enumerate(self.matrix):
            row_result = []
            for column_index, cell in enumerate(row):
                cell_boxes = []
                for line_index, line in enumerate(cell.split("\n")):
                    u = column_index * self.column_pitch + CELL_PAD
                    v = (
                        row_index * self.row_pitch
                        + FIRST_BASELINE
                        + line_index * INTRA_CELL_LINE_GAP
                    )
                    cell_boxes.extend(
                        char_bboxes(
                            line, self.rotation, u, v, self.ox, self.oy, self.size
                        )
                    )
                row_result.append(cell_boxes)
            result.append(row_result)
        return result


def paragraph(
    pdf: RawPDF,
    lines: Sequence[str],
    x: float,
    y: float,
    rotation: int = 0,
    size: float = FONT_SIZE,
    line_gap: float = 14.0,
) -> List[str]:
    ops = []
    for index, line in enumerate(lines):
        px, py = place(rotation, 0.0, -index * line_gap, x, y)
        ops.append(pdf.text(px, py, line, size, rotation))
    return ops
