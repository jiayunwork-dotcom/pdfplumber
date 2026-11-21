"""Minimal PDF writer for geometry tests.

pdfplumber's test-suite ships static PDFs, but the crop/table cases here need
shapes whose exact coordinates are known independently of pdfplumber. Drawing
them with a tiny in-repo writer keeps those expectations checkable without
adding a dependency (ReportLab is not a test requirement) and lets the same
fixtures cover unusual MediaBox origins and page rotation.

Everything here is expressed in *PDF user space*: the origin is the
lower-left corner and ``y`` grows upward. pdfplumber reports the flipped,
top-left coordinates.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

T_num = Union[int, float]


def _fmt(value: T_num) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PageContent:
    """Builds one page's content stream, in PDF user-space coordinates."""

    def __init__(self) -> None:
        self._ops: List[str] = []

    def rect(self, x: T_num, y: T_num, w: T_num, h: T_num) -> "PageContent":
        """Stroked rectangle, lower-left corner ``(x, y)``."""
        self._ops.append(f"{_fmt(x)} {_fmt(y)} {_fmt(w)} {_fmt(h)} re S")
        return self

    def line(self, x0: T_num, y0: T_num, x1: T_num, y1: T_num) -> "PageContent":
        """A single stroked straight segment."""
        self._ops.append(f"{_fmt(x0)} {_fmt(y0)} m {_fmt(x1)} {_fmt(y1)} l S")
        return self

    def polyline(
        self, points: Sequence[Tuple[T_num, T_num]], close: bool = False
    ) -> "PageContent":
        """A stroked open (or explicitly closed) straight-line path."""
        x0, y0 = points[0]
        parts = [f"{_fmt(x0)} {_fmt(y0)} m"]
        for x, y in points[1:]:
            parts.append(f"{_fmt(x)} {_fmt(y)} l")
        parts.append("h S" if close else "S")
        self._ops.append(" ".join(parts))
        return self

    def text(
        self,
        x: T_num,
        y: T_num,
        value: str,
        size: T_num = 8,
        font: str = "F1",
    ) -> "PageContent":
        """Text with its baseline at ``(x, y)``."""
        self._ops.append(
            f"BT /{font} {_fmt(size)} Tf {_fmt(x)} {_fmt(y)} Td "
            f"({_escape(value)}) Tj ET"
        )
        return self

    @property
    def stream(self) -> str:
        return "\n".join(self._ops) + "\n"


def build_pdf(
    pages: Sequence[PageContent],
    media_box: Optional[Tuple[T_num, T_num, T_num, T_num]] = None,
    rotate: int = 0,
    fonts: Optional[Dict[str, str]] = None,
) -> bytes:
    """Assemble one PDF.

    ``media_box`` defaults to ``(0, 0, 200, 200)``.  ``rotate`` applies the
    page's ``/Rotate`` entry to every page (90/180/270 are all supported by
    pdfplumber).
    """
    if media_box is None:
        media_box = (0, 0, 200, 200)
    fonts = fonts or {"F1": "Helvetica"}

    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_refs = {}
    for name, family in fonts.items():
        font_refs[name] = add(
            (
                f"<< /Type /Font /Subtype /Type1 /BaseFont /{family} "
                f"/Encoding /WinAnsiEncoding >>"
            ).encode("latin-1")
        )

    font_dict = " ".join(f"/{name} {ref} 0 R" for name, ref in font_refs.items())
    mb = " ".join(_fmt(v) for v in media_box)

    page_refs: List[int] = []
    page_objects: List[bytes] = []
    for content in pages:
        data = content.stream.encode("latin-1")
        stream_ref = add(
            b"<< /Length "
            + str(len(data)).encode()
            + b" >>\nstream\n"
            + data
            + b"endstream"
        )
        page_objects.append(
            (
                f"<< /Type /Page /Parent PAGES_REF /MediaBox [{mb}] "
                f"/Rotate {rotate} /Resources << /Font << {font_dict} >> >> "
                f"/Contents {stream_ref} 0 R >>"
            ).encode("latin-1")
        )

    # Pages object needs to reference its kids, whose ids are only known
    # after allocation; allocate a placeholder then substitute.
    pages_obj_index = len(objects) + 1 + len(page_objects)
    page_refs = []
    first_page_id = len(objects) + 1
    for i in range(len(page_objects)):
        page_refs.append(first_page_id + i)

    for body in page_objects:
        add(body.replace(b"PAGES_REF", str(pages_obj_index).encode()))

    add(
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{ref} 0 R" for ref in page_refs)
            + f"] /Count {len(page_refs)} >>"
        ).encode("latin-1")
    )

    catalog_ref = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj_index)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_ref} 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
