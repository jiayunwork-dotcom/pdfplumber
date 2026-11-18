"""Utilities supporting ``rotation_mode="auto"``.

The helpers here do three things:

* classify a PDF text matrix into one of the four orthogonal rotations,
  rejecting sheared / mirrored / non-orthogonal transforms;
* extract words with automatic rotation grouping, using a coordinate frame
  local to each rotation so that letters are never reversed and words from
  opposite orientations are never joined;
* validate the small set of text settings that auto mode supports.

All work happens on shallow copies of the page's character dictionaries; the
dictionaries owned by :class:`pdfplumber.page.Page` are never mutated.
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from .._typing import T_bbox, T_dir, T_num, T_obj, T_obj_list
from .text import WORD_EXTRACTOR_KWARGS, WordExtractor, validate_directions

ROTATIONS = (0, 90, 180, 270)

#: Angular tolerance (degrees) for accepting a matrix as an orthogonal
#: rotation.  Ordinary floating-point noise (e.g. ``cos(pi/2)``) is far
#: below this bound, while genuine skew is far above it.
DEFAULT_ANGLE_TOLERANCE = 0.5

#: Relative tolerance used to test orthogonality and the determinant.
DEFAULT_ORTHOGONALITY_TOLERANCE = 1e-3

AUTO_LINE_DIR = "ttb"
AUTO_CHAR_DIR = "ltr"
AUTO_LINE_DIR_ROTATED = "ttb"
AUTO_CHAR_DIR_ROTATED = "ltr"

_DIRECTED_TEXT_SETTINGS = (
    "line_dir",
    "char_dir",
    "line_dir_rotated",
    "char_dir_rotated",
)


class UnsupportedRotationError(ValueError):
    """Raised for sheared, mirrored, or non-orthogonal character matrices."""


def _matrix_parts(matrix: Iterable[T_num]) -> Tuple[float, float, float, float]:
    values = list(matrix)
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def matrix_rotation(
    matrix: Iterable[T_num],
    angle_tolerance: T_num = DEFAULT_ANGLE_TOLERANCE,
    orthogonality_tolerance: T_num = DEFAULT_ORTHOGONALITY_TOLERANCE,
) -> int:
    """Return the counter-clockwise rotation (0/90/180/270) of ``matrix``.

    ``matrix`` is the six-element text matrix exposed on character objects
    (only the first four values are used).  The returned angle is the
    rotation of the text baseline relative to the final visible page.

    Raises
    ------
    UnsupportedRotationError
        If the matrix is sheared, mirrored, degenerate, or rotated by an
        angle that is not a multiple of 90 degrees.  ``angle_tolerance``
        (degrees) and ``orthogonality_tolerance`` (relative) control how much
        floating-point noise is accepted.
    """

    a, b, c, d = _matrix_parts(matrix)
    sx = math.hypot(a, b)
    sy = math.hypot(c, d)

    if sx == 0 or sy == 0:
        raise UnsupportedRotationError(
            f"Character matrix {tuple(matrix)} is degenerate (zero scale)."
        )

    dot = (a * c + b * d) / (sx * sy)
    det = (a * d - b * c) / (sx * sy)

    if abs(dot) > orthogonality_tolerance:
        raise UnsupportedRotationError(
            "Character matrix is sheared/non-orthogonal: "
            f"{tuple(matrix)} (normalized dot product={dot:.6g})."
        )
    if det <= 0 or abs(det - 1.0) > orthogonality_tolerance:
        raise UnsupportedRotationError(
            "Character matrix is mirrored or otherwise non-orthogonal: "
            f"{tuple(matrix)} (normalized determinant={det:.6g})."
        )

    theta = math.degrees(math.atan2(b, a)) % 360.0
    nearest = int(round(theta / 90.0)) * 90 % 360
    delta = min((theta - nearest) % 360.0, (nearest - theta) % 360.0)
    if delta > angle_tolerance:
        raise UnsupportedRotationError(
            "Character matrix is rotated by a non-orthogonal angle "
            f"({theta:.3f} degrees): {tuple(matrix)}."
        )

    return nearest


def char_rotation(char: T_obj) -> int:
    return matrix_rotation(char["matrix"])


def validate_rotation_mode(rotation_mode: Any) -> str:
    if rotation_mode not in ("legacy", "auto"):
        raise ValueError(
            "rotation_mode must be 'legacy' or 'auto', " f"not {rotation_mode!r}"
        )
    return str(rotation_mode)


def validate_auto_settings(settings: Dict[str, Any], prefix: str = "") -> None:
    """Validate text settings for ``rotation_mode="auto"``.

    Auto mode performs its own rotation grouping, so callers may not ask it
    to re-interpret directions or to follow the raw PDF text flow.  Only the
    default direction values (and ``use_text_flow=False``) are accepted.
    """

    for key in _DIRECTED_TEXT_SETTINGS:
        if key not in settings:
            continue
        value = settings[key]
        if key == "line_dir":
            allowed: Tuple[Any, ...] = (AUTO_LINE_DIR,)
        elif key == "char_dir":
            allowed = (AUTO_CHAR_DIR,)
        elif key == "line_dir_rotated":
            allowed = (None, AUTO_LINE_DIR_ROTATED)
        else:
            allowed = (None, AUTO_CHAR_DIR_ROTATED)
        if value not in allowed:
            raise ValueError(
                "rotation_mode='auto' only supports the default "
                f"'{prefix}{key}' value ({allowed[0]!r}); received {value!r}."
            )

    if settings.get("use_text_flow", False):
        raise ValueError(
            "rotation_mode='auto' does not support use_text_flow=True "
            f"({prefix}use_text_flow)."
        )

    line_dir = cast(T_dir, settings.get("line_dir", AUTO_LINE_DIR))
    char_dir = cast(T_dir, settings.get("char_dir", AUTO_CHAR_DIR))
    line_dir_rotated = cast(
        T_dir, settings.get("line_dir_rotated") or AUTO_LINE_DIR_ROTATED
    )
    char_dir_rotated = cast(
        T_dir, settings.get("char_dir_rotated") or AUTO_CHAR_DIR_ROTATED
    )
    validate_directions(line_dir, char_dir)
    validate_directions(line_dir_rotated, char_dir_rotated, "_rotated")


def to_local_box(bbox: T_bbox, rotation: int) -> T_bbox:
    """Map an axis-aligned page bbox into the local reading frame.

    In the local frame ``u`` increases in the direction the text is read and
    ``v`` increases towards the next text line.  For a visual
    counter-clockwise rotation ``r`` the page axes map as follows (page
    coordinates are ``x`` right, ``top`` down):

    =====  ====================  ====================
    ``r``  ``u``                 ``v``
    =====  ====================  ====================
    0      ``x``                 ``top``
    90     ``-top``              ``x``
    180    ``-x``                ``-top``
    270    ``top``               ``-x``
    =====  ====================  ====================

    These are exactly the four rotations of the basis ``(u, v) = ((1, 0),
    (0, 1))`` in the page frame, so the mapping is a rigid motion and the
    returned box is the tight axis-aligned box of the transformed corners.
    """

    r = rotation % 360
    x0, top, x1, bottom = bbox
    if r == 0:
        return (x0, top, x1, bottom)
    if r == 90:
        return (-bottom, x0, -top, x1)
    if r == 180:
        return (-x1, -bottom, -x0, -top)
    if r == 270:
        return (top, -x1, bottom, -x0)
    raise ValueError(f"Unsupported rotation: {rotation}")


def from_local_box(bbox: T_bbox, rotation: int) -> T_bbox:
    """Inverse of :func:`to_local_box`; returns a page-coordinate bbox."""

    r = rotation % 360
    u0, v0, u1, v1 = bbox
    if r == 0:
        return (u0, v0, u1, v1)
    if r == 90:
        return (v0, -u1, v1, -u0)
    if r == 180:
        return (-u1, -v1, -u0, -v0)
    if r == 270:
        return (-v1, u0, -v0, u1)
    raise ValueError(f"Unsupported rotation: {rotation}")


class _LocalChar(Dict[str, Any]):
    """Marker subclass for local proxy chars (never exposed publicly)."""


def _make_local_proxy(char: T_obj, rotation: int) -> T_obj:
    u0, v0, u1, v1 = to_local_box(
        (char["x0"], char["top"], char["x1"], char["bottom"]), rotation
    )
    proxy = _LocalChar(char)
    proxy["x0"] = u0
    proxy["x1"] = u1
    proxy["top"] = v0
    proxy["bottom"] = v1
    proxy["doctop"] = v0
    proxy["width"] = u1 - u0
    proxy["height"] = v1 - v0
    proxy["upright"] = 1
    return proxy


def _word_from_local(local_word: T_obj, originals: List[T_obj], rotation: int) -> T_obj:
    x0 = min(c["x0"] for c in originals)
    top = min(c["top"] for c in originals)
    x1 = max(c["x1"] for c in originals)
    bottom = max(c["bottom"] for c in originals)
    first = originals[0]
    doctop_adj = first["doctop"] - first["top"]

    word: T_obj = {
        "text": local_word["text"],
        "x0": x0,
        "x1": x1,
        "top": top,
        "doctop": top + doctop_adj,
        "bottom": bottom,
        "upright": first.get("upright", 1),
        "height": bottom - top,
        "width": x1 - x0,
        "direction": "ltr",
        "rotation": rotation,
    }

    for key, value in local_word.items():
        if key not in word:
            word[key] = value

    return word


def extract_words_auto(
    chars: T_obj_list,
    return_chars: bool = False,
    **settings: Any,
) -> T_obj_list:
    """Extract words from characters with automatic rotation handling.

    Characters are grouped by matrix rotation.  Each group is transformed into
    a local top-to-bottom / left-to-right frame and fed to the library's
    existing :class:`WordExtractor`, which guarantees that words never mix
    opposite orientations and that letters inside a word follow the local
    reading direction.  Returned words use the original page coordinates and,
    when ``return_chars=True``, carry the original character dictionaries.
    """

    settings = dict(settings)
    settings.pop("rotation_mode", None)
    validate_auto_settings(settings)

    chars = [c for c in chars if c.get("text", "") != ""]
    if not chars:
        return []

    by_rotation: Dict[int, List[T_obj]] = {r: [] for r in ROTATIONS}
    for char in chars:
        by_rotation[matrix_rotation(char["matrix"])].append(char)

    extractor_kwargs = {k: v for k, v in settings.items() if k in WORD_EXTRACTOR_KWARGS}
    extractor_kwargs.update(
        {
            "line_dir": AUTO_LINE_DIR,
            "char_dir": AUTO_CHAR_DIR,
            "line_dir_rotated": AUTO_LINE_DIR_ROTATED,
            "char_dir_rotated": AUTO_CHAR_DIR_ROTATED,
            "use_text_flow": False,
        }
    )
    extractor = WordExtractor(**extractor_kwargs)

    words: T_obj_list = []
    for rotation in ROTATIONS:
        group = by_rotation[rotation]
        if not group:
            continue

        proxy_to_original: Dict[int, T_obj] = {}
        proxies = []
        for char in group:
            proxy = _make_local_proxy(char, rotation)
            proxy_to_original[id(proxy)] = char
            proxies.append(proxy)

        for local_word, local_chars in extractor.iter_extract_tuples(proxies):
            originals = [proxy_to_original[id(c)] for c in local_chars]
            word = _word_from_local(local_word, originals, rotation)
            if return_chars:
                word["chars"] = originals
            words.append(word)

    return sorted_auto_words(words)


def auto_word_sort_key(word: T_obj) -> Tuple[int, T_num, T_num]:
    """Stable, rotation-major ordering used for auto-mode word lists."""

    rotation = word.get("rotation", 0)
    u0, v0, _u1, _v1 = to_local_box(
        (word["x0"], word["top"], word["x1"], word["bottom"]), rotation
    )
    return (rotation, v0, u0)


def sorted_auto_words(words: T_obj_list) -> T_obj_list:
    return sorted(words, key=auto_word_sort_key)


def _lines_in_local_frame(words: T_obj_list, tolerance: T_num) -> List[T_obj_list]:
    from .clustering import cluster_objects

    if not words:
        return []
    rotation = words[0].get("rotation", 0)
    local = [
        (to_local_box((w["x0"], w["top"], w["x1"], w["bottom"]), rotation), w)
        for w in words
    ]
    clusters = cluster_objects(local, lambda item: item[0][1], tolerance)
    lines = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda item: (item[0][0], item[0][1]))
        lines.append([w for _box, w in ordered])
    return lines


def extract_text_auto(
    chars: T_obj_list,
    line_dir_render: Optional[str] = None,
    char_dir_render: Optional[str] = None,
    **settings: Any,
) -> str:
    """Rotation-aware plain-text extraction.

    Words are produced by :func:`extract_words_auto`.  Each rotation group is
    laid out in its own local reading frame (top-to-bottom lines,
    left-to-right words) and groups are emitted in the stable order
    0, 90, 180, 270.  Table cells use their table's rotation instead.
    """

    if line_dir_render is not None or char_dir_render is not None:
        validate_directions(
            cast(T_dir, line_dir_render or AUTO_LINE_DIR),
            cast(T_dir, char_dir_render or AUTO_CHAR_DIR),
            "_render",
        )

    settings = dict(settings)
    settings.pop("rotation_mode", None)
    validate_auto_settings(settings)

    words = extract_words_auto(chars, **settings)
    if not words:
        return ""

    tolerance = settings.get("y_tolerance", 3)
    chunks: List[str] = []
    for rotation in ROTATIONS:
        group = [w for w in words if w["rotation"] == rotation]
        if not group:
            continue
        lines = _lines_in_local_frame(group, tolerance)
        text = "\n".join(" ".join(w["text"] for w in line) for line in lines)
        if line_dir_render in ("btt", "rtl"):
            text = "\n".join(reversed(text.split("\n")))
        if char_dir_render == "rtl":
            text = "\n".join("".join(reversed(line)) for line in text.split("\n"))
        chunks.append(text)

    return "\n".join(chunks)
