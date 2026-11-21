import itertools
from dataclasses import dataclass
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from . import utils
from ._typing import T_bbox, T_num, T_obj, T_obj_iter, T_obj_list, T_point

DEFAULT_SNAP_TOLERANCE = 3
DEFAULT_JOIN_TOLERANCE = 3
DEFAULT_MIN_WORDS_VERTICAL = 3
DEFAULT_MIN_WORDS_HORIZONTAL = 1

T_intersections = Dict[T_point, Dict[str, T_obj_list]]
T_table_settings = Union["TableSettings", Dict[str, Any]]

if TYPE_CHECKING:  # pragma: nocover
    from .page import Page


def snap_edges(
    edges: T_obj_list,
    x_tolerance: T_num = DEFAULT_SNAP_TOLERANCE,
    y_tolerance: T_num = DEFAULT_SNAP_TOLERANCE,
) -> T_obj_list:
    """
    Given a list of edges, snap any within `tolerance` pixels of one another
    to their positional average.
    """
    by_orientation: Dict[str, T_obj_list] = {"v": [], "h": []}
    for e in edges:
        by_orientation[e["orientation"]].append(e)

    snapped_v = utils.snap_objects(by_orientation["v"], "x0", x_tolerance)
    snapped_h = utils.snap_objects(by_orientation["h"], "top", y_tolerance)
    return snapped_v + snapped_h


def join_edge_group(
    edges: T_obj_iter, orientation: str, tolerance: T_num = DEFAULT_JOIN_TOLERANCE
) -> T_obj_list:
    """
    Given a list of edges along the same infinite line, join those that
    are within `tolerance` pixels of one another.
    """
    if orientation == "h":
        min_prop, max_prop = "x0", "x1"
    elif orientation == "v":
        min_prop, max_prop = "top", "bottom"
    else:
        raise ValueError("Orientation must be 'v' or 'h'")

    sorted_edges = list(sorted(edges, key=itemgetter(min_prop)))
    joined = [sorted_edges[0]]
    for e in sorted_edges[1:]:
        last = joined[-1]
        if e[min_prop] <= (last[max_prop] + tolerance):
            if e[max_prop] > last[max_prop]:
                # Extend current edge to new extremity
                joined[-1] = utils.resize_object(last, max_prop, e[max_prop])
        else:
            # Edge is separate from previous edges
            joined.append(e)

    return joined


def merge_edges(
    edges: T_obj_list,
    snap_x_tolerance: T_num,
    snap_y_tolerance: T_num,
    join_x_tolerance: T_num,
    join_y_tolerance: T_num,
) -> T_obj_list:
    """
    Using the `snap_edges` and `join_edge_group` methods above,
    merge a list of edges into a more "seamless" list.
    """

    def get_group(edge: T_obj) -> Tuple[str, T_num]:
        if edge["orientation"] == "h":
            return ("h", edge["top"])
        else:
            return ("v", edge["x0"])

    if snap_x_tolerance > 0 or snap_y_tolerance > 0:
        edges = snap_edges(edges, snap_x_tolerance, snap_y_tolerance)

    _sorted = sorted(edges, key=get_group)
    edge_groups = itertools.groupby(_sorted, key=get_group)
    edge_gen = (
        join_edge_group(
            items, k[0], (join_x_tolerance if k[0] == "h" else join_y_tolerance)
        )
        for k, items in edge_groups
    )
    edges = list(itertools.chain(*edge_gen))
    return edges


def words_to_edges_h(
    words: T_obj_list, word_threshold: int = DEFAULT_MIN_WORDS_HORIZONTAL
) -> T_obj_list:
    """
    Find (imaginary) horizontal lines that connect the tops
    of at least `word_threshold` words.
    """
    by_top = utils.cluster_objects(words, itemgetter("top"), 1)
    large_clusters = filter(lambda x: len(x) >= word_threshold, by_top)
    rects = list(map(utils.objects_to_rect, large_clusters))
    if len(rects) == 0:
        return []
    min_x0 = min(map(itemgetter("x0"), rects))
    max_x1 = max(map(itemgetter("x1"), rects))

    edges = []
    for r in rects:
        edges += [
            # Top of text
            {
                "x0": min_x0,
                "x1": max_x1,
                "top": r["top"],
                "bottom": r["top"],
                "width": max_x1 - min_x0,
                "orientation": "h",
            },
            # For each detected row, we also add the 'bottom' line.  This will
            # generate extra edges, (some will be redundant with the next row
            # 'top' line), but this catches the last row of every table.
            {
                "x0": min_x0,
                "x1": max_x1,
                "top": r["bottom"],
                "bottom": r["bottom"],
                "width": max_x1 - min_x0,
                "orientation": "h",
            },
        ]

    return edges


def words_to_edges_v(
    words: T_obj_list, word_threshold: int = DEFAULT_MIN_WORDS_VERTICAL
) -> T_obj_list:
    """
    Find (imaginary) vertical lines that connect the left, right, or
    center of at least `word_threshold` words.
    """
    # Find words that share the same left, right, or centerpoints
    by_x0 = utils.cluster_objects(words, itemgetter("x0"), 1)
    by_x1 = utils.cluster_objects(words, itemgetter("x1"), 1)

    def get_center(word: T_obj) -> T_num:
        return float(word["x0"] + word["x1"]) / 2

    by_center = utils.cluster_objects(words, get_center, 1)
    clusters = by_x0 + by_x1 + by_center

    # Find the points that align with the most words
    sorted_clusters = sorted(clusters, key=lambda x: -len(x))
    large_clusters = filter(lambda x: len(x) >= word_threshold, sorted_clusters)

    # For each of those points, find the bboxes fitting all matching words
    bboxes = list(map(utils.objects_to_bbox, large_clusters))

    # Iterate through those bboxes, condensing overlapping bboxes
    condensed_bboxes: List[T_bbox] = []
    for bbox in bboxes:
        overlap = any(utils.get_bbox_overlap(bbox, c) for c in condensed_bboxes)
        if not overlap:
            condensed_bboxes.append(bbox)

    if len(condensed_bboxes) == 0:
        return []

    condensed_rects = map(utils.bbox_to_rect, condensed_bboxes)
    sorted_rects = list(sorted(condensed_rects, key=itemgetter("x0")))

    max_x1 = max(map(itemgetter("x1"), sorted_rects))
    min_top = min(map(itemgetter("top"), sorted_rects))
    max_bottom = max(map(itemgetter("bottom"), sorted_rects))

    return [
        {
            "x0": b["x0"],
            "x1": b["x0"],
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
        for b in sorted_rects
    ] + [
        {
            "x0": max_x1,
            "x1": max_x1,
            "top": min_top,
            "bottom": max_bottom,
            "height": max_bottom - min_top,
            "orientation": "v",
        }
    ]


def edges_to_intersections(
    edges: T_obj_list, x_tolerance: T_num = 1, y_tolerance: T_num = 1
) -> T_intersections:
    """
    Given a list of edges, return the points at which they intersect
    within `tolerance` pixels.
    """
    intersections: T_intersections = {}
    v_edges, h_edges = [
        list(filter(lambda x: x["orientation"] == o, edges)) for o in ("v", "h")
    ]
    for v in sorted(v_edges, key=itemgetter("x0", "top")):
        for h in sorted(h_edges, key=itemgetter("top", "x0")):
            if (
                (v["top"] <= (h["top"] + y_tolerance))
                and (v["bottom"] >= (h["top"] - y_tolerance))
                and (v["x0"] >= (h["x0"] - x_tolerance))
                and (v["x0"] <= (h["x1"] + x_tolerance))
            ):
                vertex = (v["x0"], h["top"])
                if vertex not in intersections:
                    intersections[vertex] = {"v": [], "h": []}
                intersections[vertex]["v"].append(v)
                intersections[vertex]["h"].append(h)
    return intersections


def intersections_to_cells(intersections: T_intersections) -> List[T_bbox]:
    """
    Given a list of points (`intersections`), return all rectangular "cells"
    that those points describe.

    `intersections` should be a dictionary with (x0, top) tuples as keys,
    and a list of edge objects as values. The edge objects should correspond
    to the edges that touch the intersection.
    """

    def edge_connects(p1: T_point, p2: T_point) -> bool:
        def edges_to_set(edges: T_obj_list) -> Set[T_bbox]:
            return set(map(utils.obj_to_bbox, edges))

        if p1[0] == p2[0]:
            common = edges_to_set(intersections[p1]["v"]).intersection(
                edges_to_set(intersections[p2]["v"])
            )
            if len(common):
                return True

        if p1[1] == p2[1]:
            common = edges_to_set(intersections[p1]["h"]).intersection(
                edges_to_set(intersections[p2]["h"])
            )
            if len(common):
                return True
        return False

    points = list(sorted(intersections.keys()))
    n_points = len(points)

    def find_smallest_cell(points: List[T_point], i: int) -> Optional[T_bbox]:
        if i == n_points - 1:
            return None
        pt = points[i]
        rest = points[i + 1 :]
        # Get all the points directly below and directly right
        below = [x for x in rest if x[0] == pt[0]]
        right = [x for x in rest if x[1] == pt[1]]
        for below_pt in below:
            if not edge_connects(pt, below_pt):
                continue

            for right_pt in right:
                if not edge_connects(pt, right_pt):
                    continue

                bottom_right = (right_pt[0], below_pt[1])

                if (
                    (bottom_right in intersections)
                    and edge_connects(bottom_right, right_pt)
                    and edge_connects(bottom_right, below_pt)
                ):

                    return (pt[0], pt[1], bottom_right[0], bottom_right[1])
        return None

    cell_gen = (find_smallest_cell(points, i) for i in range(len(points)))
    return list(filter(None, cell_gen))


def cells_to_tables(cells: List[T_bbox]) -> List[List[T_bbox]]:
    """
    Given a list of bounding boxes (`cells`), return a list of tables that
    hold those cells most simply (and contiguously).
    """

    def bbox_to_corners(bbox: T_bbox) -> Tuple[T_point, T_point, T_point, T_point]:
        x0, top, x1, bottom = bbox
        return ((x0, top), (x0, bottom), (x1, top), (x1, bottom))

    remaining_cells = list(cells)

    # Iterate through the cells found above, and assign them
    # to contiguous tables

    current_corners: Set[T_point] = set()
    current_cells: List[T_bbox] = []

    tables = []
    while len(remaining_cells):
        initial_cell_count = len(current_cells)
        for cell in list(remaining_cells):
            cell_corners = bbox_to_corners(cell)
            # If we're just starting a table ...
            if len(current_cells) == 0:
                # ... immediately assign it to the empty group
                current_corners |= set(cell_corners)
                current_cells.append(cell)
                remaining_cells.remove(cell)
            else:
                # How many corners does this table share with the current group?
                corner_count = sum(c in current_corners for c in cell_corners)

                # If touching on at least one corner...
                if corner_count > 0:
                    # ... assign it to the current group
                    current_corners |= set(cell_corners)
                    current_cells.append(cell)
                    remaining_cells.remove(cell)

        # If this iteration did not find any more cells to append...
        if len(current_cells) == initial_cell_count:
            # ... start a new cell group
            tables.append(list(current_cells))
            current_corners.clear()
            current_cells.clear()

    # Once we have exhausting the list of cells ...

    # ... and we have a cell group that has not been stored
    if len(current_cells):
        # ... store it.
        tables.append(list(current_cells))

    # Sort the tables top-to-bottom-left-to-right based on the value of the
    # topmost-and-then-leftmost coordinate of a table.
    _sorted = sorted(tables, key=lambda t: min((c[1], c[0]) for c in t))
    filtered = [t for t in _sorted if len(t) > 1]
    return filtered


class CellGroup(object):
    def __init__(self, cells: List[Optional[T_bbox]]):
        self.cells = cells
        self.bbox = (
            min(map(itemgetter(0), filter(None, cells))),
            min(map(itemgetter(1), filter(None, cells))),
            max(map(itemgetter(2), filter(None, cells))),
            max(map(itemgetter(3), filter(None, cells))),
        )


class Row(CellGroup):
    pass


class Column(CellGroup):
    pass


class Table(object):
    def __init__(
        self,
        page: "Page",
        cells: List[T_bbox],
        rotation: Optional[int] = None,
    ):
        self.page = page
        self.cells = cells
        self._auto = rotation is not None
        if rotation is not None:
            self.rotation = rotation % 360

    @property
    def bbox(self) -> T_bbox:
        c = self.cells
        return (
            min(map(itemgetter(0), c)),
            min(map(itemgetter(1), c)),
            max(map(itemgetter(2), c)),
            max(map(itemgetter(3), c)),
        )

    def _get_rows_or_cols(self, kind: Type[CellGroup]) -> List[CellGroup]:
        if self._auto:
            return self._get_auto_rows_or_cols(kind)

        axis = 0 if kind is Row else 1
        antiaxis = int(not axis)

        # Sort first by top/x0, then by x0/top
        _sorted = sorted(self.cells, key=itemgetter(antiaxis, axis))

        # Sort get all x0s/tops
        xs = list(sorted(set(map(itemgetter(axis), self.cells))))

        # Group by top/x0
        grouped = itertools.groupby(_sorted, itemgetter(antiaxis))

        rows = []
        # for y/x, row/column-cells ...
        for y, row_cells in grouped:
            xdict = {cell[axis]: cell for cell in row_cells}
            row = kind([xdict.get(x) for x in xs])
            rows.append(row)
        return rows

    def _get_auto_rows_or_cols(self, kind: Type[CellGroup]) -> List[CellGroup]:
        """Group cells in the table's own logical order.

        The first row is the one read first in the table's rotation, so a
        180-degree table is not reversed just because its page coordinates
        decrease.  Cells themselves stay in original Page coordinates; only
        the sequence is logical.  Grouping is a single sort plus a single
        groupby, i.e. O(cells log cells).
        """
        from .utils.rotation import to_local_box

        rotation = self.rotation
        axis = 1 if kind is Row else 0
        cross = 0 if kind is Row else 1

        local = [(to_local_box(cell, rotation), cell) for cell in self.cells]
        local.sort(key=lambda item: (item[0][axis], item[0][cross]))

        groups: List[CellGroup] = []
        for _key, items in itertools.groupby(
            local, key=lambda item: round(item[0][axis], 9)
        ):
            members = [cell for _box, cell in items]
            groups.append(kind(cast(List[Optional[T_bbox]], members)))
        return groups

    @property
    def rows(self) -> List[CellGroup]:
        return self._get_rows_or_cols(Row)

    @property
    def columns(self) -> List[CellGroup]:
        return self._get_rows_or_cols(Column)

    def _auto_local_grid(self) -> Tuple[List[T_num], List[T_num]]:
        """Return sorted local u/v boundaries for this auto table's grid."""

        from .utils.rotation import to_local_box

        rotation = self.rotation
        us: Set[T_num] = set()
        vs: Set[T_num] = set()
        for cell in self.cells:
            u0, v0, u1, v1 = to_local_box(cell, rotation)
            us.add(round(u0, 9))
            us.add(round(u1, 9))
            vs.add(round(v0, 9))
            vs.add(round(v1, 9))
        return sorted(us), sorted(vs)

    @staticmethod
    def _index_of(bounds: List[T_num], value: T_num) -> int:
        if len(bounds) < 2:
            return -1
        low, high = 0, len(bounds) - 1
        while low + 1 < high:
            mid = (low + high) // 2
            if value < bounds[mid]:
                high = mid
            else:
                low = mid
        if bounds[low] <= value < bounds[low + 1]:
            return low
        return -1

    def _auto_assign_chars(self) -> List[List[T_obj]]:
        """
        Assign characters to cells in O(chars * log(cells)) time.

        Row and column boundaries are binary-searched, so no character/cell
        pair enumeration is performed.  Each character is assigned to at
        most one cell because the local intervals are half-open.  Only
        characters whose rotation matches the table's rotation participate.
        """

        from .utils.rotation import matrix_rotation, to_local_box

        rotation = self.rotation
        us, vs = self._auto_local_grid()
        n_columns = max(0, len(us) - 1)
        n_rows = max(0, len(vs) - 1)
        selected: List[List[T_obj]] = [[] for _ in range(n_columns * n_rows)]
        if n_columns == 0 or n_rows == 0:
            return selected

        for char in self.page.chars:
            try:
                if matrix_rotation(char["matrix"]) != rotation:
                    continue
            except ValueError:
                continue
            u0, v0, u1, v1 = to_local_box(
                (char["x0"], char["top"], char["x1"], char["bottom"]), rotation
            )
            column = self._index_of(us, (u0 + u1) / 2.0)
            row = self._index_of(vs, (v0 + v1) / 2.0)
            if column >= 0 and row >= 0:
                selected[row * n_columns + column].append(char)
        return selected

    def claimed_char_ids(self) -> Set[int]:
        """Identity set of the characters this table would extract."""

        if self._auto:
            cached = getattr(self, "_claimed_ids", None)
            if cached is None:
                cached = {
                    id(char)
                    for cell_chars in self._auto_assign_chars()
                    for char in cell_chars
                }
                self._claimed_ids = cached
            return cached

        chars = self.page.chars
        ids = set()
        for cell in self.cells:
            x0, top, x1, bottom = cell
            for char in chars:
                h_mid = (char["x0"] + char["x1"]) / 2
                v_mid = (char["top"] + char["bottom"]) / 2
                if x0 <= h_mid < x1 and top <= v_mid < bottom:
                    ids.add(id(char))
        return ids

    def _extract_auto(self, **kwargs: Any) -> List[List[Optional[str]]]:
        from .utils.rotation import _make_local_proxy, to_local_box

        rotation = self.rotation
        kwargs = dict(kwargs)
        kwargs.pop("rotation_mode", None)
        us, vs = self._auto_local_grid()
        n_columns = max(0, len(us) - 1)
        selected = self._auto_assign_chars()

        table_arr: List[List[Optional[str]]] = []
        for row in self._get_auto_rows_or_cols(Row):
            arr: List[Optional[str]] = []
            for cell in row.cells:
                assert cell is not None
                cell_bbox = to_local_box(cell, rotation)
                column = self._index_of(us, (cell_bbox[0] + cell_bbox[2]) / 2.0)
                row_index = self._index_of(vs, (cell_bbox[1] + cell_bbox[3]) / 2.0)
                cell_chars = (
                    selected[row_index * n_columns + column]
                    if column >= 0 and row_index >= 0
                    else []
                )
                if len(cell_chars):
                    proxies = [_make_local_proxy(char, rotation) for char in cell_chars]
                    if "layout" in kwargs:
                        kwargs["layout_width"] = cell_bbox[2] - cell_bbox[0]
                        kwargs["layout_height"] = cell_bbox[3] - cell_bbox[1]
                        kwargs["layout_bbox"] = cell_bbox
                    cell_text = utils.extract_text(proxies, **kwargs)
                else:
                    cell_text = ""
                arr.append(cell_text)
            table_arr.append(arr)
        return table_arr

    def extract(self, **kwargs: Any) -> List[List[Optional[str]]]:
        if self._auto:
            return self._extract_auto(**kwargs)

        chars = self.page.chars
        table_arr = []

        def char_in_bbox(char: T_obj, bbox: T_bbox) -> bool:
            v_mid = (char["top"] + char["bottom"]) / 2
            h_mid = (char["x0"] + char["x1"]) / 2
            x0, top, x1, bottom = bbox
            return bool(
                (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
            )

        for row in self.rows:
            arr = []
            row_chars = [char for char in chars if char_in_bbox(char, row.bbox)]

            for cell in row.cells:
                if cell is None:
                    cell_text = None
                else:
                    cell_chars = [
                        char for char in row_chars if char_in_bbox(char, cell)
                    ]

                    if len(cell_chars):
                        if "layout" in kwargs:
                            kwargs["layout_width"] = cell[2] - cell[0]
                            kwargs["layout_height"] = cell[3] - cell[1]
                            kwargs["layout_bbox"] = cell
                        cell_text = utils.extract_text(cell_chars, **kwargs)
                    else:
                        cell_text = ""
                arr.append(cell_text)
            table_arr.append(arr)

        return table_arr


TABLE_STRATEGIES = ["lines", "lines_strict", "text", "explicit"]
NON_NEGATIVE_SETTINGS = [
    "snap_tolerance",
    "snap_x_tolerance",
    "snap_y_tolerance",
    "join_tolerance",
    "join_x_tolerance",
    "join_y_tolerance",
    "edge_min_length",
    "min_words_vertical",
    "min_words_horizontal",
    "intersection_tolerance",
    "intersection_x_tolerance",
    "intersection_y_tolerance",
]


class UnsetFloat(float):
    pass


UNSET = UnsetFloat(0)


@dataclass
class TableSettings:
    vertical_strategy: str = "lines"
    horizontal_strategy: str = "lines"
    explicit_vertical_lines: Optional[List[Union[T_obj, T_num]]] = None
    explicit_horizontal_lines: Optional[List[Union[T_obj, T_num]]] = None
    snap_tolerance: T_num = DEFAULT_SNAP_TOLERANCE
    snap_x_tolerance: T_num = UNSET
    snap_y_tolerance: T_num = UNSET
    join_tolerance: T_num = DEFAULT_JOIN_TOLERANCE
    join_x_tolerance: T_num = UNSET
    join_y_tolerance: T_num = UNSET
    edge_min_length: T_num = 3
    min_words_vertical: int = DEFAULT_MIN_WORDS_VERTICAL
    min_words_horizontal: int = DEFAULT_MIN_WORDS_HORIZONTAL
    intersection_tolerance: T_num = 3
    intersection_x_tolerance: T_num = UNSET
    intersection_y_tolerance: T_num = UNSET
    text_settings: Optional[Dict[str, Any]] = None
    rotation_mode: str = "legacy"

    def __post_init__(self) -> None:
        """Clean up user-provided table settings.

        Validates that the table settings provided consists of acceptable values and
        returns a cleaned up version. The cleaned up version fills out the missing
        values with the default values in the provided settings.

        TODO: Can be further used to validate that the values are of the correct
            type. For example, raising a value error when a non-boolean input is
            provided for the key ``keep_blank_chars``.

        :param table_settings: User-provided table settings.
        :returns: A cleaned up version of the user-provided table settings.
        :raises ValueError: When an unrecognised key is provided.
        """

        for setting in NON_NEGATIVE_SETTINGS:
            if (getattr(self, setting) or 0) < 0:
                raise ValueError(f"Table setting '{setting}' cannot be negative")

        if self.rotation_mode not in ("legacy", "auto"):
            raise ValueError(
                "rotation_mode must be 'legacy' or 'auto', "
                f"not {self.rotation_mode!r}"
            )

        if self.text_settings is not None and "rotation_mode" in self.text_settings:
            text_mode = self.text_settings["rotation_mode"]
            if text_mode not in ("legacy", "auto"):
                raise ValueError(
                    "text_rotation_mode must be 'legacy' or 'auto', "
                    f"not {text_mode!r}"
                )
            if self.rotation_mode == "legacy" and text_mode == "auto":
                # The text_ prefix may initiate auto mode; the effective
                # mode becomes auto so strategy validation applies.
                self.rotation_mode = "auto"

        for orientation in ["horizontal", "vertical"]:
            strategy = getattr(self, orientation + "_strategy")
            if strategy not in TABLE_STRATEGIES:
                raise ValueError(
                    f"{orientation}_strategy must be one of"
                    f'{{{",".join(TABLE_STRATEGIES)}}}'
                )

        if self.text_settings is None:
            self.text_settings = {}

        # This next section is for backwards compatibility
        for attr in ["x_tolerance", "y_tolerance"]:
            if attr not in self.text_settings:
                self.text_settings[attr] = self.text_settings.get("tolerance", 3)

        if "tolerance" in self.text_settings:
            del self.text_settings["tolerance"]
        # End of that section

        for attr, fallback in [
            ("snap_x_tolerance", "snap_tolerance"),
            ("snap_y_tolerance", "snap_tolerance"),
            ("join_x_tolerance", "join_tolerance"),
            ("join_y_tolerance", "join_tolerance"),
            ("intersection_x_tolerance", "intersection_tolerance"),
            ("intersection_y_tolerance", "intersection_tolerance"),
        ]:
            if getattr(self, attr) is UNSET:
                setattr(self, attr, getattr(self, fallback))

    @classmethod
    def resolve(cls, settings: Optional[T_table_settings]) -> "TableSettings":
        if settings is None:
            return cls()
        elif isinstance(settings, cls):
            return settings
        elif isinstance(settings, dict):
            core_settings = {}
            text_settings = {}
            for k, v in settings.items():
                if k[:5] == "text_":
                    text_settings[k[5:]] = v
                else:
                    core_settings[k] = v
            core_settings["text_settings"] = text_settings
            return cls(**core_settings)
        else:
            raise ValueError(f"Cannot resolve settings: {settings}")


def _hint_in_region(value: T_num, region: T_bbox, orientation: str) -> bool:
    """Whether a numeric explicit hint lies in the current region.

    Numeric hints are page coordinates. On a cropped view, a hint outside the
    region is dropped; it is never snapped onto the nearest boundary.
    """
    if orientation == "v":
        return region[0] <= value <= region[2]
    return region[1] <= value <= region[3]


class TableFinder(object):
    """
    Given a PDF page, find plausible table structures.

    Largely borrowed from Anssi Nurminen's master's thesis:
    http://dspace.cc.tut.fi/dpub/bitstream/handle/123456789/21520/Nurminen.pdf?sequence=3

    ... and inspired by Tabula:
    https://github.com/tabulapdf/tabula-extractor/issues/16
    """

    def __init__(self, page: "Page", settings: Optional[T_table_settings] = None):
        self.page = page
        self.settings = TableSettings.resolve(settings)
        self.edges = self.get_edges()
        self.intersections = edges_to_intersections(
            self.edges,
            self.settings.intersection_x_tolerance,
            self.settings.intersection_y_tolerance,
        )
        self.cells = intersections_to_cells(self.intersections)
        self.tables = [
            Table(self.page, cell_group) for cell_group in cells_to_tables(self.cells)
        ]

    def _edge_region(self) -> Optional[T_bbox]:
        """Region to which this finder clips geometry, or ``None`` for none.

        `CroppedPage` reports its crop box; an ordinary page reports ``None``,
        preserving historical behaviour (some pages legitimately draw slightly
        outside their box). Object-form explicit hints use this exact same
        region, so they cannot smuggle geometry from outside the view.
        """
        return getattr(self.page, "_edge_bbox", None)

    def _hint_region(self) -> Optional[T_bbox]:
        """Region outside which numeric hints are dropped, or ``None``.

        On a real crop, numeric hints are confined to the crop. On an
        ordinary (or merely filtered/bbox-selected) view, they keep their
        historical full-page extent and are not filtered.
        """
        return getattr(self.page, "_table_region", None)

    def get_edges(self) -> T_obj_list:
        settings = self.settings

        for orientation in ["vertical", "horizontal"]:
            strategy = getattr(settings, orientation + "_strategy")
            if strategy == "explicit":
                lines = getattr(settings, "explicit_" + orientation + "_lines")
                if len(lines) < 2:
                    raise ValueError(
                        f"If {orientation}_strategy == 'explicit', "
                        f"explicit_{orientation}_lines "
                        f"must be specified as a list/tuple of two or more "
                        f"floats/ints."
                    )

        v_strat = settings.vertical_strategy
        h_strat = settings.horizontal_strategy

        region = self._edge_region()

        if v_strat == "text" or h_strat == "text":
            words = self.page.extract_words(**(settings.text_settings or {}))

        v_explicit = []
        # Numeric hints keep their historical extent: the full height/width
        # of this view's bounding box. On a real crop, a hint outside the
        # crop region is dropped rather than pulled onto a boundary by snap
        # tolerance. Uncapped views (`_table_region is None`) are unchanged.
        hint_bbox = self.page.bbox
        hint_region = self._hint_region()
        for desc in settings.explicit_vertical_lines or []:
            if isinstance(desc, dict):
                for e in utils.obj_to_edges(desc, region):
                    if e["orientation"] == "v":
                        v_explicit.append(e)
            else:
                if hint_region is not None and not _hint_in_region(
                    desc, hint_region, "v"
                ):
                    continue
                v_explicit.append(
                    {
                        "x0": desc,
                        "x1": desc,
                        "top": hint_bbox[1],
                        "bottom": hint_bbox[3],
                        "height": hint_bbox[3] - hint_bbox[1],
                        "orientation": "v",
                    }
                )

        if v_strat == "lines":
            v_base = utils.filter_edges(self.page.edges, "v")
        elif v_strat == "lines_strict":
            v_base = utils.filter_edges(self.page.edges, "v", edge_type="line")
        elif v_strat == "text":
            v_base = words_to_edges_v(words, word_threshold=settings.min_words_vertical)
        elif v_strat == "explicit":
            v_base = []

        v = v_base + v_explicit

        h_explicit = []
        for desc in settings.explicit_horizontal_lines or []:
            if isinstance(desc, dict):
                for e in utils.obj_to_edges(desc, region):
                    if e["orientation"] == "h":
                        h_explicit.append(e)
            else:
                if hint_region is not None and not _hint_in_region(
                    desc, hint_region, "h"
                ):
                    continue
                h_explicit.append(
                    {
                        "x0": hint_bbox[0],
                        "x1": hint_bbox[2],
                        "width": hint_bbox[2] - hint_bbox[0],
                        "top": desc,
                        "bottom": desc,
                        "orientation": "h",
                    }
                )

        if h_strat == "lines":
            h_base = utils.filter_edges(self.page.edges, "h")
        elif h_strat == "lines_strict":
            h_base = utils.filter_edges(self.page.edges, "h", edge_type="line")
        elif h_strat == "text":
            h_base = words_to_edges_h(
                words, word_threshold=settings.min_words_horizontal
            )
        elif h_strat == "explicit":
            h_base = []

        h = h_base + h_explicit

        edges = list(v) + list(h)

        edges = merge_edges(
            edges,
            snap_x_tolerance=settings.snap_x_tolerance,
            snap_y_tolerance=settings.snap_y_tolerance,
            join_x_tolerance=settings.join_x_tolerance,
            join_y_tolerance=settings.join_y_tolerance,
        )

        return utils.filter_edges(edges, min_length=settings.edge_min_length)
