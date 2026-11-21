import itertools
from operator import itemgetter
from typing import Dict, Iterable, List, Optional, Tuple

from .._typing import T_bbox, T_num, T_obj, T_obj_list, T_point
from .clustering import cluster_objects


def objects_to_rect(objects: Iterable[T_obj]) -> Dict[str, T_num]:
    """
    Given an iterable of objects, return the smallest rectangle (i.e. a
    dict with "x0", "top", "x1", and "bottom" keys) that contains them
    all.
    """
    return bbox_to_rect(objects_to_bbox(objects))


def objects_to_bbox(objects: Iterable[T_obj]) -> T_bbox:
    """
    Given an iterable of objects, return the smallest bounding box that
    contains them all.
    """
    return merge_bboxes(map(bbox_getter, objects))


bbox_getter = itemgetter("x0", "top", "x1", "bottom")


def obj_to_bbox(obj: T_obj) -> T_bbox:
    """
    Return the bounding box for an object.
    """
    bbox: T_bbox = bbox_getter(obj)
    return bbox


def bbox_to_rect(bbox: T_bbox) -> Dict[str, T_num]:
    """
    Return the rectangle (i.e a dict with keys "x0", "top", "x1",
    "bottom") for an object.
    """
    return {"x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3]}


def merge_bboxes(bboxes: Iterable[T_bbox]) -> T_bbox:
    """
    Given an iterable of bounding boxes, return the smallest bounding box
    that contains them all.
    """
    x0, top, x1, bottom = zip(*bboxes)
    return (min(x0), min(top), max(x1), max(bottom))


def get_bbox_overlap(a: T_bbox, b: T_bbox) -> Optional[T_bbox]:
    a_left, a_top, a_right, a_bottom = a
    b_left, b_top, b_right, b_bottom = b
    o_left = max(a_left, b_left)
    o_right = min(a_right, b_right)
    o_bottom = min(a_bottom, b_bottom)
    o_top = max(a_top, b_top)
    o_width = o_right - o_left
    o_height = o_bottom - o_top
    if o_height >= 0 and o_width >= 0 and o_height + o_width > 0:
        return (o_left, o_top, o_right, o_bottom)
    else:
        return None


def calculate_area(bbox: T_bbox) -> T_num:
    left, top, right, bottom = bbox
    if left > right or top > bottom:
        raise ValueError(f"{bbox} has a negative width or height.")
    return (right - left) * (bottom - top)


def clip_obj(obj: T_obj, bbox: T_bbox) -> Optional[T_obj]:
    overlap = get_bbox_overlap(obj_to_bbox(obj), bbox)
    if overlap is None:
        return None

    dims = bbox_to_rect(overlap)
    copy = dict(obj)

    for attr in ["x0", "top", "x1", "bottom"]:
        copy[attr] = dims[attr]

    diff = dims["top"] - obj["top"]
    if "doctop" in copy:
        copy["doctop"] = obj["doctop"] + diff
    copy["width"] = copy["x1"] - copy["x0"]
    copy["height"] = copy["bottom"] - copy["top"]

    return copy


def intersects_bbox(objs: Iterable[T_obj], bbox: T_bbox) -> T_obj_list:
    """
    Filters objs to only those intersecting the bbox
    """
    return [obj for obj in objs if get_bbox_overlap(obj_to_bbox(obj), bbox) is not None]


def within_bbox(objs: Iterable[T_obj], bbox: T_bbox) -> T_obj_list:
    """
    Filters objs to only those fully within the bbox
    """
    return [
        obj
        for obj in objs
        if get_bbox_overlap(obj_to_bbox(obj), bbox) == obj_to_bbox(obj)
    ]


def outside_bbox(objs: Iterable[T_obj], bbox: T_bbox) -> T_obj_list:
    """
    Filters objs to only those fully outside the bbox
    """
    return [obj for obj in objs if get_bbox_overlap(obj_to_bbox(obj), bbox) is None]


def crop_to_bbox(objs: Iterable[T_obj], bbox: T_bbox) -> T_obj_list:
    """
    Filters objs to only those intersecting the bbox,
    and crops the extent of the objects to the bbox.
    """
    return list(filter(None, (clip_obj(obj, bbox) for obj in objs)))


def move_object(obj: T_obj, axis: str, value: T_num) -> T_obj:
    assert axis in ("h", "v")
    if axis == "h":
        new_items = [
            ("x0", obj["x0"] + value),
            ("x1", obj["x1"] + value),
        ]
    if axis == "v":
        new_items = [
            ("top", obj["top"] + value),
            ("bottom", obj["bottom"] + value),
        ]
        if "doctop" in obj:
            new_items += [("doctop", obj["doctop"] + value)]
        if "y0" in obj:
            new_items += [
                ("y0", obj["y0"] - value),
                ("y1", obj["y1"] - value),
            ]
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


def snap_objects(objs: Iterable[T_obj], attr: str, tolerance: T_num) -> T_obj_list:
    axis = {"x0": "h", "x1": "h", "top": "v", "bottom": "v"}[attr]
    list_objs = list(objs)
    clusters = cluster_objects(list_objs, itemgetter(attr), tolerance)
    avgs = [sum(map(itemgetter(attr), cluster)) / len(cluster) for cluster in clusters]
    snapped_clusters = [
        [move_object(obj, axis, avg - obj[attr]) for obj in cluster]
        for cluster, avg in zip(clusters, avgs)
    ]
    return list(itertools.chain(*snapped_clusters))


def resize_object(obj: T_obj, key: str, value: T_num) -> T_obj:
    assert key in ("x0", "x1", "top", "bottom")
    old_value = obj[key]
    diff = value - old_value
    new_items = [
        (key, value),
    ]
    if key == "x0":
        assert value <= obj["x1"]
        new_items.append(("width", obj["x1"] - value))
    elif key == "x1":
        assert value >= obj["x0"]
        new_items.append(("width", value - obj["x0"]))
    elif key == "top":
        assert value <= obj["bottom"]
        new_items.append(("doctop", obj["doctop"] + diff))
        new_items.append(("height", obj["height"] - diff))
        if "y1" in obj:
            new_items.append(("y1", obj["y1"] - diff))
    elif key == "bottom":
        assert value >= obj["top"]
        new_items.append(("height", obj["height"] + diff))
        if "y0" in obj:
            new_items.append(("y0", obj["y0"] - diff))
    return obj.__class__(tuple(obj.items()) + tuple(new_items))


# Tolerance used only to absorb floating-point noise in segment clipping.
# It is deliberately far smaller than any table tolerance, and is not used to
# admit numeric hints that lie outside a region.
CLIP_EPS = 1e-9


def _source_pts(obj: T_obj) -> Optional[List[T_point]]:
    """Return an object's original points, if it exposes them.

    Cropping (`crop_to_bbox`) rewrites an object's bounding-box fields but
    leaves ``pts`` (and ``path``) untouched, so ``pts`` still describes the
    stroke as drawn on the original page.  Using it here means a crop never
    invents geometry and no per-view copy of the page's paths is needed.
    """
    pts = obj.get("pts")
    if not isinstance(pts, (list, tuple)) or len(pts) < 2:
        return None
    try:
        return [(float(pt[0]), float(pt[1])) for pt in pts]
    except (TypeError, ValueError, IndexError):  # pragma: nocover
        return None


def _source_bbox(obj: T_obj) -> T_bbox:
    """Bounding box of the original stroke (not of a clipped object)."""
    pts = _source_pts(obj)
    if pts is not None:
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return obj_to_bbox(obj)


def _pdf_height(obj: T_obj, source_top: T_num) -> Optional[T_num]:
    """Recover the constant ``y1 + top`` used by pdfplumber's y-flip.

    Holding it constant lets a clipped edge carry consistent ``y0``/``y1``
    values without consulting the page.
    """
    y1 = obj.get("y1")
    if y1 is None:
        return None
    return float(y1) + source_top


def clip_segment(
    p0: T_point, p1: T_point, bbox: T_bbox
) -> Optional[Tuple[T_point, T_point]]:
    """Clip a line segment to an axis-aligned, inclusive bounding box.

    Returns the pair of clipped endpoints, or ``None`` when no part of the
    segment lies within the box.  A non-zero segment that only touches the
    box at a single point counts as outside: it contributes no edge.
    """
    x0, top, x1, bottom = bbox
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay

    if abs(dx) <= CLIP_EPS and abs(dy) <= CLIP_EPS:
        inside = (x0 - CLIP_EPS) <= ax <= (x1 + CLIP_EPS) and (
            top - CLIP_EPS
        ) <= ay <= (bottom + CLIP_EPS)
        return (p0, p1) if inside else None

    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, ax - x0),
        (dx, x1 - ax),
        (-dy, ay - top),
        (dy, bottom - ay),
    ):
        if abs(p) <= CLIP_EPS:
            if q < -CLIP_EPS:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r

    if t1 < t0 - CLIP_EPS:
        return None

    def snap(value: T_num, lo: T_num, hi: T_num) -> T_num:
        if lo - CLIP_EPS <= value <= lo + CLIP_EPS:
            return lo
        if hi - CLIP_EPS <= value <= hi + CLIP_EPS:
            return hi
        return value

    q0 = (snap(ax + dx * t0, x0, x1), snap(ay + dy * t0, top, bottom))
    q1 = (snap(ax + dx * t1, x0, x1), snap(ay + dy * t1, top, bottom))

    if abs(q0[0] - q1[0]) <= CLIP_EPS and abs(q0[1] - q1[1]) <= CLIP_EPS:
        # Tangential contact only.
        return None

    return (q0, q1)


def _edge_from_segment(
    source: T_obj,
    object_type: str,
    p0: T_point,
    p1: T_point,
    orientation: Optional[str],
    source_top: T_num,
) -> T_obj:
    x0, x1 = sorted((p0[0], p1[0]))
    top, bottom = sorted((p0[1], p1[1]))
    delta = source.get("doctop", 0) - source.get("top", 0)
    height = _pdf_height(source, source_top)
    edge = {k: v for k, v in source.items() if k not in ("pts", "path")}
    edge.update(
        {
            "object_type": object_type,
            "x0": x0,
            "x1": x1,
            "top": top,
            "bottom": bottom,
            "doctop": top + delta,
            "width": x1 - x0,
            "height": bottom - top,
            "orientation": orientation,
            "pts": [p0, p1],
        }
    )
    if height is not None:
        edge["y1"] = height - top
        edge["y0"] = height - bottom
    return edge


def clip_edge_to_bbox(edge: T_obj, bbox: T_bbox) -> Optional[T_obj]:
    """Clip an existing edge object to ``bbox`` without mutating it."""
    source_top = edge.get("top", 0)
    seg = clip_segment((edge["x0"], edge["top"]), (edge["x1"], edge["bottom"]), bbox)
    if seg is None:
        return None
    if seg == ((edge["x0"], edge["top"]), (edge["x1"], edge["bottom"])):
        return dict(edge)
    return _edge_from_segment(
        edge,
        edge.get("object_type", "edge"),
        seg[0],
        seg[1],
        edge.get("orientation"),
        source_top,
    )


def curve_to_edges(curve: T_obj, bbox: Optional[T_bbox] = None) -> T_obj_list:
    point_pairs = list(zip(curve["pts"], curve["pts"][1:]))

    if bbox is None:
        return [
            {
                "object_type": "curve_edge",
                "x0": min(p0[0], p1[0]),
                "x1": max(p0[0], p1[0]),
                "top": min(p0[1], p1[1]),
                "doctop": min(p0[1], p1[1]) + (curve["doctop"] - curve["top"]),
                "bottom": max(p0[1], p1[1]),
                "width": abs(p0[0] - p1[0]),
                "height": abs(p0[1] - p1[1]),
                "orientation": (
                    "v" if p0[0] == p1[0] else ("h" if p0[1] == p1[1] else None)
                ),
            }
            for p0, p1 in point_pairs
        ]

    # Clip each drawn segment on its own: segments that leave the region and
    # return are cut, never bridged across the gap; no closing side is added.
    src = _source_bbox(curve)
    edges = []
    for p0, p1 in point_pairs:
        orientation = "v" if p0[0] == p1[0] else ("h" if p0[1] == p1[1] else None)
        seg = clip_segment(p0, p1, bbox)
        if seg is None:
            continue
        edges.append(
            _edge_from_segment(curve, "curve_edge", seg[0], seg[1], orientation, src[1])
        )
    return edges


def _rect_corners(rect: T_obj) -> T_bbox:
    """The rectangle's original outline, independent of any crop."""
    pts = _source_pts(rect)
    if pts is not None:
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return (
        rect["x0"],
        rect["top"],
        rect["x1"],
        rect["bottom"],
    )


def rect_to_edges(rect: T_obj, bbox: Optional[T_bbox] = None) -> T_obj_list:
    if bbox is None:
        top, bottom, left, right = [dict(rect) for x in range(4)]
        top.update(
            {
                "object_type": "rect_edge",
                "height": 0,
                "y0": rect["y1"],
                "bottom": rect["top"],
                "orientation": "h",
            }
        )
        bottom.update(
            {
                "object_type": "rect_edge",
                "height": 0,
                "y1": rect["y0"],
                "top": rect["top"] + rect["height"],
                "doctop": rect["doctop"] + rect["height"],
                "orientation": "h",
            }
        )
        left.update(
            {
                "object_type": "rect_edge",
                "width": 0,
                "x1": rect["x0"],
                "orientation": "v",
            }
        )
        right.update(
            {
                "object_type": "rect_edge",
                "width": 0,
                "x0": rect["x1"],
                "orientation": "v",
            }
        )
        return [top, bottom, left, right]

    src_x0, src_top, src_x1, src_bottom = _rect_corners(rect)
    candidates = [
        ((src_x0, src_top), (src_x1, src_top), "h"),
        ((src_x0, src_bottom), (src_x1, src_bottom), "h"),
        ((src_x0, src_top), (src_x0, src_bottom), "v"),
        ((src_x1, src_top), (src_x1, src_bottom), "v"),
    ]
    edges = []
    for p0, p1, orientation in candidates:
        seg = clip_segment(p0, p1, bbox)
        if seg is None:
            continue
        edges.append(
            _edge_from_segment(rect, "rect_edge", seg[0], seg[1], orientation, src_top)
        )
    return edges


def line_to_edge(line: T_obj, bbox: Optional[T_bbox] = None) -> Optional[T_obj]:
    edge = dict(line)
    edge["orientation"] = "h" if (line["top"] == line["bottom"]) else "v"
    if bbox is None:
        return edge

    src_x0, src_top, src_x1, src_bottom = _source_bbox(line)
    seg = clip_segment((src_x0, src_top), (src_x1, src_bottom), bbox)
    if seg is None:
        return None
    return _edge_from_segment(
        line, "line", seg[0], seg[1], edge["orientation"], src_top
    )


def obj_to_edges(obj: T_obj, bbox: Optional[T_bbox] = None) -> T_obj_list:
    t = obj["object_type"]
    if "_edge" in t:
        if bbox is None:
            # Copy: callers may hand us their own dicts, and neither table
            # finding nor later merging may modify them.
            return [dict(obj)]
        edge = clip_edge_to_bbox(obj, bbox)
        return [] if edge is None else [edge]
    elif t == "line":
        edge = line_to_edge(obj, bbox)
        return [] if edge is None else [edge]
    else:
        return {"rect": rect_to_edges, "curve": curve_to_edges}[t](obj, bbox)


def filter_edges(
    edges: Iterable[T_obj],
    orientation: Optional[str] = None,
    edge_type: Optional[str] = None,
    min_length: T_num = 1,
) -> T_obj_list:
    if orientation not in ("v", "h", None):
        raise ValueError("Orientation must be 'v' or 'h'")

    def test(e: T_obj) -> bool:
        dim = "height" if e["orientation"] == "v" else "width"
        et_correct = e["object_type"] == edge_type if edge_type is not None else True
        orient_correct = orientation is None or e["orientation"] == orientation
        return bool(et_correct and orient_correct and (e[dim] >= min_length))

    return list(filter(test, edges))
