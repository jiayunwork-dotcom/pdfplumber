# Crop semantics and derived geometry

This note records how `pdfplumber` derives edges for cropped views, why it is
implemented that way, and what it costs.

## The rule

An edge reported for a view must be a piece of a stroke that the PDF actually
drew, intersected with that view's region. Concretely:

* A rectangle contributes its four drawn sides, clipped to the region. A crop
  never adds sides along its own boundary, and a rectangle entirely outside
  the region contributes nothing.
* A `curve` path is clipped segment by segment. A path that leaves the region
  and returns is cut at each crossing; the gap between the pieces is not
  bridged. A closed path keeps only the closing side it drew — never an added
  one.
* A `line` is clipped like any other segment.
* A stroke exactly on the region boundary is kept (the clip is inclusive).
  Every returned edge lies inside the region.

The same intersections feed table finding: snapping, joining, intersecting and
cell construction operate only on the pieces that survived the clip.

`.crop(...)` establishes a region. `.within_bbox(...)`, `.outside_bbox(...)`
and `.filter(...)` only select objects; they do not re-crop, so they keep the
parent's region. A crop whose box equals the parent's full box removes
nothing, so it is treated as the parent view and leaves geometry untouched.

## Where the pieces come from

`crop_to_bbox` rewrites an object's bounding-box fields (`x0`, `top`, ...) but
leaves its `pts` / `path` intact. That is the "geometry source": a clipped
object still remembers the stroke it was drawn from. Edge derivation reads the
source points and clips each drawn segment against the view's region. This is
handled inside `utils.rect_to_edges`, `utils.curve_to_edges`,
`utils.line_to_edge`, `utils.obj_to_edges` and `utils.clip_edge_to_bbox`, so
the direct properties (`.rect_edges`, `.curve_edges`, `.edges`), a fresh
`TableFinder`, and object-form explicit hints all see the same geometry.

Numeric explicit hints are page coordinates. On a real crop they run the full
extent of the crop and are dropped if they fall outside it — before snapping,
so tolerance can never drag an outside hint onto the boundary. Object-form
hints are clipped exactly like page-derived edges.

## Design choice and cost

Two implementations were possible:

1. **Persist fragments.** When a page is filtered/cropped, split every path
   into its in-region pieces and store them.
2. **Keep the source and clip on demand** (chosen). Store nothing extra; when
   a view's edges are requested, derive them from the objects the crop
   already holds and clip against the view's box, caching the result on that
   view.

Option 2 is chosen because a crop already stores only the objects that
intersect its box; the source geometry is available for free in each such
object's `pts`. Deriving edges from that subset needs no additional
per-object copy of the full page's paths, and it avoids a second
representation that could drift from `pts`. The cache is per view and is
invalidated with the view, so interleaving reads of the original page and
several crops cannot leak geometry between them.

Consequences of the choice:

* **Work scales with intersecting paths, not page size.** In the benchmark's
  `localized` regime — scattered paths, small crops — a view holds a handful
  of objects regardless of how many paths the page has, and total work grows
  linearly in the number of paths but with a tiny constant.
* **Worst case is paths × views.** In the `full-width` regime every crop
  intersects every path, so each view must clip all of them. That is inherent
  to the requirement — those pieces really are in every view — not a copying
  artifact.
* **Memory tracks intersecting objects.** Peak traced memory rises with the
  total number of (view, intersecting-object) pairs, again proportional to
  what the crops already store.

Run `python examples/crop_edges/benchmark_crop_edges.py` for current numbers.

## Explicit input

`TableSettings` is never mutated. The same settings object may be passed to the
original page and to several crops; each view interprets numeric hints and
object hints against its own region. Object dictionaries supplied by the
caller are copied where they are returned, so neither page objects nor caller
configuration change.
