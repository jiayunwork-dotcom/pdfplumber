"""Table discovery for ``rotation_mode="auto"``.

The auto finder works in a coordinate frame local to each rotation: ``u``
runs in the direction the text is read and ``v`` runs towards the next text
line.  Words come from
:func:`pdfplumber.utils.rotation.extract_words_auto`, so opposite
orientations are never mixed.

Outline of the algorithm (all steps are near-linear in the number of words
once sorted, and none enumerate word, cell, or character pairs):

1. Cluster words into raw text lines using ``y_tolerance``.
2. Build *column cores* by clustering aligned word starts, ends and centres
   that recur on at least ``min_words_vertical`` distinct lines.  Cores that
   share a word are merged, because one word's start, centre and end all
   describe the same column.
3. Connect neighbouring cores that share at least two raw text lines.  A
   paragraph beside a table shares at most one baseline and so stays out.
4. Split each connected component at a blank gutter wider than twice the
   component's median positive inter-core gutter.  This is the documented
   spacing convention for same-orientation neighbours, and also separates
   side-by-side tables whose rows are baseline-aligned.  A fragment with
   fewer than two columns is merged back, since a one-column table is never
   emitted and such fragments arise from wide columns inside a table.
5. Assign every word to the nearest core start within the component's reach,
   extending each column's extent to the words it owns.  This keeps words
   that legitimately run past a narrow anchor in the correct cell while
   leaving a neighbouring table or paragraph alone.
6. Take lines that touch at least two columns as row seeds, split them into
   vertical runs at ``ROW_RUN_FACTOR`` times the median seed pitch, then
   cluster all lines of a run into logical rows using half the seed pitch
   (a cell's second line is closer than that in normal tables).
7. Emit one :class:`~pdfplumber.table.Table` per (column group, row run).
   The cells are stored in page coordinates; the table's ``rows`` and
   ``columns`` are returned in logical reading order.

Candidate tables that claim the same character raise
:class:`AmbiguousTableError`; the ambiguity is never resolved by traversal
order.  Pages whose evidence is insufficient yield no tables.
"""

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ._typing import T_num, T_obj, T_obj_list
from .table import Table, TableSettings
from .utils.rotation import (
    ROTATIONS,
    from_local_box,
    to_local_box,
    validate_auto_settings,
)

#: Tolerance (points) for clustering word starts/ends/centres into anchors.
ANCHOR_TOLERANCE = 1.0

#: Gutter multiplier used to split a component into separate tables.
GUTTER_FACTOR = 2.0

#: Multiplier applied to the median seed pitch to split vertically separated
#: row runs.
ROW_RUN_FACTOR = 1.75

#: A table must contain at least this many logical rows and columns.
MIN_ROWS = 2
MIN_COLS = 2


class AmbiguousTableError(ValueError):
    """Raised when two candidate tables claim the same characters.

    ``tables`` holds the competing candidates as
    ``(rotation, page_bbox, table)`` tuples.
    """

    def __init__(self, message: str, tables: Sequence[Tuple[Any, ...]]):
        super().__init__(message)
        self.tables = list(tables)


def _median(values: Iterable[T_num]) -> T_num:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _cluster_indices(
    pairs: List[Tuple[T_num, int]], tolerance: T_num
) -> List[List[int]]:
    """Cluster ``(value, index)`` pairs, returning lists of indices."""

    pairs = sorted(pairs, key=lambda pair: pair[0])
    clusters: List[List[int]] = []
    current: List[int] = []
    last: Optional[T_num] = None
    for value, index in pairs:
        if last is None or (value - last) <= tolerance:
            current.append(index)
        else:
            clusters.append(current)
            current = [index]
        last = value
    if current:
        clusters.append(current)
    return clusters


class _LocalWord:
    __slots__ = ("word", "u0", "v0", "u1", "v1")

    def __init__(self, word: T_obj) -> None:
        self.word = word
        self.u0 = 0.0
        self.v0 = 0.0
        self.u1 = 0.0
        self.v1 = 0.0

    @property
    def uc(self) -> T_num:
        return (self.u0 + self.u1) / 2.0

    @property
    def vc(self) -> T_num:
        return (self.v0 + self.v1) / 2.0

    @property
    def text(self) -> str:
        return str(self.word["text"])


class _Line:
    __slots__ = ("words", "v0", "v1", "u0", "u1")

    def __init__(self, words: List[int]) -> None:
        self.words = words
        self.v0 = 0.0
        self.v1 = 0.0
        self.u0 = 0.0
        self.u1 = 0.0


class _Core:
    __slots__ = ("words", "lines", "u0", "u1")

    def __init__(self, words: List[int], lines: Set[int]) -> None:
        self.words = words
        self.lines = lines
        self.u0 = 0.0
        self.u1 = 0.0


class AutoTableFinder:
    """Find tables on a page, respecting each word's rotation."""

    def __init__(self, page: Any, settings: Optional[Any] = None):
        self.page = page
        self.settings = TableSettings.resolve(settings)
        self._validate()
        # Reject unsupported character transforms before doing any work, so
        # configuration/transform errors surface before results are returned.
        self._validate_transforms()
        text_settings = dict(self.settings.text_settings or {})
        text_settings.pop("rotation_mode", None)
        text_settings["rotation_mode"] = "auto"
        self.words = page.extract_words(**text_settings)
        #: Diagnostic counters: sizes that the algorithm actually touches.
        self.stats: Dict[str, int] = {
            "words": len(self.words),
            "lines": 0,
            "anchor_groups": 0,
            "column_cores": 0,
            "components": 0,
            "candidates": 0,
        }
        self.tables = self._find_tables()
        self.stats["candidates"] = len(self.tables)
        # Debug-parity attributes with TableFinder.  Auto discovery does not
        # build geometric edges/intersections, so those are empty; ``cells``
        # exposes every emitted cell for inspection.
        self.edges: List[T_obj] = []
        self.intersections: Dict[Any, Any] = {}
        self.cells = [cell for table in self.tables for cell in table.cells]

    def _validate_transforms(self) -> None:
        from .utils.rotation import matrix_rotation

        for char in self.page.chars:
            if not char.get("text", ""):
                continue
            matrix_rotation(char["matrix"])

    # -- settings ---------------------------------------------------------
    def _validate(self) -> None:
        settings = self.settings
        if (
            settings.vertical_strategy != "text"
            or settings.horizontal_strategy != "text"
        ):
            raise ValueError(
                "rotation_mode='auto' requires vertical_strategy and "
                "horizontal_strategy to both be 'text'; received "
                f"{settings.vertical_strategy!r} and "
                f"{settings.horizontal_strategy!r}."
            )
        text_settings = settings.text_settings or {}
        forwarded_mode = text_settings.get("rotation_mode")
        if forwarded_mode is not None and forwarded_mode != "auto":
            raise ValueError(
                "rotation_mode='auto' requires a text_rotation_mode of "
                f"'auto' when that setting is supplied; received "
                f"{forwarded_mode!r}."
            )
        validate_auto_settings(text_settings, prefix="text_")

    # -- word / line helpers ---------------------------------------------
    def _localize(self, rotation: int, words: T_obj_list) -> List[_LocalWord]:
        localized = []
        for word in words:
            item = _LocalWord(word)
            item.u0, item.v0, item.u1, item.v1 = to_local_box(
                (word["x0"], word["top"], word["x1"], word["bottom"]), rotation
            )
            localized.append(item)
        return localized

    def _make_lines(self, words: List[_LocalWord], y_tolerance: T_num) -> List[_Line]:
        pairs = [(item.v0, index) for index, item in enumerate(words)]
        line_index_groups = _cluster_indices(pairs, max(y_tolerance, 1.0))
        lines = []
        for indices in line_index_groups:
            indices.sort(key=lambda index: (words[index].u0, words[index].v0))
            line = _Line(list(indices))
            line.v0 = min(words[index].v0 for index in indices)
            line.v1 = max(words[index].v1 for index in indices)
            line.u0 = min(words[index].u0 for index in indices)
            line.u1 = max(words[index].u1 for index in indices)
            lines.append(line)
        lines.sort(key=lambda line: (line.v0, line.u0))
        return lines

    # -- column cores -----------------------------------------------------
    def _column_cores(self, words: List[_LocalWord], line_of: List[int]) -> List[_Core]:
        min_lines = max(1, int(self.settings.min_words_vertical))
        raw: List[_Core] = []
        for attr in ("u0", "u1"):
            pairs = [(getattr(item, attr), index) for index, item in enumerate(words)]
            for members in _cluster_indices(pairs, ANCHOR_TOLERANCE):
                lines = {line_of[index] for index in members}
                if len(lines) >= min_lines:
                    raw.append(_Core(list(dict.fromkeys(members)), lines))
        pairs = [((item.u0 + item.u1) / 2.0, index) for index, item in enumerate(words)]
        for members in _cluster_indices(pairs, ANCHOR_TOLERANCE):
            lines = {line_of[index] for index in members}
            if len(lines) >= min_lines:
                raw.append(_Core(list(dict.fromkeys(members)), lines))

        self.stats["anchor_groups"] += len(raw)
        if not raw:
            return []

        parent = list(range(len(raw)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        by_word: Dict[int, List[int]] = defaultdict(list)
        for core_index, core in enumerate(raw):
            for word_index in core.words:
                by_word[word_index].append(core_index)
        for group in by_word.values():
            for other in group[1:]:
                union(group[0], other)

        merged: Dict[int, _Core] = {}
        for core_index, core in enumerate(raw):
            root = find(core_index)
            if root not in merged:
                merged[root] = _Core([], set())
            target = merged[root]
            target.words.extend(core.words)
            target.lines |= core.lines

        cores = []
        for core in merged.values():
            core.words = list(dict.fromkeys(core.words))
            core.u0 = min(words[index].u0 for index in core.words)
            core.u1 = max(words[index].u1 for index in core.words)
            cores.append(core)
        cores.sort(key=lambda core: (core.u0, core.u1))
        return cores

    def _connect_cores(
        self, cores: List[_Core], line_of: List[int]
    ) -> List[List[_Core]]:
        """Group column cores into table components.

        Two neighbouring cores belong to the same table only if they share at
        least two raw text lines.  A paragraph running beside a table can
        produce a repeated left anchor, but it shares at most one baseline
        with a table row, so it does not connect.  Cores connected this way
        are then split at a gutter wider than twice the normal inter-column
        gap of the component; this is the documented convention for
        side-by-side tables (the blank band between tables is at least twice
        the table's normal column spacing).

        The split threshold deliberately ignores the largest gap when
        computing the "normal" spacing, so a two-column table whose second
        column is far away still splits into two tables instead of averaging
        the huge gap into the median.
        """

        n = len(cores)
        if n == 0:
            return []

        line_cores: Dict[int, List[int]] = defaultdict(list)
        for core_index, core in enumerate(cores):
            for word_index in core.words:
                line_cores[line_of[word_index]].append(core_index)

        co_occurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        for core_indices in line_cores.values():
            unique = sorted(set(core_indices))
            # Connect each core to the next occupied core on the same text
            # line.  This is linear in the number of core *appearances*, so
            # no core/word/cell pair enumeration is needed.
            for left, right in zip(unique, unique[1:]):
                co_occurrence[(left, right)] += 1

        parent = list(range(n))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (left, right), count in co_occurrence.items():
            if count >= 2:
                union(left, right)

        components: Dict[int, List[int]] = defaultdict(list)
        for index in range(n):
            components[find(index)].append(index)

        groups: List[List[_Core]] = []
        for component in components.values():
            ordered = sorted(component)
            # Split at a *blank gutter* wider than twice the component's
            # median positive inter-core gutter.  This is the documented
            # spacing convention for side-by-side tables.  Using the blank
            # gutter (not start-to-start pitch) tolerates tables whose
            # columns have uneven widths: a wide column with long text has a
            # small blank gutter even when its pitch is large.
            gaps = [
                cores[right].u0 - cores[left].u1
                for left, right in zip(ordered, ordered[1:])
            ]
            positive = sorted(gap for gap in gaps if gap > 0)
            threshold = GUTTER_FACTOR * _median(positive) if positive else math.inf

            local: List[List[_Core]] = []
            current = [cores[ordered[0]]]
            for left, right, gap in zip(ordered, ordered[1:], gaps):
                joined = co_occurrence.get((left, right), 0) >= 2
                if joined and gap <= threshold:
                    current.append(cores[right])
                else:
                    local.append(current)
                    current = [cores[right]]
            local.append(current)

            # A split that would leave a fragment with fewer than MIN_COLS
            # cores is almost certainly a wide column inside one table, not a
            # table boundary (a one-column "table" is never emitted).  Merge
            # such a fragment back into its nearest *connected* neighbour
            # within the same component only, so a separate side-by-side
            # table is never rejoined.
            while len(local) > 1:
                small_index = -1
                for candidate, group in enumerate(local):
                    if len(group) < MIN_COLS:
                        small_index = candidate
                        break
                if small_index < 0:
                    break
                index = small_index
                if index > 0:
                    local[index - 1].extend(local[index])
                else:
                    local[index + 1] = local[index] + local[index + 1]
                del local[index]

            for group in local:
                if len(group) >= MIN_COLS:
                    groups.append(group)

        groups = [
            sorted(group, key=lambda core: (core.u0, core.u1)) for group in groups
        ]
        return [group for group in groups if len(group) >= MIN_COLS]

    # -- assignment -------------------------------------------------------
    def _assign_columns(
        self, cores: List[_Core], lines: List[_Line], words: List[_LocalWord]
    ) -> Tuple[List[T_num], List[List[int]]]:
        """Assign each word to a core and compute final column bounds.

        Words are only assigned inside the component's horizontal reach: the
        outer columns are widened by half the typical column pitch.  Text
        farther away (a paragraph beside the table, another table) is left
        alone instead of being swallowed by the nearest column.
        """

        starts = [core.u0 for core in cores]
        reach_left, reach_right = self._component_reach(cores)

        owner = [-1] * len(words)
        for line in lines:
            for word_index in line.words:
                uc = words[word_index].uc
                if uc < reach_left or uc > reach_right:
                    continue
                # Find the core whose start is nearest at or before the word.
                position = max(0, min(len(starts) - 1, self._bisect(starts, uc)))
                best = position
                if position > 0:
                    left_distance = abs(uc - starts[position - 1])
                    right_distance = abs(uc - starts[position])
                    if left_distance < right_distance:
                        best = position - 1
                owner[word_index] = best

        per_column_words: List[List[int]] = [[] for _ in cores]
        for word_index, column in enumerate(owner):
            if column >= 0:
                per_column_words[column].append(word_index)

        bounds = [cores[0].u0]
        for left, right in zip(range(len(cores) - 1), range(1, len(cores))):
            left_words = per_column_words[left]
            right_words = per_column_words[right]
            left_end = max(
                (words[i].u1 for i in left_words),
                default=cores[left].u1,
            )
            right_start = min(
                (words[i].u0 for i in right_words),
                default=cores[right].u0,
            )
            if right_start <= left_end:
                midpoint = (cores[left].u1 + cores[right].u0) / 2.0
            else:
                midpoint = (left_end + right_start) / 2.0
            bounds.append(midpoint)
        last_end = max(
            (words[i].u1 for i in per_column_words[-1]),
            default=cores[-1].u1,
        )
        bounds.append(min(last_end, reach_right))
        for index in range(1, len(bounds)):
            if bounds[index] <= bounds[index - 1]:
                bounds[index] = math.nextafter(bounds[index - 1], math.inf)
        return bounds, per_column_words

    @staticmethod
    def _bisect(sorted_values: List[T_num], value: T_num) -> int:
        low, high = 0, len(sorted_values)
        while low < high:
            mid = (low + high) // 2
            if sorted_values[mid] <= value:
                low = mid + 1
            else:
                high = mid
        return max(0, low - 1)

    @staticmethod
    def _lower_bound(values: List[T_num], target: T_num) -> int:
        low, high = 0, len(values)
        while low < high:
            mid = (low + high) // 2
            if values[mid] < target:
                low = mid + 1
            else:
                high = mid
        return low

    @staticmethod
    def _upper_bound(values: List[T_num], target: T_num) -> int:
        low, high = 0, len(values)
        while low < high:
            mid = (low + high) // 2
            if values[mid] <= target:
                low = mid + 1
            else:
                high = mid
        return low

    @staticmethod
    def _column_of(bounds: List[T_num], value: T_num) -> Optional[int]:
        low, high = 0, len(bounds) - 1
        while low + 1 < high:
            mid = (low + high) // 2
            if value < bounds[mid]:
                high = mid
            else:
                low = mid
        if bounds[low] <= value < bounds[low + 1]:
            return low
        return None

    def _occupancy(
        self, lines: List[_Line], bounds: List[T_num], words: List[_LocalWord]
    ) -> List[Set[int]]:
        result = []
        for line in lines:
            occupied = set()
            for word_index in line.words:
                column = self._column_of(bounds, words[word_index].uc)
                if column is not None:
                    occupied.add(column)
            result.append(occupied)
        return result

    # -- rows -------------------------------------------------------------
    def _row_runs(self, lines: List[_Line], occ: List[Set[int]]) -> List[List[int]]:
        seeds = [index for index, occupied in enumerate(occ) if len(occupied) >= 2]
        if len(seeds) < MIN_ROWS:
            return []
        seeds.sort(key=lambda index: lines[index].v0)
        tops = [lines[index].v0 for index in seeds]
        deltas = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
        positive = [delta for delta in deltas if delta > 0]
        pitch = _median(positive) if positive else 0.0
        if pitch <= 0:
            return [seeds]
        runs = [[seeds[0]]]
        for index, delta in enumerate(deltas):
            if delta > ROW_RUN_FACTOR * pitch:
                runs.append([seeds[index + 1]])
            else:
                runs[-1].append(seeds[index + 1])
        return [run for run in runs if len(run) >= MIN_ROWS]

    # -- main -------------------------------------------------------------
    def _find_tables(self) -> List[Table]:
        candidates: List[Table] = []
        for rotation in ROTATIONS:
            group = [word for word in self.words if word["rotation"] == rotation]
            if len(group) < MIN_COLS * MIN_ROWS:
                continue
            candidates.extend(self._find_for_rotation(rotation, group))
        self._raise_on_ambiguity(candidates)
        return candidates

    def _find_for_rotation(self, rotation: int, group: T_obj_list) -> List[Table]:
        y_tolerance = float((self.settings.text_settings or {}).get("y_tolerance", 3))
        words = self._localize(rotation, group)
        self._words = words
        lines = self._make_lines(words, y_tolerance)
        self.stats["lines"] += len(lines)

        line_of = [0] * len(words)
        for line_index, line in enumerate(lines):
            for word_index in line.words:
                line_of[word_index] = line_index

        cores = self._column_cores(words, line_of)
        self.stats["column_cores"] += len(cores)
        if len(cores) < MIN_COLS:
            return []

        tables: List[Table] = []
        components = self._connect_cores(cores, line_of)
        self.stats["components"] += len(components)
        line_tops = [line.v0 for line in lines]
        for column_cores in components:
            member_words = [index for core in column_cores for index in core.words]
            reach_left, reach_right = self._component_reach(column_cores)
            v0 = min(words[index].v0 for index in member_words)
            v1 = max(words[index].v1 for index in member_words)
            # Only lines that vertically intersect the component are
            # candidates; a binary search avoids scanning the whole page per
            # component, keeping the work proportional to words plus the
            # candidate's own size.
            lo = self._lower_bound(line_tops, v0 - max(1.0, y_tolerance))
            hi = self._upper_bound(line_tops, v1 + max(1.0, y_tolerance))
            group_lines = self._filter_lines(
                lines[lo:hi], set(member_words), reach_left, reach_right
            )
            if not group_lines:
                continue
            bounds, _per_column_words = self._assign_columns(
                column_cores, group_lines, words
            )
            occluded = self._occupancy(group_lines, bounds, words)
            for run in self._row_runs(group_lines, occluded):
                table = self._build_table(
                    rotation, group_lines, occluded, bounds, run, words
                )
                if table is not None:
                    tables.append(table)
        return tables

    def _component_reach(self, cores: List[_Core]) -> Tuple[T_num, T_num]:
        if len(cores) > 1:
            pitches = [
                cores[index + 1].u0 - cores[index].u0 for index in range(len(cores) - 1)
            ]
            pitch = _median([value for value in pitches if value > 0])
        else:
            pitch = max(cores[0].u1 - cores[0].u0, 1.0)
        return cores[0].u0 - 0.5 * pitch, cores[-1].u1 + 0.5 * pitch

    def _filter_lines(
        self,
        lines: List[_Line],
        members: Set[int],
        reach_left: T_num,
        reach_right: T_num,
    ) -> List[_Line]:
        """Keep the words of lines that touch this component.

        Words are restricted to the component's horizontal reach so a
        neighbouring table or paragraph sharing the same baseline does not
        leak into this table.  Non-anchor words (the second token of a cell,
        for example) are kept because they fall inside the reach.
        """

        filtered = []
        for line in lines:
            if not any(index in members for index in line.words):
                continue
            kept = [
                index
                for index in line.words
                if reach_left <= self._words[index].uc <= reach_right
            ]
            if not kept:
                continue
            new_line = _Line(kept)
            new_line.v0 = min(self._words[index].v0 for index in kept)
            new_line.v1 = max(self._words[index].v1 for index in kept)
            new_line.u0 = min(self._words[index].u0 for index in kept)
            new_line.u1 = max(self._words[index].u1 for index in kept)
            filtered.append(new_line)
        return filtered

    def _has_table_evidence(
        self,
        lines: List[_Line],
        rows: List[List[int]],
        bounds: List[T_num],
        words: List[_LocalWord],
    ) -> bool:
        """Reject paragraph-like structures that only have word-space gutters.

        For each adjacent pair of detected columns the finder measures, per
        logical row, the horizontal gap between the last word of the left
        column and the first word of the right column.  A real table has at
        least one column boundary whose *median* gutter clearly exceeds the
        page's ordinary inter-word spacing; a justified paragraph, whose
        apparent columns are merely margins, has only word spaces at every
        boundary and is rejected.
        """

        word_gaps = []
        for line in lines:
            ordered = sorted(line.words, key=lambda index: words[index].u0)
            for left, right in zip(ordered, ordered[1:]):
                # Skip pairs that straddle a column boundary: their gap is a
                # gutter, not ordinary word spacing.
                left_column = self._column_of(bounds, words[left].uc)
                right_column = self._column_of(bounds, words[right].uc)
                if left_column != right_column:
                    continue
                gap = words[right].u0 - words[left].u1
                if gap > 0:
                    word_gaps.append(gap)
        typical_word_gap = _median(word_gaps) if word_gaps else 0.0
        heights = [item.v1 - item.v0 for item in words if item.v1 > item.v0]
        typical_height = _median(heights) if heights else 0.0
        minimum_gutter = max(2.0 * typical_word_gap, 0.3 * typical_height, 0.5)

        has_wide_gutter = False
        for column in range(len(bounds) - 2):
            gutters = []
            for row in rows:
                left_edges = []
                right_edges = []
                for line_index in row:
                    for word_index in lines[line_index].words:
                        item = words[word_index]
                        if bounds[column] <= item.uc < bounds[column + 1]:
                            left_edges.append(item.u1)
                        elif bounds[column + 1] <= item.uc < bounds[column + 2]:
                            right_edges.append(item.u0)
                if left_edges and right_edges:
                    gutters.append(min(right_edges) - max(left_edges))
            if gutters and _median(gutters) >= minimum_gutter:
                has_wide_gutter = True
                break

        if not has_wide_gutter:
            return False

        # Distinguish a table from ordinary two-column prose.  Prose lines
        # are wrapped sentences, so most column-rows contain many words; a
        # table cell is a short value, even in a descriptive column.  This is
        # a deliberate conservative bound: pages that look like multi-column
        # prose are refused rather than guessed at.
        for column in range(len(bounds) - 1):
            left_bound = bounds[column]
            right_bound = bounds[column + 1]
            counts = []
            for row in rows:
                count = sum(
                    1
                    for line_index in row
                    for index in lines[line_index].words
                    if left_bound <= words[index].uc < right_bound
                )
                if count:
                    counts.append(count)
            if counts and _median(counts) >= 5:
                return False
        return True

    def _build_table(
        self,
        rotation: int,
        lines: List[_Line],
        occ: List[Set[int]],
        bounds: List[T_num],
        run: List[int],
        words: List[_LocalWord],
    ) -> Optional[Table]:
        run_top = min(lines[index].v0 for index in run)
        run_bottom = max(lines[index].v1 for index in run)

        seed_tops = sorted(lines[index].v0 for index in run)
        seed_deltas = [
            seed_tops[index + 1] - seed_tops[index]
            for index in range(len(seed_tops) - 1)
        ]
        positive = [delta for delta in seed_deltas if delta > 0]
        pitch = _median(positive) if positive else 0.0
        tolerance = 0.5 * pitch if pitch > 0 else 0.0

        in_run = [
            index
            for index in range(len(lines))
            if occ[index] and run_top - 1e-9 <= lines[index].v0 <= run_bottom + 1e-9
        ]
        in_run.sort(key=lambda index: lines[index].v0)

        rows: List[List[int]] = []
        for index in in_run:
            top = lines[index].v0
            if rows and top - lines[rows[-1][0]].v0 <= tolerance:
                rows[-1].append(index)
            else:
                rows.append([index])

        if len(rows) < MIN_ROWS:
            return None

        if not self._has_table_evidence(lines, rows, bounds, words):
            return None

        used_columns = sorted(set().union(*[occ[index] for index in run]))
        if len(used_columns) < MIN_COLS:
            return None

        used_bounds = [bounds[column] for column in used_columns]
        used_bounds.append(bounds[used_columns[-1] + 1])
        for index in range(1, len(used_bounds)):
            if used_bounds[index] <= used_bounds[index - 1]:
                used_bounds[index] = math.nextafter(used_bounds[index - 1], math.inf)

        row_bounds = [lines[rows[0][0]].v0]
        for previous, current in zip(rows, rows[1:]):
            prev_bottom = max(lines[index].v1 for index in previous)
            curr_top = min(lines[index].v0 for index in current)
            row_bounds.append((prev_bottom + curr_top) / 2.0)
        row_bounds.append(max(lines[index].v1 for index in rows[-1]))
        for index in range(1, len(row_bounds)):
            if row_bounds[index] <= row_bounds[index - 1]:
                row_bounds[index] = math.nextafter(row_bounds[index - 1], math.inf)

        local_cells = [
            (
                used_bounds[column],
                row_bounds[row],
                used_bounds[column + 1],
                row_bounds[row + 1],
            )
            for row in range(len(rows))
            for column in range(len(used_bounds) - 1)
        ]
        page_cells = [from_local_box(cell, rotation) for cell in local_cells]
        return Table(self.page, page_cells, rotation=rotation)

    # -- ambiguity --------------------------------------------------------
    def _raise_on_ambiguity(self, candidates: List[Table]) -> None:
        from .utils.geometry import calculate_area, get_bbox_overlap

        if len(candidates) < 2:
            return
        # Sweep by top coordinate so only pairs whose bboxes can overlap are
        # examined; no cell, word, or character pair enumeration is done.
        ordered = sorted(enumerate(candidates), key=lambda item: item[1].bbox[1])
        active: List[Tuple[int, Table]] = []
        for index, table in ordered:
            active = [entry for entry in active if entry[1].bbox[3] > table.bbox[1]]
            for other_index, other in active:
                overlap = get_bbox_overlap(table.bbox, other.bbox)
                if overlap is None:
                    continue
                shared = table.claimed_char_ids() & other.claimed_char_ids()
                if shared:
                    # Two candidates claiming the same character is always a
                    # genuine conflict, however small the box overlap.
                    pass
                else:
                    overlap_area = calculate_area(overlap)
                    smaller_area = min(
                        calculate_area(table.bbox), calculate_area(other.bbox)
                    )
                    if smaller_area <= 0:
                        continue
                    # Mere touching or a sliver of overlap between two
                    # same-rotation candidates is not a genuine ownership
                    # conflict; require a substantial intersection of the
                    # smaller table's area.
                    if (
                        table.rotation != other.rotation
                        or overlap_area < 0.2 * smaller_area
                    ):
                        continue
                if shared or table.rotation == other.rotation:
                    raise AmbiguousTableError(
                        (
                            "Ambiguous table detection: candidate tables "
                            f"(rotation {other.rotation}, bbox {other.bbox}) "
                            f"and (rotation {table.rotation}, bbox {table.bbox}) "
                            "overlap and cannot be assigned unique characters."
                            if shared
                            else "Ambiguous table detection: two candidate tables "
                            "with the same rotation have overlapping bounding "
                            f"boxes: {other.bbox} and {table.bbox}."
                        ),
                        [
                            (other.rotation, other.bbox, other),
                            (table.rotation, table.bbox, table),
                        ],
                    )
            active.append((index, table))
