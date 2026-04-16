"""Citation detection across multiple citation styles.

Detects and extracts citations from scientific text:
- Numbered: [1], [2,3], [5-7], [1][2][3]
- Author-year parenthetical: (Smith, 2020), (Smith et al., 2020)
- Author-year bracketed: [Smith, 2020]
- Harvard (no comma): (Smith 2020), (Smith et al. 2020)
- Narrative: Smith (2020), Smith & Jones (2020), Smith et al. (2020)
- Multiple: (Smith, 2020; Jones, 2019)
- Multi-year: (Smith, 2019, 2020), (Smith, 2020a, 2020b)
- Three-author APA: (Smith, Jones, & Lee, 2020)
"""

import re
from dataclasses import dataclass, field


def normalize_author_name(name: str) -> str:
    """Normalize author name for consistent matching.

    Removes hyphens, dots, apostrophes, spaces; converts to lowercase.
    """
    name = name.lower().strip()
    name = re.sub(r"[-.'\s]", "", name)
    return name


@dataclass
class DetectedCitation:
    """A citation occurrence found in text."""

    marker: str  # e.g. "[5]", "(Smith, 2020)"
    keys: list[str] = field(default_factory=list)  # normalized IDs: ["5"] or ["smith2020"]
    position: int = 0  # character offset in text


# Author name pattern: supports "Smith", "Smith-Jones", "O'Brien",
# "De Vries", "Van der Berg", "Al-Rashid", "McDonald"
# Structure: optional prefix words (De, Van, Von, ...) + surname
_NAME_PREFIX = r"(?:[Dd]e|[Vv]an|[Vv]on|[Dd]el|[Dd]er|[Dd]en|[Dd]i|[Ll]a|[Ll]e|[Aa]l|[Ee]l|[Bb]in|[Ii]bn)"
_AUTHOR = rf"(?:{_NAME_PREFIX}\s+)*[A-Z][A-Za-z'\-]+"
_ET_AL = r"(?:\s+et\s+al\.?)?"
_YEAR = r"\d{4}[a-z]?"


class CitationDetector:
    """Detects citations in scientific text across all major styles."""

    def detect_all(self, text: str) -> list[DetectedCitation]:
        """Find all citation occurrences in text.

        Returns a list of DetectedCitation with marker, keys, and position.
        Deduplicates by position (same span won't be returned twice).
        """
        all_citations: list[DetectedCitation] = []

        all_citations.extend(self._find_numbered(text))
        all_citations.extend(self._find_adjacent_brackets(text))
        all_citations.extend(self._find_author_year_parenthetical(text))
        all_citations.extend(self._find_author_year_bracketed(text))
        all_citations.extend(self._find_harvard(text))
        all_citations.extend(self._find_narrative(text))
        all_citations.extend(self._find_multiple(text))
        all_citations.extend(self._find_multi_year(text))

        # Deduplicate by position — longer match wins when spans overlap
        all_citations.sort(key=lambda c: (c.position, -len(c.marker)))
        seen_ranges: list[tuple[int, int]] = []
        unique: list[DetectedCitation] = []
        for cit in all_citations:
            cit_end = cit.position + len(cit.marker)
            overlaps = any(
                not (cit_end <= s or cit.position >= e) for s, e in seen_ranges
            )
            if not overlaps:
                seen_ranges.append((cit.position, cit_end))
                unique.append(cit)

        return sorted(unique, key=lambda c: c.position)

    # --- Numbered: [1], [2,3], [5-7] ---

    _NUMBERED_RE = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")

    # Words that precede a bracket when it's NOT a citation (e.g. "Table [5]")
    _NON_CITATION_PREFIX = re.compile(
        r"(?:Table|Figure|Fig|Equation|Eq|Algorithm|Alg|Section|Sec|"
        r"Appendix|App|Step|Layer|Line|Theorem|Lemma|Definition|"
        r"Proposition|Corollary|Example|Case|Condition)\s*$",
        re.IGNORECASE,
    )

    # Bracket content that is NOT a citation (editorial, Latin abbreviations)
    _NON_CITATION_CONTENT = {
        "sic", "emphasis added", "emphasis mine", "translated",
        "ibid", "ibid.", "op. cit.", "loc. cit.", "dataset",
        "original emphasis", "my translation", "emphasis in original",
        "online", "accessed", "in press", "forthcoming", "unpublished",
    }

    # Math operators that precede a bracket in equations, not citations
    _MATH_PREFIX_RE = re.compile(r"[=+\-*/^]\s*$")

    def _find_numbered(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._NUMBERED_RE.finditer(text):
            marker = match.group(0)
            inner = marker.strip("[] ").lower()

            # Skip editorial/non-citation bracket content
            if inner in self._NON_CITATION_CONTENT:
                continue

            # Skip if preceded by a non-citation keyword
            prefix = text[max(0, match.start() - 30) : match.start()]
            if self._NON_CITATION_PREFIX.search(prefix):
                continue

            # Skip if preceded by math operator (likely array/index)
            if self._MATH_PREFIX_RE.search(prefix):
                continue

            # Skip if inside LaTeX math environment ($...$)
            if self._is_in_math_env(text, match.start()):
                continue

            # Skip [0, ...] — citations are 1-indexed
            numbers = self._parse_number_list(marker.strip("[] "))
            if numbers and numbers[0] == "0":
                continue

            # Skip enumeration: [N] at start of line followed by text
            if self._is_enumeration(text, match.start(), match.end()):
                continue

            results.append(
                DetectedCitation(marker=marker, keys=numbers, position=match.start())
            )
        return results

    # --- Author-year parenthetical: (Smith, 2020), (De Vries, 2020) ---

    _AUTHOR_YEAR_PAREN_RE = re.compile(
        rf"\(({_AUTHOR}{_ET_AL}"
        rf"(?:\s+(?:and|&)\s+{_AUTHOR}{_ET_AL})?"
        rf"(?:,\s*{_AUTHOR})*"
        rf"(?:,\s*(?:&|and)\s*{_AUTHOR})?"
        rf",\s*{_YEAR})\)"
    )

    def _find_author_year_parenthetical(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._AUTHOR_YEAR_PAREN_RE.finditer(text):
            marker = match.group(0)
            content = match.group(1)
            keys = self._parse_author_year(content)
            results.append(
                DetectedCitation(marker=marker, keys=keys, position=match.start())
            )
        return results

    # --- Author-year bracketed: [Smith, 2020] ---

    _AUTHOR_YEAR_BRACKET_RE = re.compile(
        rf"\[({_AUTHOR}{_ET_AL}"
        rf"(?:\s+(?:and|&)\s+{_AUTHOR}{_ET_AL})?"
        rf"(?:,\s*{_AUTHOR})*"
        rf"(?:,\s*(?:&|and)\s*{_AUTHOR})?"
        rf",\s*{_YEAR})\]"
    )

    def _find_author_year_bracketed(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._AUTHOR_YEAR_BRACKET_RE.finditer(text):
            marker = match.group(0)
            content = match.group(1)
            keys = self._parse_author_year(content)
            results.append(
                DetectedCitation(marker=marker, keys=keys, position=match.start())
            )
        return results

    # --- Narrative: Smith (2020), Smith and Jones (2020), Smith & Jones (2020) ---

    _NARRATIVE_RE = re.compile(
        rf"({_AUTHOR}{_ET_AL}"
        rf"(?:\s+(?:and|&)\s+{_AUTHOR}{_ET_AL})?)"
        rf"\s+\(({_YEAR})\)"
    )

    # Words that look like author names but are organizations/common nouns
    _NON_AUTHOR_WORDS = {
        "foundation", "organization", "organisation", "committee", "association",
        "institute", "institution", "university", "college", "department",
        "ministry", "government", "council", "commission", "agency", "company",
        "corporation", "group", "team", "project", "program", "programme",
        "network", "consortium", "society", "academy", "board", "authority",
        "conference", "workshop", "symposium", "congress", "chapter", "section",
        "version", "release", "edition", "update", "revision", "amendment",
    }

    def _find_narrative(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._NARRATIVE_RE.finditer(text):
            marker = match.group(0)
            author_part = match.group(1)
            year = match.group(2)

            # Skip if "author" is actually an organization/common noun
            first_word = author_part.split()[0].lower().rstrip(".,")
            if first_word in self._NON_AUTHOR_WORDS:
                continue

            # For the key, use the first author only (consistent with parenthetical)
            first_author = re.split(r"\s+(?:and|&)\s+", author_part)[0]
            first_author = first_author.replace(" et al.", "etal").replace(" et al", "etal")
            author_norm = normalize_author_name(first_author)
            key = f"{author_norm}{year}"

            results.append(
                DetectedCitation(marker=marker, keys=[key], position=match.start())
            )
        return results

    # --- Multiple: (Smith, 2020; Jones, 2019) ---

    _MULTI_PAREN_RE = re.compile(
        rf"\("
        rf"(?:{_AUTHOR}{_ET_AL},\s*{_YEAR}\s*;\s*)+"
        rf"{_AUTHOR}{_ET_AL},\s*{_YEAR}"
        rf"\)"
    )

    _MULTI_BRACKET_RE = re.compile(
        rf"\["
        rf"(?:{_AUTHOR}{_ET_AL},\s*{_YEAR}\s*;\s*)+"
        rf"{_AUTHOR}{_ET_AL},\s*{_YEAR}"
        rf"\]"
    )

    def _find_multiple(self, text: str) -> list[DetectedCitation]:
        results = []

        for pattern, strip_chars in [
            (self._MULTI_PAREN_RE, "()"),
            (self._MULTI_BRACKET_RE, "[]"),
        ]:
            for match in pattern.finditer(text):
                marker = match.group(0)
                content = marker.strip(strip_chars)
                keys: list[str] = []
                for part in content.split(";"):
                    part = part.strip()
                    if part:
                        keys.extend(self._parse_author_year(part))
                results.append(
                    DetectedCitation(
                        marker=marker, keys=keys, position=match.start()
                    )
                )

        return results

    # --- Harvard no-comma: (Smith 2020), (Smith et al. 2020) ---

    _HARVARD_PAREN_RE = re.compile(
        rf"\(({_AUTHOR}{_ET_AL}"
        rf"(?:\s+(?:and|&)\s+{_AUTHOR}{_ET_AL})?)"
        rf"\s+({_YEAR})\)"
    )

    def _find_harvard(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._HARVARD_PAREN_RE.finditer(text):
            marker = match.group(0)
            author_part = match.group(1)
            year = match.group(2)

            first_author = re.split(r"\s+(?:and|&)\s+", author_part)[0]
            first_author = first_author.replace(" et al.", "etal").replace(" et al", "etal")
            author_norm = normalize_author_name(first_author)
            key = f"{author_norm}{year}"

            results.append(
                DetectedCitation(marker=marker, keys=[key], position=match.start())
            )
        return results

    # --- Adjacent brackets: [1][2][3] ---

    _ADJACENT_BRACKET_RE = re.compile(r"(\[\d+\])(\s*\[\d+\])+")

    def _find_adjacent_brackets(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._ADJACENT_BRACKET_RE.finditer(text):
            marker = match.group(0)
            numbers = re.findall(r"\[(\d+)\]", marker)
            # Validate: skip zero-indexed
            if numbers and numbers[0] == "0":
                continue
            # Skip if preceded by non-citation keyword
            prefix = text[max(0, match.start() - 30) : match.start()]
            if self._NON_CITATION_PREFIX.search(prefix):
                continue
            results.append(
                DetectedCitation(marker=marker, keys=numbers, position=match.start())
            )
        return results

    # --- Multi-year: (Smith, 2019, 2020) or (Smith, 2020a, 2020b) ---

    _MULTI_YEAR_RE = re.compile(
        rf"\(({_AUTHOR}{_ET_AL}),\s*({_YEAR}(?:,\s*{_YEAR})+)\)"
    )

    def _find_multi_year(self, text: str) -> list[DetectedCitation]:
        results = []
        for match in self._MULTI_YEAR_RE.finditer(text):
            marker = match.group(0)
            author_part = match.group(1)
            years_str = match.group(2)

            first_author = author_part.replace(" et al.", "etal").replace(" et al", "etal")
            author_norm = normalize_author_name(first_author)

            years = re.findall(r"\d{4}[a-z]?", years_str)
            keys = [f"{author_norm}{y}" for y in years]

            results.append(
                DetectedCitation(marker=marker, keys=keys, position=match.start())
            )
        return results

    # --- Helpers ---

    @staticmethod
    def _parse_number_list(s: str) -> list[str]:
        """Parse '3, 5-7' into ['3', '5', '6', '7']."""
        numbers: list[str] = []
        for part in s.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-")
                    start_n, end_n = int(start), int(end)
                    if end_n - start_n > 100:
                        # Sanity limit: treat as single reference
                        numbers.append(str(start_n))
                    else:
                        numbers.extend(
                            str(i) for i in range(start_n, end_n + 1)
                        )
                except ValueError:
                    pass
            else:
                try:
                    numbers.append(str(int(part)))
                except ValueError:
                    pass
        return numbers

    @staticmethod
    def _is_enumeration(text: str, start: int, end: int) -> bool:
        """Check if a bracketed number is part of an enumeration list, not a citation.

        Heuristic: [N] at or near start of line, especially [1] or sequential.
        """
        # Find the start of the current line
        line_start = text.rfind("\n", 0, start) + 1
        before_on_line = text[line_start:start].strip()

        # [N] at start of line (possibly after whitespace) = enumeration
        if before_on_line == "":
            return True

        return False

    @staticmethod
    def _is_in_math_env(text: str, pos: int) -> bool:
        """Check if position is inside a LaTeX math environment ($...$)."""
        # Count unescaped $ signs before position
        before = text[:pos]
        dollar_count = 0
        i = 0
        while i < len(before):
            if before[i] == "$" and (i == 0 or before[i - 1] != "\\"):
                dollar_count += 1
            i += 1
        # Odd count means we're inside a $...$ environment
        return dollar_count % 2 == 1

    @staticmethod
    def _parse_author_year(content: str) -> list[str]:
        """Parse 'Smith et al., 2020a' into ['smithetal2020a']."""
        year_match = re.search(r"(\d{4})([a-z])?", content)
        if not year_match:
            return []

        year = year_match.group(1)
        suffix = year_match.group(2) or ""

        author_part = content[: year_match.start()].strip().rstrip(",")
        author_part = author_part.replace(" et al.", "etal").replace(" et al", "etal")
        author_part = re.sub(r"\s+and\s+.*$", "", author_part, flags=re.IGNORECASE)
        author_part = re.sub(r"\s*&\s*.*$", "", author_part)
        # Strip trailing commas left after removing co-authors
        author_part = author_part.rstrip(", ")
        # Remove internal commas (from multi-author lists like "Smith, Jones")
        # to keep only the first author surname
        if "," in author_part:
            author_part = author_part.split(",")[0].strip()
        author_norm = normalize_author_name(author_part)

        return [f"{author_norm}{year}{suffix}"]
