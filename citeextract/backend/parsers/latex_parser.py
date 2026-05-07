
import re
from pathlib import Path
from typing import Optional

from citeextract.citation.context_extractor import extract_context
from citeextract.models.citation import Citation
from citeextract.models.parsed_paper import ParsedPaper
from citeextract.models.reference import Reference
from citeextract.parsers.base import BaseParser
from citeextract.parsers.bibtex_parser import BibtexParser


_CITE_PATTERN = re.compile(
    r'\\(?:cite[tp]?|autocite|parencite|textcite|citealp|citeauthor|Cite[tp]?)\*?'
    r'(?:\[[^\]]*\])*'
    r'\{([^}]+)\}',
    re.MULTILINE,
)

_STRIP_COMMANDS = re.compile(
    r'\\(?:emph|textbf|textit|textsc|textrm|textsf|texttt|underline)\{([^}]*)\}'
)
_STRIP_BRACES = re.compile(r'[{}]')


def _find_brace_group(text: str, command_pattern: str) -> Optional[str]:
    match = re.search(command_pattern, text)
    if not match:
        return None
    start = match.end() - 1
    if start >= len(text) or text[start] != "{":
        return None
    depth = 1
    pos = start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start + 1 : pos - 1]


class LatexParser(BaseParser):

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".tex"

    def parse(self, file_path: str) -> ParsedPaper:
        tex_path = Path(file_path)
        tex_content = tex_path.read_text(encoding="utf-8", errors="replace")
        warnings: list[str] = []

        bib_path = self._find_bib_file(tex_path, tex_content)
        references: list[Reference] = []
        ref_by_key: dict[str, Reference] = {}

        if bib_path and bib_path.exists():
            bib_parser = BibtexParser()
            bib_result = bib_parser.parse(str(bib_path))
            references = bib_result.references
            ref_by_key = {ref.ref_id: ref for ref in references}
        else:
            warnings.append(
                "No .bib file found. Citation keys extracted but references "
                "will need to be resolved via database lookup."
            )

        clean_text = self._clean_latex(tex_content)
        citations: list[Citation] = []
        seen_keys: set[str] = set()

        for match in _CITE_PATTERN.finditer(tex_content):
            keys_str = match.group(1)
            raw_keys = [k.strip() for k in keys_str.split(',')]

            cite_pos = self._find_clean_position(tex_content, clean_text, match.start())

            ctx = extract_context(clean_text, cite_pos)

            for key in raw_keys:
                if not key:
                    continue

                if key not in ref_by_key and key not in seen_keys:
                    ref = Reference(
                        ref_id=key,
                        title=None,
                        raw_text=f"\\cite{{{key}}}",
                        source_format="latex",
                        citation_format=None,
                    )
                    references.append(ref)
                    ref_by_key[key] = ref
                    seen_keys.add(key)

                citations.append(Citation(
                    ref_id=key,
                    citing_sentence=ctx["citing_sentence"],
                    context_before=ctx["context_before"],
                    context_after=ctx["context_after"],
                    marker=f"\\cite{{{key}}}",
                    position=cite_pos,
                ))

        metadata = self._extract_metadata(tex_content)

        return ParsedPaper(
            references=references,
            citations=citations,
            has_body_text=True,
            body_text=clean_text,
            input_format="latex",
            metadata=metadata,
            warnings=warnings,
        )

    def _find_bib_file(self, tex_path: Path, tex_content: str) -> Path | None:
        bib_match = re.search(
            r'\\(?:bibliography|addbibresource)\{([^}]+)\}', tex_content
        )
        if bib_match:
            bib_name = bib_match.group(1)
            if not bib_name.endswith('.bib'):
                bib_name += '.bib'
            candidate = tex_path.parent / bib_name
            if candidate.exists():
                return candidate

        same_name = tex_path.with_suffix('.bib')
        if same_name.exists():
            return same_name

        bib_files = list(tex_path.parent.glob('*.bib'))
        if len(bib_files) == 1:
            return bib_files[0]

        return None

    @staticmethod
    def _clean_latex(tex: str) -> str:
        text = re.sub(r'(?<!\\)%.*$', '', tex, flags=re.MULTILINE)
        text = re.sub(r'\\(?:begin|end)\{[^}]*\}', '', text)
        for _ in range(3):
            text = _STRIP_COMMANDS.sub(r'\1', text)
        text = re.sub(r'\\(?!cite)[a-zA-Z]+\b\s*', ' ', text)
        text = _STRIP_BRACES.sub('', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _find_clean_position(raw: str, clean: str, raw_pos: int) -> int:
        cite_match = re.match(
            r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}',
            raw[raw_pos:raw_pos + 120],
        )
        if cite_match:
            key = cite_match.group(1).split(',')[0].strip()
            key_pos = clean.find(key)
            if key_pos >= 0:
                return key_pos

        for window in (40, 80, 20):
            snippet_start = max(0, raw_pos - window)
            snippet_end = min(len(raw), raw_pos + window)
            snippet = raw[snippet_start:snippet_end]
            clean_snippet = _STRIP_COMMANDS.sub(r'\1', snippet)
            clean_snippet = _STRIP_BRACES.sub('', clean_snippet).strip()

            for length in (30, 20, 10):
                search_str = clean_snippet[:length]
                if search_str:
                    idx = clean.find(search_str)
                    if idx >= 0:
                        return idx

        if len(raw) > 0:
            ratio = raw_pos / len(raw)
            return int(ratio * len(clean))
        return 0

    @staticmethod
    def _extract_metadata(tex: str) -> dict:
        metadata: dict = {}
        title_match = _find_brace_group(tex, r"\\title\s*\{")
        if title_match:
            metadata["title"] = title_match.replace("{", "").replace("}", "").strip()
        author_match = _find_brace_group(tex, r"\\author\s*\{")
        if author_match:
            metadata["authors_raw"] = author_match.replace("{", "").replace("}", "").strip()
        return metadata
