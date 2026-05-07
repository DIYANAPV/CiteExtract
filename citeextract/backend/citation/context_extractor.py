
import functools
import re
from typing import Optional

import spacy

from citeextract import config


def split_sentences(text: str, method: str = "spacy") -> list[str]:
    if method == "spacy":
        return _spacy_sentence_split(text)
    return _simple_sentence_split(text)


def _simple_sentence_split(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


@functools.lru_cache(maxsize=1)
def _get_spacy_nlp() -> spacy.Language:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    return nlp


def _spacy_sentence_split(text: str) -> list[str]:
    try:
        nlp = _get_spacy_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    except (ImportError, OSError):
        return _simple_sentence_split(text)


def _find_paragraph_bounds(text: str, position: int) -> tuple[int, int]:
    para_start = 0
    search_pos = position
    while search_pos > 0:
        nl_pos = text.rfind("\n\n", 0, search_pos)
        if nl_pos == -1:
            break
        para_start = nl_pos + 2
        break

    nl_pos = text.find("\n\n", position)
    para_end = nl_pos if nl_pos != -1 else len(text)

    return para_start, para_end


def extract_context(
    text: str,
    position: int,
    sentences_before: Optional[int] = None,
    sentences_after: Optional[int] = None,
    method: Optional[str] = None,
) -> dict:
    cfg = config.parsing()
    if sentences_before is None:
        sentences_before = cfg["context_sentences_before"]
    if sentences_after is None:
        sentences_after = cfg["context_sentences_after"]
    if method is None:
        method = cfg["sentence_splitter"]

    para_start, para_end = _find_paragraph_bounds(text, position)
    paragraph = text[para_start:para_end]
    local_position = position - para_start

    sentences = split_sentences(paragraph, method=method)
    if not sentences:
        return {"citing_sentence": "", "context_before": "", "context_after": ""}

    target_idx = len(sentences) - 1
    current_pos = 0
    for i, sentence in enumerate(sentences):
        sent_start = paragraph.find(sentence, current_pos)
        if sent_start == -1:
            continue
        sent_end = sent_start + len(sentence)

        if sent_start <= local_position < sent_end:
            target_idx = i
            break

        current_pos = sent_end

    start_idx = max(0, target_idx - sentences_before)
    end_idx = min(len(sentences), target_idx + sentences_after + 1)

    before_sents = sentences[start_idx:target_idx]
    after_sents = sentences[target_idx + 1 : end_idx]

    return {
        "citing_sentence": sentences[target_idx],
        "context_before": " ".join(before_sents),
        "context_after": " ".join(after_sents),
    }


