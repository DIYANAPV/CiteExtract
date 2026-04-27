"""Existence checking — L2 smart cascade.

For each reference, determines if it exists in scholarly databases:
  DOI present  →  CrossRef (DOI authority)  →  verify title match
  No DOI       →  Semantic Scholar (title)  →  OpenAlex  →  PubMed

All references are checked in parallel with asyncio.gather(),
respecting per-API rate limits via aiolimiter.
"""

import asyncio
import hashlib
import logging
import re
from typing import Optional

import httpx

log = logging.getLogger(__name__)

from src.models.reference import Reference
from src.models.verdict import ExistenceResult
from src import config as _config
from src.verification.api_clients import arxiv, crossref, openalex, openreview, pubmed, semantic_scholar
from src.verification.cache import APICache
from src.verification.matching import (
    PREPRINT_RE,
    is_title_match,
    normalize_title,
    title_similarity,
)


def _cache_key(reference: Reference) -> str:
    """Build a cache key scoped by both ref_id and title.

    Numeric ref_ids (from GROBID PDFs) collide across papers. Including a
    hash of the normalized title ensures different papers with the same
    ref_id get separate cache entries.
    """
    if reference.title:
        title_hash = hashlib.sha256(
            normalize_title(reference.title).encode()
        ).hexdigest()[:8]
        return f"existence:{reference.ref_id}:{title_hash}"
    return f"existence:{reference.ref_id}"


async def check_existence(
    reference: Reference,
    client: httpx.AsyncClient,
    cache: APICache,
) -> ExistenceResult:
    """Smart cascade for a single reference.

    Returns FOUND (with matched record + abstract) or NOT_FOUND.
    After finding a match, checks whether the DOI actually resolves
    to a live page (dead DOI detection).
    """
    result = await _check_existence_cascade(reference, client, cache)

    if result.status == "FOUND":
        # Dead DOI detection: if FOUND with a DOI, verify it resolves.
        doi = result.matched_doi or reference.doi
        if doi:
            resolves = await check_doi_resolves(doi, client)
            if resolves is False:
                result.flags.append(f"dead_doi: https://doi.org/{doi} does not resolve")

        # Author cross-validation: check if reference authors appear in a
        # second database. Catches a real LLM hallucination pattern where the
        # paper title is real but one or more co-authors are invented.
        if reference.authors and result.matched_authors:
            await _cross_validate_authors(
                reference, result, client,
            )

    return result


async def _check_existence_cascade(
    reference: Reference,
    client: httpx.AsyncClient,
    cache: APICache,
) -> ExistenceResult:
    """Internal cascade logic — called by check_existence()."""
    databases_checked: list[str] = []
    all_flags: list[str] = []

    # Check cache first — keyed by ref_id + title hash to avoid cross-paper collision
    cached = await cache.get(_cache_key(reference))
    if cached:
        return ExistenceResult(**cached)

    # Fuzzy cache lookup — if a very similar title was already resolved,
    # reuse that result instead of hitting APIs again. This handles minor
    # title variations (capitalization, punctuation) across references.
    if reference.title:
        norm_title = normalize_title(reference.title)

        # Fast path: exact normalized-title match (O(1) SQLite lookup)
        exact_hit = await cache.get_title_ref_id(norm_title)
        if exact_hit and exact_hit != reference.ref_id:
            cached_result = await cache.get(f"existence:{exact_hit}")
            if cached_result and cached_result.get("status") == "FOUND":
                result = ExistenceResult(
                    **{**cached_result, "ref_id": reference.ref_id,
                       "flags": cached_result.get("flags", []) + [
                           f"resolved_via_cache: matched cached ref {exact_hit} "
                           f"(exact_title_match)"
                       ]}
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

        # Slow path: fuzzy scan (only if exact match missed)
        title_entries = await cache.get_all_title_keys()
        for cached_title, cached_ref_id in title_entries:
            if cached_ref_id == reference.ref_id:
                continue
            sim = title_similarity(reference.title, cached_title)
            if sim >= 0.90:
                cached_result = await cache.get(f"existence:{cached_ref_id}")
                if cached_result and cached_result.get("status") == "FOUND":
                    result = ExistenceResult(
                        **{**cached_result, "ref_id": reference.ref_id,
                           "flags": cached_result.get("flags", []) + [
                               f"resolved_via_cache: matched cached ref {cached_ref_id} "
                               f"(title_sim={sim:.2f})"
                           ]}
                    )
                    await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                    return result

    # --- Step 0: Extract arXiv ID from raw_text/URL if not already set ---
    # Papers citing arXiv preprints often have the ID in the raw text,
    # URL, or DOI but GROBID may not extract it into the structured field.
    if not reference.arxiv_id:
        # Try raw_text: "arXiv:2501.12948" or "arXiv preprint arXiv:2501.12948"
        search_fields = [reference.raw_text or "", reference.url or "", reference.doi or ""]
        for field in search_fields:
            arxiv_match = re.search(
                r'(?:arXiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)',
                field, re.IGNORECASE,
            )
            if arxiv_match:
                reference = reference.model_copy(update={"arxiv_id": arxiv_match.group(1)})
                all_flags.append(f"arxiv_id_extracted: {reference.arxiv_id}")
                break

    # --- Step 1: If DOI present → CrossRef ---
    if reference.doi:
        db_record = await crossref.lookup_doi(reference.doi, client)
        databases_checked.append("crossref")

        if db_record:
            matched, sim, flags = is_title_match(reference.title, db_record["title"])
            all_flags.extend(flags)

            if matched:
                result = _build_found(
                    reference, db_record, "crossref", sim, databases_checked, all_flags
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result
            else:
                all_flags.append(
                    f"doi_title_mismatch: DOI {reference.doi} returned "
                    f"'{db_record['title'][:60]}' (sim={sim:.2f}), treating as no-DOI"
                )

    # --- Step 2: Semantic Scholar (by ID if available, else title search) ---
    s2_record = None
    if reference.doi or reference.arxiv_id:
        s2_id = f"DOI:{reference.doi}" if reference.doi else f"ARXIV:{reference.arxiv_id}"
        s2_record = await semantic_scholar.lookup_by_id(s2_id, client)
    if s2_record is None and reference.title:
        s2_record = await semantic_scholar.search_by_title(reference.title, client)
    databases_checked.append("semantic_scholar")

    if s2_record:
        matched, sim, flags = is_title_match(reference.title, s2_record["title"])
        all_flags.extend(flags)
        if matched:
            # If this is a preprint record but the reference cites a
            # conference/journal, try CrossRef for the published version
            # before returning. A human reviewer would do the same.
            upgraded = await _try_upgrade_preprint(
                s2_record, reference, client, databases_checked, all_flags
            )
            if upgraded:
                await _cache_result(cache, reference.ref_id, upgraded, _cache_key(reference))
                return upgraded
            result = _build_found(
                reference, s2_record, "semantic_scholar", sim, databases_checked, all_flags
            )
            await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
            return result

    # --- Step 3: OpenAlex ---
    if reference.title:
        oa_record = await openalex.search_by_title(reference.title, client)
        databases_checked.append("openalex")

        if oa_record:
            matched, sim, flags = is_title_match(reference.title, oa_record["title"])
            all_flags.extend(flags)
            if matched:
                upgraded = await _try_upgrade_preprint(
                    oa_record, reference, client, databases_checked, all_flags
                )
                if upgraded:
                    await _cache_result(cache, reference.ref_id, upgraded, _cache_key(reference))
                    return upgraded
                result = _build_found(
                    reference, oa_record, "openalex", sim, databases_checked, all_flags
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

    # --- Step 3.5: CrossRef title search ---
    # Only fires when CrossRef wasn't already consulted via DOI lookup
    # (Step 1 added it to ``databases_checked`` if so). Catches journal
    # articles that S2/OpenAlex returned without a DOI as well as papers
    # those DBs have weaker coverage on. CrossRef is also the canonical
    # source for retraction signals, so a hit here gives the verdict
    # better grounding even when the title was already matched elsewhere.
    if reference.title and "crossref" not in databases_checked:
        cr_record = await crossref.search_by_title(reference.title, client)
        databases_checked.append("crossref")
        if cr_record:
            matched, sim, flags = is_title_match(reference.title, cr_record["title"])
            all_flags.extend(flags)
            if matched:
                upgraded = await _try_upgrade_preprint(
                    cr_record, reference, client, databases_checked, all_flags,
                )
                if upgraded:
                    await _cache_result(cache, reference.ref_id, upgraded, _cache_key(reference))
                    return upgraded
                result = _build_found(
                    reference, cr_record, "crossref", sim, databases_checked, all_flags,
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

    # --- Step 4: PubMed (last resort) ---
    if reference.title:
        pm_record = await pubmed.search_by_title(reference.title, client)
        databases_checked.append("pubmed")

        if pm_record:
            matched, sim, flags = is_title_match(reference.title, pm_record["title"])
            all_flags.extend(flags)
            if matched:
                result = _build_found(
                    reference, pm_record, "pubmed", sim, databases_checked, all_flags
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

    # --- Step 5: arXiv (last scholarly source) ---
    # arXiv is free, no key needed, and catches recent preprints that
    # may not yet be indexed by S2 or OpenAlex.
    if reference.title:
        arxiv_record = await arxiv.search_by_title(reference.title, client)
        databases_checked.append("arxiv")

        if arxiv_record:
            matched, sim, flags = is_title_match(reference.title, arxiv_record["title"])
            all_flags.extend(flags)
            if matched:
                upgraded = await _try_upgrade_preprint(
                    arxiv_record, reference, client, databases_checked, all_flags
                )
                if upgraded:
                    await _cache_result(cache, reference.ref_id, upgraded, _cache_key(reference))
                    return upgraded
                result = _build_found(
                    reference, arxiv_record, "arxiv", sim, databases_checked, all_flags
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

    # --- Step 5a: arXiv author + year fallback (renamed papers) ---
    # Recovers papers whose arXiv title evolved across versions while the
    # author set stayed stable. Concrete case: ``2505.14376`` was uploaded
    # as "AutoRev: Automatic Peer Review System for Academic Research
    # Papers" (v1) and later renamed to "Graph-Guided Passage Retrieval
    # for Author-Centric Structured Feedback" (v3). The strict title
    # search above can't match the user's reference against a wholly
    # different current title, but author surnames + year still uniquely
    # identify the paper. Gated on having the title (we still hint with
    # it for verification) and ≥2 reference authors.
    if (
        reference.title
        and reference.authors
        and len({a for a in reference.authors if a and a.strip()}) >= 2
    ):
        ax_record = await arxiv.search_by_authors_year(
            reference.authors, reference.year, reference.title, client,
        )
        if ax_record:
            all_flags.append(
                "title_evolved: matched arXiv via authors+year — "
                "the paper's title appears to have been renamed across "
                "versions on arXiv"
            )
            matched, sim, flags = is_title_match(
                reference.title, ax_record["title"], threshold=0.0,
            )
            all_flags.extend(flags)
            result = _build_found(
                reference, ax_record, "arxiv", ax_record.get("title_similarity", sim),
                databases_checked, all_flags,
            )
            await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
            return result

    # --- Step 5b: OpenReview (experimental, flag-gated) ---
    # Closes the structural gap for tech reports that have no DOI and no
    # arXiv ID — LeCun's "A Path Towards Autonomous Machine Intelligence"
    # is the canonical example. Disabled by default to keep cascade
    # behaviour stable; flip ``experimental_fallbacks.openreview: true``
    # in config.yaml to opt in.
    if reference.title and _config.experimental_fallbacks().get("openreview"):
        or_record = await openreview.search_by_title(reference.title, client)
        databases_checked.append("openreview")
        if or_record:
            matched, sim, flags = is_title_match(reference.title, or_record["title"])
            all_flags.extend(flags)
            if matched:
                result = _build_found(
                    reference, or_record, "openreview", sim,
                    databases_checked, all_flags,
                )
                await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
                return result

    # --- Step 6: Web verification fallback ---
    # For non-scholarly sources (blogs, tech reports) or unknown refs that
    # weren't found in any scholarly DB, try to verify via URL/Wayback Machine.
    from src.verification.api_clients.web_verifier import (
        classify_source_type, verify_web_source,
    )
    source_type = classify_source_type(
        reference.title, reference.doi, reference.arxiv_id,
        reference.url, reference.raw_text,
    )
    if source_type in ("web", "unknown"):
        web_result = await verify_web_source(reference.url, reference.raw_text, client)
        databases_checked.append("web")
        if web_result:
            all_flags.append(f"verified_via_{web_result['method']}: {web_result['verified_url'][:80]}")
            result = ExistenceResult(
                ref_id=reference.ref_id,
                status="FOUND",
                source=web_result["source"],
                matched_title=reference.title,
                matched_authors=reference.authors,
                matched_year=reference.year,
                matched_venue=None,
                venue_aliases=[],
                matched_doi=reference.doi,
                title_similarity=1.0,
                databases_checked=databases_checked,
                flags=all_flags,
            )
            await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
            return result

    # --- All failed ---
    all_flags.append("not_found_in_any_database")
    result = ExistenceResult(
        ref_id=reference.ref_id,
        status="NOT_FOUND",
        databases_checked=databases_checked,
        flags=all_flags,
    )
    await _cache_result(cache, reference.ref_id, result, _cache_key(reference))
    return result


def _is_preprint_record(record: dict) -> bool:
    """Check if a DB record is from a preprint server."""
    venue = record.get("venue", "")
    return bool(PREPRINT_RE.search(venue))


async def _try_upgrade_preprint(
    preprint_record: dict,
    reference: Reference,
    client: httpx.AsyncClient,
    databases_checked: list[str],
    flags: list[str],
) -> Optional[ExistenceResult]:
    """If we found a preprint but the reference cites a conference/journal,
    try CrossRef to find the published version with correct year/venue.

    This is what a human reviewer would do: "I see arXiv 2021, but the
    reference says CVPR 2022 — let me check CrossRef for the published DOI."

    Returns the upgraded ExistenceResult if found, or None to keep the
    preprint result.
    """
    if not _is_preprint_record(preprint_record):
        return None  # Not a preprint — no upgrade needed

    # Only upgrade if the reference claims a non-preprint venue
    ref_venue = (reference.venue or "").strip()
    if not ref_venue or PREPRINT_RE.search(ref_venue):
        return None  # Reference itself says preprint — no mismatch

    # Try CrossRef with the DOI from the preprint record
    preprint_doi = preprint_record.get("doi", "")
    publisher_doi = None

    # arXiv DOIs (10.48550/arxiv.XXXX) aren't the publisher DOI.
    # Extract the arXiv ID and look for a publisher DOI via S2 externalIds.
    arxiv_id = preprint_record.get("arxiv_id")
    if not arxiv_id and preprint_doi:
        m = re.match(r'10\.48550/arxiv\.(.+)', preprint_doi, re.IGNORECASE)
        if m:
            arxiv_id = m.group(1)

    # If the preprint record has a non-arXiv DOI, use that directly
    if preprint_doi and not preprint_doi.startswith("10.48550"):
        publisher_doi = preprint_doi

    if publisher_doi:
        cr_record = await crossref.lookup_doi(publisher_doi, client)
        if "crossref" not in databases_checked:
            databases_checked.append("crossref")
        if cr_record:
            matched, sim, cr_flags = is_title_match(reference.title, cr_record["title"])
            if matched:
                flags.append(
                    f"upgraded_from_preprint: found published version via "
                    f"CrossRef DOI {publisher_doi}"
                )
                flags.extend(cr_flags)
                return _build_found(
                    reference, cr_record, "crossref", sim,
                    databases_checked, flags,
                )

    # No publisher DOI available — try OpenAlex which often has the
    # published version even when S2 only has the preprint
    if "openalex" not in databases_checked and reference.title:
        oa_record = await openalex.search_by_title(reference.title, client)
        databases_checked.append("openalex")
        if oa_record and not _is_preprint_record(oa_record):
            matched, sim, oa_flags = is_title_match(reference.title, oa_record["title"])
            if matched:
                # OpenAlex found a non-preprint version — check if it has a DOI
                oa_doi = oa_record.get("doi")
                if oa_doi and not oa_doi.startswith("10.48550"):
                    # Try CrossRef with this publisher DOI for best metadata
                    cr_record = await crossref.lookup_doi(oa_doi, client)
                    if "crossref" not in databases_checked:
                        databases_checked.append("crossref")
                    if cr_record:
                        cr_matched, cr_sim, cr_flags = is_title_match(
                            reference.title, cr_record["title"]
                        )
                        if cr_matched:
                            flags.append(
                                f"upgraded_from_preprint: found published version via "
                                f"OpenAlex → CrossRef DOI {oa_doi}"
                            )
                            flags.extend(cr_flags)
                            return _build_found(
                                reference, cr_record, "crossref", cr_sim,
                                databases_checked, flags,
                            )

                # Use the OpenAlex record directly
                flags.append("upgraded_from_preprint: found published version via OpenAlex")
                flags.extend(oa_flags)
                return _build_found(
                    reference, oa_record, "openalex", sim,
                    databases_checked, flags,
                )

    return None  # Could not find published version — keep preprint result


async def check_doi_resolves(doi: str, client: httpx.AsyncClient) -> bool | None:
    """Check whether a DOI actually resolves to a live page.

    Sends an HTTP HEAD to https://doi.org/{doi}. A DOI can be registered
    in CrossRef (metadata exists) but point to a dead URL (publisher offline,
    paper removed without formal retraction).

    Args:
        doi: The DOI to check (without https://doi.org/ prefix).
        client: httpx async client.

    Returns:
        True if DOI resolves (2xx/3xx), False if dead (4xx/5xx/timeout),
        None if the check itself failed (network error).
    """
    url = f"https://doi.org/{doi}"
    try:
        resp = await client.head(url, follow_redirects=True, timeout=10.0)
        if resp.status_code < 400:
            return True
        log.info(f"Dead DOI detected: {doi} returned HTTP {resp.status_code}")
        return False
    except Exception as e:
        log.debug(f"DOI resolution check failed for {doi}: {e}")
        return None


async def _cross_validate_authors(
    reference: Reference,
    result: ExistenceResult,
    client: httpx.AsyncClient,
) -> None:
    """Check reference authors against a second database for cross-validation.

    Catches a specific LLM hallucination pattern: the paper title is real
    but one or more co-authors are fabricated. If reference authors don't
    appear in ANY database, they're flagged as suspect.

    Modifies result.flags in-place.
    """
    from src.verification.matching import _author_tokens, _is_consortium_name

    ref_tokens = _author_tokens(reference.authors)
    primary_tokens = _author_tokens(result.matched_authors)
    if not ref_tokens:
        # Reference author list is empty after consortium / corporate-author
        # filtering. Llama 3 cites itself as authored by "Meta AI"; Qwen
        # papers credit "Qwen Team". Suppressing the validation here and
        # emitting a benign flag avoids the false-positive where
        # ``_cross_validate_authors`` previously marked these as fabricated.
        if reference.authors and all(
            _is_consortium_name(a) for a in reference.authors if a.strip()
        ):
            result.flags.append(
                "authors_corporate: ref author list is org-only "
                "(e.g., Meta AI, Google Research) — skipping cross-validation"
            )
        return

    # Pick a second DB that wasn't the primary source
    second_db_authors: list[str] = []
    source = result.source or ""
    try:
        if source != "openalex" and reference.title:
            oa_record = await openalex.search_by_title(reference.title, client)
            if oa_record:
                second_db_authors = oa_record.get("authors", [])
        elif source != "semantic_scholar" and reference.title:
            s2_record = await semantic_scholar.search_by_title(reference.title, client)
            if s2_record:
                second_db_authors = s2_record.get("authors", [])
    except Exception as e:
        log.debug(f"Author cross-validation secondary lookup failed: {e}")
        return

    if not second_db_authors:
        return

    second_tokens = _author_tokens(second_db_authors)

    # Authors confirmed across both databases
    confirmed = ref_tokens & primary_tokens & second_tokens
    # Authors in reference but in NEITHER database
    suspect = ref_tokens - primary_tokens - second_tokens

    if confirmed:
        result.flags.append(
            f"authors_cross_validated: {len(confirmed)} author(s) confirmed "
            f"across {source} + {'openalex' if source != 'openalex' else 'semantic_scholar'}"
        )

    if suspect:
        suspect_names = []
        for author in reference.authors:
            from src.verification.matching import normalize_author
            parts = author.split(",")[0].split()
            surname = parts[-1] if parts else author
            if normalize_author(surname) in suspect:
                suspect_names.append(author)
        if suspect_names:
            result.flags.append(
                f"suspect_authors: {', '.join(suspect_names)} — not found in any database. "
                f"Possibly fabricated."
            )


def _build_found(
    reference: Reference,
    db_record: dict,
    source: str,
    sim: float,
    databases_checked: list[str],
    flags: list[str],
) -> ExistenceResult:
    """Build a FOUND ExistenceResult from a DB record."""
    retraction_status = db_record.get("retraction_status")
    # Only CrossRef provides retraction data. Flag the gap for other sources.
    if retraction_status is None and source != "crossref":
        flags.append(f"retraction_not_checked: found via {source} (no retraction data)")

    # Extract open-access URL if available (S2 openAccessPdf, OA landing page)
    oa_url = None
    oa_pdf = db_record.get("openAccessPdf")
    if isinstance(oa_pdf, dict):
        oa_url = oa_pdf.get("url")
    if not oa_url:
        oa_url = db_record.get("oa_url")

    return ExistenceResult(
        ref_id=reference.ref_id,
        status="FOUND",
        source=source,
        matched_title=db_record.get("title"),
        matched_authors=db_record.get("authors", []),
        matched_year=db_record.get("year"),
        matched_venue=db_record.get("venue", ""),
        venue_aliases=db_record.get("venue_aliases", []),
        matched_doi=db_record.get("doi"),
        matched_arxiv_id=db_record.get("arxiv_id"),
        abstract=db_record.get("abstract"),
        oa_url=oa_url,
        retraction_status=retraction_status,
        title_similarity=sim,
        databases_checked=databases_checked,
        flags=flags,
    )


async def _cache_result(cache: APICache, ref_id: str, result: ExistenceResult,
                        cache_key: str = "") -> None:
    """Cache existence result, abstract, and title index."""
    from src.verification.cache import TTL_METADATA, TTL_NOT_FOUND

    ttl = TTL_METADATA if result.status == "FOUND" else TTL_NOT_FOUND
    key = cache_key or f"existence:{ref_id}"
    await cache.set(key, result.model_dump(), ttl)

    if result.abstract:
        await cache.set_abstract(ref_id, result.abstract)

    # Index by normalized title for fuzzy cache lookups on future references
    if result.status == "FOUND" and result.matched_title:
        norm = normalize_title(result.matched_title)
        if norm:
            await cache.set_title_index(norm, ref_id)


async def check_all_references(
    references: list[Reference],
    cache: Optional[APICache] = None,
    retry_failed: bool = False,
) -> list[ExistenceResult]:
    """Check all references in parallel, respecting rate limits.

    Each reference runs its own cascade sequentially, but multiple
    references execute concurrently via asyncio.gather().

    Args:
        references: References to check.
        cache: Optional shared cache. If None, creates and closes its own.
    """
    if not references:
        return []

    own_cache = cache is None
    if own_cache:
        cache = APICache()
    if retry_failed:
        cleared = await cache.clear_not_found()
        if cleared:
            log.info(f"Cleared {cleared} NOT_FOUND cache entries for retry")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            total = len(references)
            completed = 0

            async def _check_one(ref: Reference) -> ExistenceResult:
                nonlocal completed
                result = await check_existence(ref, client, cache)
                completed += 1
                if total > 5 and completed % max(1, total // 5) == 0:
                    log.info(f"Existence check progress: {completed}/{total}")
                return result

            log.info(f"Checking {total} references across scholarly databases...")
            tasks = [_check_one(ref) for ref in references]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to NOT_FOUND results so one failure
        # doesn't crash the whole pipeline.
        results: list[ExistenceResult] = []
        for ref, result in zip(references, raw_results):
            if isinstance(result, Exception):
                log.warning(f"Existence check failed for {ref.ref_id}: {result}")
                results.append(ExistenceResult(
                    ref_id=ref.ref_id,
                    status="NOT_FOUND",
                    databases_checked=[],
                    flags=[f"existence_check_error: {type(result).__name__}: {result}"],
                ))
            else:
                results.append(result)

        found = sum(1 for r in results if r.status == "FOUND")
        log.info(f"Existence check complete: {found}/{len(results)} references found")

        return results
    finally:
        if own_cache:
            await cache.close()
