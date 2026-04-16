"""Tool definitions and executor for the agentic citation verification agent.

Wraps existing API clients (CrossRef, Semantic Scholar, OpenAlex, PubMed)
and matching utilities as OpenAI function-calling tools.
"""

import json
import logging
from typing import Any

import httpx

from src.verification.api_clients import crossref, semantic_scholar, openalex, pubmed
from src.verification.api_clients.rate_limiter import acquire
from src.verification.cache import APICache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI function-calling tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_crossref_by_doi",
            "description": (
                "Look up a paper by its DOI in CrossRef. Returns title, authors, "
                "year, venue, DOI, and retraction_status. Best for DOI-based lookups. "
                "Also the only source that provides retraction information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "The DOI to look up (e.g. '10.1145/3292500.3330672').",
                    }
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": (
                "Search Semantic Scholar by title or look up by ID. Returns title, "
                "authors, year, venue, abstract, DOI, and arXiv ID. Best for CS/AI "
                "papers. Use search_type='id' with formats like 'DOI:10.xxx' or "
                "'ARXIV:2301.xxx' for precise lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Title text for search, or an identifier for ID lookup "
                            "(e.g. 'DOI:10.xxx', 'ARXIV:2301.12345')."
                        ),
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["title", "id"],
                        "description": "Whether to search by title or look up by ID.",
                    },
                },
                "required": ["query", "search_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_openalex",
            "description": (
                "Search OpenAlex by title. Broadest coverage (250M+ works). Returns "
                "title, authors, year, venue, abstract, and DOI. Good fallback when "
                "Semantic Scholar misses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The paper title to search for.",
                    }
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": (
                "Search PubMed by title. Covers 36M+ biomedical articles. Returns "
                "title, authors, year, venue, abstract, DOI, and PMID. Use for "
                "biomedical and life sciences papers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The paper title to search for.",
                    }
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_titles",
            "description": (
                "Compare two titles with word-level diff. Returns whether they are "
                "an exact match (after normalization) and lists specific word-level "
                "differences. Do NOT rely solely on the similarity score — read the "
                "differences list to understand what exactly differs. If titles differ, "
                "investigate whether the difference is meaningful (e.g., a different "
                "paper) or cosmetic (e.g., subtitle truncation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_title": {
                        "type": "string",
                        "description": "The reference title from the bibliography.",
                    },
                    "db_title": {
                        "type": "string",
                        "description": "The title found in the database.",
                    },
                },
                "required": ["ref_title", "db_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_authors",
            "description": (
                "Compare reference authors against database authors using canonical "
                "name matching. Normalizes names across citation formats (Vancouver "
                "'Huang H', APA 'Huang, H.', IEEE 'H. Huang' all match 'Hai Huang'). "
                "Returns per-author match status showing which authors matched and "
                "which did not. Also checks if truncation is expected for the detected "
                "citation format. Unmatched authors are a potential fabrication signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Author list from the reference.",
                    },
                    "db_authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Author list from the database.",
                    },
                    "citation_format": {
                        "type": "string",
                        "description": (
                            "The detected citation format (apa, vancouver, ieee, etc.). "
                            "Use the value from the reference metadata if available."
                        ),
                    },
                },
                "required": ["ref_authors", "db_authors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_web_source",
            "description": (
                "Verify that a non-scholarly web source (blog post, tech report, "
                "website) exists by checking if the URL resolves or is archived "
                "in the Wayback Machine. Use this when scholarly database searches "
                "fail AND the reference appears to be a blog, website, or informal "
                "source. Returns verification info if the URL is live or archived."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "The URL to check (e.g. 'https://blog.example.com/post'). "
                            "Extract from the reference's raw text if available."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_cited_paper_passages",
            "description": (
                "Fetch the full text of a cited paper and retrieve the most "
                "relevant passages for a given citing sentence. Use this AFTER "
                "you have found the paper in a database (need at least a DOI, "
                "arXiv ID, or open-access URL). Returns the top passages from "
                "the cited paper that are most relevant to the claim being made. "
                "Use these passages to verify whether the citation accurately "
                "represents the cited paper's content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "citing_sentence": {
                        "type": "string",
                        "description": "The citing sentence or claim to check against the paper.",
                    },
                    "paper_doi": {
                        "type": "string",
                        "description": "DOI of the cited paper (optional, but preferred).",
                    },
                    "paper_title": {
                        "type": "string",
                        "description": "Title of the cited paper.",
                    },
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv ID of the cited paper (optional).",
                    },
                    "oa_url": {
                        "type": "string",
                        "description": "Open-access URL for the paper (optional).",
                    },
                },
                "required": ["citing_sentence", "paper_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_current_paper_context",
            "description": (
                "Search the CURRENT paper (the one containing the citation) "
                "for more context about what the authors actually did. Use "
                "this when the citing sentence is vague — e.g. 'We followed "
                "the approach of [5]' — and you need to understand what was "
                "actually done before you can judge whether the citation is "
                "accurate. Returns relevant passages from the current paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to search for in the current paper. "
                            "E.g. 'training protocol details', 'model architecture', "
                            "'experimental setup', 'dosage regimen'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google_scholar",
            "description": (
                "Search Google Scholar for a paper by title. Returns what "
                "people see on Google Scholar — title, authors, year, venue. "
                "Useful when other databases return a different version of "
                "the paper (e.g. the published journal version when the "
                "reference cites the arXiv preprint). Google Scholar often "
                "shows the version that people actually cite from."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Paper title to search for.",
                    },
                },
                "required": ["title"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor — dispatches tool calls to existing API clients
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Executes tool calls by routing to existing API clients."""

    def __init__(self, http_client: httpx.AsyncClient, cache: APICache) -> None:
        self._client = http_client
        self._cache = cache
        self._current_paper_chunks: list | None = None

    def set_current_paper_chunks(self, chunks: list) -> None:
        """Set the current paper's chunks for retrieve_current_paper_context."""
        self._current_paper_chunks = chunks

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return its result as a JSON-serializable dict.

        On failure, returns {"error": "description"} so the agent can reason about it.
        """
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return {"error": f"Unknown tool: {tool_name}"}
            return await handler(arguments)
        except Exception as e:
            log.warning(f"Tool {tool_name} failed: {e}")
            return {"error": f"{tool_name} failed: {str(e)[:200]}"}

    # -- Individual tool handlers --

    async def _tool_search_crossref_by_doi(self, args: dict) -> dict:
        doi = args["doi"]
        result = await crossref.lookup_doi(doi, self._client)
        if result is None:
            return {"error": "not_found", "message": f"DOI '{doi}' not found in CrossRef."}
        return result

    async def _tool_search_semantic_scholar(self, args: dict) -> dict:
        query = args["query"]
        search_type = args.get("search_type", "title")

        if search_type == "id":
            result = await semantic_scholar.lookup_by_id(query, self._client)
        else:
            result = await semantic_scholar.search_by_title(query, self._client)

        if result is None:
            return {"error": "not_found", "message": f"No match for '{query}' in Semantic Scholar."}
        return result

    async def _tool_search_openalex(self, args: dict) -> dict:
        title = args["title"]
        result = await openalex.search_by_title(title, self._client)
        if result is None:
            return {"error": "not_found", "message": f"No match for '{title}' in OpenAlex."}
        return result

    async def _tool_search_pubmed(self, args: dict) -> dict:
        title = args["title"]
        result = await pubmed.search_by_title(title, self._client)
        if result is None:
            return {"error": "not_found", "message": f"No match for '{title}' in PubMed."}
        return result

    async def _tool_compare_titles(self, args: dict) -> dict:
        from src.verification.title_comparison import compare_titles
        ref_title = args["ref_title"]
        db_title = args["db_title"]
        result = compare_titles(ref_title, db_title)
        return result.to_dict()

    async def _tool_compare_authors(self, args: dict) -> dict:
        from src.verification.author_comparison import compare_authors
        ref_authors = args["ref_authors"]
        db_authors = args["db_authors"]
        citation_format = args.get("citation_format")
        result = compare_authors(ref_authors, db_authors, citation_format)
        return result.to_dict()

    async def _tool_verify_web_source(self, args: dict) -> dict:
        from src.verification.api_clients.web_verifier import verify_web_source
        url = args["url"]
        result = await verify_web_source(url, "", self._client)
        if result is None:
            return {"error": "not_found", "message": f"URL '{url}' is not live and not archived."}
        return result

    async def _tool_retrieve_cited_paper_passages(self, args: dict) -> dict:
        from src.models.verdict import ExistenceResult
        from src.verification.api_clients.fulltext import get_full_text
        from src.verification.comprehension import chunk_text, retrieve_passages_hybrid
        from src import config

        citing_sentence = args["citing_sentence"]
        doi = args.get("paper_doi")
        title = args.get("paper_title", "")
        arxiv_id = args.get("arxiv_id")
        oa_url = args.get("oa_url")

        # Build a minimal ExistenceResult for fulltext fetcher
        exist = ExistenceResult(
            ref_id="agent_lookup",
            status="FOUND",
            source="agentic",
            matched_title=title,
            matched_doi=doi,
            matched_arxiv_id=arxiv_id,
            oa_url=oa_url,
            databases_checked=[],
        )

        ft = await get_full_text(exist, self._client, self._cache)
        text = ft.full_text or ft.abstract or ""
        if not text.strip():
            return {
                "error": "no_text",
                "message": f"Could not retrieve full text for '{title}'.",
                "source": ft.source,
            }

        sections = ft.sections if ft.sections else None
        chunks = chunk_text(text, sections=sections)
        if not chunks:
            return {"error": "no_chunks", "message": "Text could not be chunked."}

        comp_cfg = config.comprehension()
        top_k = comp_cfg.get("top_k", 3)
        dense_model = comp_cfg.get("dense_model")
        if dense_model:
            passages = retrieve_passages_hybrid(
                citing_sentence, chunks, top_k=top_k,
                model_name=dense_model,
                bm25_candidates=comp_cfg.get("bm25_candidates", 10),
                dense_candidates=comp_cfg.get("dense_candidates", 10),
                rrf_k=comp_cfg.get("rrf_k", 60),
            )
        else:
            from src.verification.comprehension import retrieve_passages_bm25
            passages = retrieve_passages_bm25(citing_sentence, chunks, top_k=top_k)

        result_passages = []
        for sc in passages:
            score = sc.rrf_score or sc.dense_score or sc.bm25_score
            result_passages.append({
                "text": sc.chunk.text,
                "section": sc.chunk.section_name,
                "score": round(score, 4),
            })

        return {
            "source": ft.source,
            "full_text_available": bool(ft.full_text),
            "passages": result_passages,
        }

    async def _tool_retrieve_current_paper_context(self, args: dict) -> dict:
        from src.verification.comprehension import retrieve_passages_bm25

        query = args["query"]

        if not self._current_paper_chunks:
            return {
                "error": "no_text",
                "message": "Current paper text not available (input was a bibliography file).",
            }

        passages = retrieve_passages_bm25(query, self._current_paper_chunks, top_k=3)

        result_passages = []
        for sc in passages:
            result_passages.append({
                "text": sc.chunk.text,
                "section": sc.chunk.section_name,
                "score": round(sc.bm25_score, 4),
            })

        return {
            "source": "current_paper",
            "passages": result_passages,
        }

    async def _tool_search_google_scholar(self, args: dict) -> dict:
        from src.verification.api_clients.google_scholar import search_google_scholar
        from src import config as app_cfg

        title = args["title"]
        api_key = app_cfg.serpapi_key()
        if not api_key:
            return {"error": "not_configured", "message": "SERPAPI_KEY not set in .env"}

        result = await search_google_scholar(title, self._client, api_key)
        if result is None:
            return {"error": "not_found", "message": f"No match for '{title}' on Google Scholar."}
        return result
