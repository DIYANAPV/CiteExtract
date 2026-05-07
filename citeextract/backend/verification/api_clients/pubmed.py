
import re
import xml.etree.ElementTree as ET
from typing import Optional

from defusedxml.ElementTree import fromstring as safe_fromstring

import httpx

from citeextract import config
from citeextract.verification.api_clients.rate_limiter import fetch_with_retry
from citeextract.verification.matching import title_similarity

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def search_by_title(
    title: str, client: httpx.AsyncClient
) -> Optional[dict]:
    if not title or len(title.strip()) < 5:
        return None

    cfg = config.api("pubmed")
    timeout = cfg.get("timeout", 15)
    results_per_page = cfg.get("results_per_page", 5)

    query = re.sub(r"[^\w\s]", " ", title).strip()
    try:
        resp = await fetch_with_retry(
            client, "pubmed", "GET", ESEARCH_URL,
            params={
                "db": "pubmed",
                "term": f'{query}[Title]',
                "retmode": "json",
                "retmax": results_per_page,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    search_data = resp.json().get("esearchresult", {})
    pmids = search_data.get("idlist", [])
    if not pmids:
        return None

    try:
        resp = await fetch_with_retry(
            client, "pubmed", "GET", EFETCH_URL,
            params={
                "db": "pubmed",
                "id": ",".join(pmids[:3]),
                "retmode": "xml",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError):
        return None

    return _parse_and_match(resp.text, title)


def _parse_and_match(xml_text: str, query_title: str) -> Optional[dict]:
    try:
        root = safe_fromstring(xml_text)
    except ET.ParseError:
        return None

    best: Optional[dict] = None
    best_sim = 0.0

    for article in root.findall(".//PubmedArticle"):
        parsed = _parse_article(article)
        if not parsed:
            continue
        sim = title_similarity(query_title, parsed["title"])
        if sim > best_sim:
            best_sim = sim
            best = parsed
            best["title_similarity"] = sim

    if best and best_sim >= config.thresholds()["title_match"]:
        return best
    return None


def _parse_article(article: ET.Element) -> Optional[dict]:
    medline = article.find(".//MedlineCitation")
    if medline is None:
        return None

    title_elem = medline.find(".//ArticleTitle")
    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
    if not title:
        return None

    authors: list[str] = []
    for author in medline.findall(".//Author"):
        last = author.findtext("LastName", "")
        first = author.findtext("ForeName", "") or author.findtext("Initials", "")
        name = f"{first} {last}".strip()
        if name:
            authors.append(name)

    year: Optional[int] = None
    year_elem = medline.find(".//PubDate/Year")
    if year_elem is not None and year_elem.text:
        try:
            year = int(year_elem.text)
        except ValueError:
            pass
    if year is None:
        medline_date = medline.findtext(".//PubDate/MedlineDate", "")
        year_match = re.search(r"(\d{4})", medline_date)
        if year_match:
            year = int(year_match.group(1))

    venue = medline.findtext(".//Journal/Title", "")

    pmid = medline.findtext("PMID", "")

    abstract_parts = []
    for text_elem in medline.findall(".//AbstractText"):
        if text_elem.text:
            abstract_parts.append(text_elem.text.strip())
    abstract = " ".join(abstract_parts) if abstract_parts else None

    doi: Optional[str] = None
    for eid in article.findall(".//ArticleIdList/ArticleId"):
        if eid.get("IdType") == "doi" and eid.text:
            doi = eid.text.strip()

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "abstract": abstract,
        "doi": doi,
        "pmid": pmid,
    }
