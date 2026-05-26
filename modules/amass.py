"""Amass API client: BiomedCore literature + TrialCore linkage with a fallback query chain."""

from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st


BASE_URL = "https://api.amass.tech/api/v1"
BIOMEDCORE = f"{BASE_URL}/cores/biomedcore/records"
TRIALCORE_GET = f"{BASE_URL}/cores/trialcore/records"


def _api_key() -> Optional[str]:
    try:
        key = st.secrets.get("AMASS_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("AMASS_API_KEY")


def _headers() -> dict:
    key = _api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _extract_records(payload) -> list[dict]:
    """Tolerant of {data: [...]}, {results: [...]}, or a bare list."""
    if isinstance(payload, dict):
        for key in ("data", "results", "records", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    if isinstance(payload, list):
        return payload
    return []


def _do_search(query: str, k: int, min_jufo: Optional[int]) -> list[dict]:
    params = {
        "query": query,
        "limit": k,
        "include": "referencesTrialCore",
    }
    if min_jufo is not None:
        params["minJournalQualityJufo"] = min_jufo
    try:
        r = requests.get(BIOMEDCORE, params=params, headers=_headers(), timeout=15)
    except requests.RequestException as e:
        st.warning(f"Amass request failed: {e}")
        return []
    if r.status_code >= 400:
        # Surface but don't crash — the rest of the app should still work.
        st.warning(f"Amass returned {r.status_code} for query '{query}'")
        return []
    try:
        return _extract_records(r.json())
    except ValueError:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_region_literature(
    region_name: str,
    drug: str = "semaglutide",
    k: int = 5,
) -> dict:
    """Run the fallback query chain. Returns {'papers', 'query_used', 'broadened'}.

    Chain:
      1. "{region_name} {drug} OR GLP-1"  + JUFO ≥ 2
      2. "{region_name} GLP-1"            + JUFO ≥ 2
      3. "{region_name} c-fos"            + JUFO ≥ 2
      4. "{region_name}"                  (no JUFO filter)
    """
    chain = [
        (f"{region_name} {drug} OR GLP-1", 2, False),
        (f"{region_name} GLP-1",           2, True),
        (f"{region_name} c-fos",           2, True),
        (region_name,                      None, True),
    ]
    for query, jufo, broadened in chain:
        papers = _do_search(query, k, jufo)
        if papers:
            return {
                "papers": papers,
                "query_used": query,
                "broadened": broadened,
            }
    return {"papers": [], "query_used": chain[0][0], "broadened": False}


def _paper_trial_ids(paper: dict) -> list[str]:
    """Collect AMTC_ trial IDs from a paper's referencesTrialCore field, tolerantly."""
    ids: list[str] = []
    refs = (
        paper.get("referencesTrialCore")
        or paper.get("references_trialcore")
        or paper.get("trial_references")
        or []
    )
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str):
                ids.append(ref)
            elif isinstance(ref, dict):
                rid = ref.get("id") or ref.get("amass_id") or ref.get("amassId")
                if rid:
                    ids.append(rid)
    seen, uniq = set(), []
    for rid in ids:
        if rid and rid not in seen:
            seen.add(rid)
            uniq.append(rid)
    return uniq


@st.cache_data(ttl=3600, show_spinner=False)
def linked_trials(papers: list[dict], max_trials: int = 3) -> list[dict]:
    """Resolve TrialCore IDs across papers, dedupe, and return up to max_trials."""
    ids: list[str] = []
    for p in papers:
        for tid in _paper_trial_ids(p):
            if tid not in ids:
                ids.append(tid)

    out: list[dict] = []
    for tid in ids[: max_trials * 3]:  # over-fetch then trim
        try:
            r = requests.get(f"{TRIALCORE_GET}/{tid}", headers=_headers(), timeout=15)
            if r.status_code >= 400:
                continue
            payload = r.json()
            trial = payload.get("data", payload) if isinstance(payload, dict) else payload
            if isinstance(trial, dict):
                out.append(trial)
        except (requests.RequestException, ValueError):
            continue
        if len(out) >= max_trials:
            break
    return out


def paper_display(paper: dict) -> dict:
    """Normalize a BiomedCore record to a small display dict."""
    title = paper.get("title") or paper.get("article_title") or "(untitled)"
    journal = paper.get("journal") or paper.get("journal_name") or ""
    year = paper.get("year") or paper.get("publication_year") or paper.get("publishedYear") or ""
    citations = paper.get("citationCount") or paper.get("citation_count") or paper.get("citations")
    jufo = paper.get("journalQualityJufo") or paper.get("jufo") or paper.get("journal_quality_jufo")
    doi = paper.get("doi")
    authors = paper.get("authors") or paper.get("author_list") or []
    if isinstance(authors, list):
        author_str = ", ".join(
            (a.get("name") if isinstance(a, dict) else str(a)) for a in authors[:3]
        )
        if len(authors) > 3:
            author_str += " et al."
    else:
        author_str = str(authors)
    abstract = paper.get("abstract") or paper.get("abstract_text") or ""
    url = paper.get("url") or (f"https://doi.org/{doi}" if doi else "")
    return {
        "title": title,
        "journal": journal,
        "year": year,
        "citations": citations,
        "jufo": jufo,
        "authors": author_str,
        "abstract": abstract,
        "url": url,
        "doi": doi,
    }


def trial_display(trial: dict) -> dict:
    """Normalize a TrialCore record for display."""
    return {
        "id": trial.get("id") or trial.get("amass_id") or trial.get("amassId"),
        "title": trial.get("title") or trial.get("brief_title") or "(untitled)",
        "phase": trial.get("phase") or trial.get("study_phase") or "",
        "status": trial.get("status") or trial.get("overall_status") or "",
        "conditions": trial.get("conditions") or trial.get("condition") or [],
        "sponsor": trial.get("sponsor") or trial.get("lead_sponsor") or "",
        "nct": trial.get("nct_id") or trial.get("nctId") or "",
    }
