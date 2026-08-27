"""No-key public literature/database/open-access lookup tools.

This adapter completes the long-standing ``tool_server:literature_lookup``
contract without introducing new credentials. It intentionally stays small:
search a few public sources, normalize records, and degrade per source when a
remote endpoint is unavailable. External literature is contextual evidence for
research planning, not task evaluation truth.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from ipaddress import ip_address
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from praxist.plugins.tools.arxiv import adapter as arxiv_adapter

try:  # claude_agent_sdk is only required when the MCP server is spun up.
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        create_sdk_mcp_server,
        tool,
    )
except ImportError:  # pragma: no cover - SDK missing in some test envs
    tool = None
    create_sdk_mcp_server = None


_SCHEMA_VERSION = 1
_DEFAULT_MAX_RESULTS = 10
_MAX_RESULTS_CAP = 25
_DEFAULT_SOURCES = ("arxiv", "openalex", "pubmed", "crossref")
_SOURCE_ALIASES = {
    "arxiv": "arxiv",
    "arxiv.org": "arxiv",
    "openalex": "openalex",
    "pubmed": "pubmed",
    "ncbi": "pubmed",
    "crossref": "crossref",
    "semantic": "semantic_scholar",
    "semantic_scholar": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "europepmc": "europepmc",
    "europe_pmc": "europepmc",
    "pmc": "europepmc",
    "uniprot": "uniprot",
    "protein": "uniprot",
    "clinicaltrials": "clinicaltrials",
    "clinical_trials": "clinicaltrials",
    "clinicaltrials.gov": "clinicaltrials",
}
_OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_CROSSREF_WORK_URL = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_CLINICALTRIALS_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
_DEFAULT_TIMEOUT_SECONDS = 15.0
_DEFAULT_MAX_TEXT_CHARS = 12000
_MAX_TEXT_CHARS_CAP = 50000
_MAX_FULL_TEXT_BYTES = 8 * 1024 * 1024
_DIRECT_OPEN_ACCESS_HOST_SUFFIXES = (
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "pmc.ncbi.nlm.nih.gov",
    "europepmc.org",
    "plos.org",
    "frontiersin.org",
    "mdpi.com",
    "elifesciences.org",
    "springeropen.com",
    "biomedcentral.com",
)
_CURRENT_ENVIRONMENT_RESOURCE_POLICY = (
    "Current-environment only: if a source mentions a dataset, checkpoint, "
    "simulator, package, license, API, or runtime environment that is not already "
    "available in the task project/runtime, do not download, install, provision, "
    "or treat it as available during the Praxist run. Extract reusable ideas from "
    "the source and adapt them to the existing local data, simulator, "
    "dependencies, evaluator, and hardware. Record missing resources only as "
    "task-local notes for the user."
)

__all__ = [
    "create_literature_lookup_server",
    "create_tool_plugin",
    "handle_literature_resolve",
    "handle_literature_search",
    "handle_literature_open_access_text",
    "handle_scientific_database_search",
    "literature_open_access_text",
    "handle_literature_source_guide",
    "literature_resolve",
    "literature_search",
    "literature_source_guide",
    "scientific_database_search",
]


def literature_search(
    query: str,
    sources: str | list[str] | tuple[str, ...] | None = None,
    max_results: int = _DEFAULT_MAX_RESULTS,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Search public literature sources and return normalized records."""
    if not isinstance(query, str) or not query.strip():
        return _with_resource_policy(
            {"schema_version": _SCHEMA_VERSION, "error": "query is required"}
        )

    capped_max = _coerce_max_results(max_results)
    selected_sources = _normalize_sources(sources)
    per_source = max(1, capped_max)
    records_by_source: list[list[dict[str, Any]]] = []
    warnings: list[str] = []

    for source in selected_sources:
        try:
            if source == "arxiv":
                payload = _search_arxiv(
                    query.strip(),
                    per_source,
                    http_client_factory=http_client_factory,
                    clock=clock,
                )
            elif source == "openalex":
                payload = _search_openalex(
                    query.strip(),
                    per_source,
                    http_client_factory=http_client_factory,
                )
            elif source == "pubmed":
                payload = _search_pubmed(
                    query.strip(),
                    per_source,
                    http_client_factory=http_client_factory,
                )
            elif source == "crossref":
                payload = _search_crossref(
                    query.strip(),
                    per_source,
                    http_client_factory=http_client_factory,
                )
            elif source == "semantic_scholar":
                payload = _search_semantic_scholar(
                    query.strip(),
                    per_source,
                    http_client_factory=http_client_factory,
                )
            else:
                warnings.append(f"unknown source ignored: {source}")
                continue
        except Exception as exc:  # noqa: BLE001 - remote lookup must degrade.
            warnings.append(f"{source}: {type(exc).__name__}: {exc}")
            continue

        if "error" in payload:
            warnings.append(f"{source}: {payload['error']}")
            continue
        records_by_source.append(list(payload.get("records") or []))
        warnings.extend(str(item) for item in payload.get("warnings") or [])

    records = _round_robin_records(records_by_source, capped_max)
    return _with_resource_policy(
        {
            "schema_version": _SCHEMA_VERSION,
            "query": query.strip(),
            "sources": selected_sources,
            "result_count": len(records),
            "results": records,
            "warnings": warnings,
            "evidence_role": "contextual_literature_signal",
            "promotion_note": (
                "Literature records may inform hypotheses and research directions; "
                "they are not task evaluation facts."
            ),
        }
    )


def scientific_database_search(
    query: str,
    sources: str | list[str] | tuple[str, ...] | None = None,
    max_results: int = _DEFAULT_MAX_RESULTS,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    """Search no-key public scientific databases beyond paper indexes.

    This is deliberately read-only and compact. It is meant to help task
    initialization and literature-scout roles find authoritative entity,
    trial, and biomedical records without adding API-key requirements.
    """
    if not isinstance(query, str) or not query.strip():
        return _with_resource_policy(
            {"schema_version": _SCHEMA_VERSION, "error": "query is required"}
        )

    selected_sources = (
        _normalize_sources(sources)
        if sources
        else [
            "europepmc",
            "uniprot",
            "clinicaltrials",
        ]
    )
    capped_max = _coerce_max_results(max_results)
    records_by_source: list[list[dict[str, Any]]] = []
    warnings: list[str] = []
    for source in selected_sources:
        try:
            if source == "europepmc":
                payload = _search_europepmc(
                    query.strip(),
                    capped_max,
                    http_client_factory=http_client_factory,
                )
            elif source == "uniprot":
                payload = _search_uniprot(
                    query.strip(),
                    capped_max,
                    http_client_factory=http_client_factory,
                )
            elif source == "clinicaltrials":
                payload = _search_clinicaltrials(
                    query.strip(),
                    capped_max,
                    http_client_factory=http_client_factory,
                )
            else:
                warnings.append(f"source is not a scientific database connector: {source}")
                continue
        except Exception as exc:  # noqa: BLE001 - remote lookup must degrade.
            warnings.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        if "error" in payload:
            warnings.append(f"{source}: {payload['error']}")
            continue
        records_by_source.append(list(payload.get("records") or []))
        warnings.extend(str(item) for item in payload.get("warnings") or [])

    records = _round_robin_records(records_by_source, capped_max)
    return _with_resource_policy(
        {
            "schema_version": _SCHEMA_VERSION,
            "query": query.strip(),
            "sources": selected_sources,
            "result_count": len(records),
            "results": records,
            "warnings": warnings,
            "evidence_role": "contextual_database_signal",
            "promotion_note": (
                "Database records may inform task context and hypotheses; they are "
                "not task evaluation facts."
            ),
        }
    )


def literature_open_access_text(
    identifier_or_url: str,
    max_chars: int = _DEFAULT_MAX_TEXT_CHARS,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Fetch open-access text or PDF bytes metadata with provenance.

    The function never tries to bypass paywalls. DOI/arXiv/OpenAlex/PubMed
    identifiers are first resolved to public metadata, then the best open URL
    is fetched when one is available. PDF responses are recorded with content
    hash and size; OCR/text extraction remains delegated to ``pdf_reader``.
    """
    if not isinstance(identifier_or_url, str) or not identifier_or_url.strip():
        return _with_resource_policy(
            {"schema_version": _SCHEMA_VERSION, "error": "identifier_or_url is required"}
        )

    value = identifier_or_url.strip()
    max_text_chars = _coerce_max_text_chars(max_chars)
    resolution: dict[str, Any] | None = None
    resolve_value = _identifier_from_url(value) or value
    target_url = (
        value if value.startswith(("http://", "https://")) and resolve_value == value else ""
    )
    warnings: list[str] = []
    if target_url and not _is_allowed_open_access_url(target_url):
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier_or_url": value,
                "error": (
                    "direct URL fetch is limited to known public scientific/open-access hosts; "
                    "use tool_server:browser for general web pages"
                ),
            }
        )

    if not target_url:
        doi = _extract_doi(resolve_value)
        resolution = literature_resolve(
            resolve_value,
            http_client_factory=http_client_factory,
            clock=clock,
        )
        if "error" in resolution:
            if not doi:
                return _with_resource_policy(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "identifier_or_url": value,
                        "error": resolution["error"],
                    }
                )
            warnings.append(f"primary resolver: {resolution['error']}")
        record = resolution.get("record") if isinstance(resolution, dict) else {}
        if isinstance(record, dict):
            target_url = _best_open_access_url(record)
            if not doi:
                doi = _extract_doi(
                    " ".join(
                        str(part)
                        for part in (
                            record.get("doi"),
                            record.get("url"),
                            record.get("identifiers"),
                        )
                    )
                )
        if not target_url and doi:
            openalex = _fetch_json(
                _openalex_doi_url(doi),
                http_client_factory=http_client_factory,
            )
            if "error" in openalex:
                warnings.append(f"openalex DOI lookup: {openalex['error']}")
            else:
                open_access_record = _normalize_openalex_work(openalex.get("json") or {})
                if not isinstance(resolution, dict) or "error" in resolution:
                    resolution = {
                        "schema_version": _SCHEMA_VERSION,
                        "identifier": value,
                        "record": open_access_record,
                        "evidence_role": "contextual_literature_signal",
                    }
                else:
                    resolution = {
                        **resolution,
                        "open_access_record": open_access_record,
                    }
                target_url = _best_open_access_url(open_access_record)
        if not target_url:
            return _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "identifier_or_url": value,
                    "resolved": resolution,
                    "error": "no open-access URL found without credentials",
                    "warnings": warnings,
                    "paywall_policy": "No key bypasses a paywall; provide licensed access outside Praxist if needed.",
                }
            )

    fetch = _fetch_text_or_pdf(target_url, http_client_factory=http_client_factory)
    if "error" in fetch:
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier_or_url": value,
                "resolved": resolution,
                "retrieval_url": target_url,
                "error": fetch["error"],
                "warnings": warnings,
            }
        )
    final_url = str(fetch.get("final_url") or target_url)
    if not _is_allowed_open_access_url(final_url):
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier_or_url": value,
                "resolved": resolution,
                "retrieval_url": target_url,
                "final_url": final_url,
                "error": "open-access fetch redirected to a non-approved or non-OA host",
                "warnings": warnings,
            }
        )

    fetched_at = _now_iso_utc(clock)
    content_type = str(fetch.get("content_type") or "")
    raw_bytes = fetch.get("bytes") if isinstance(fetch.get("bytes"), bytes) else b""
    text = str(fetch.get("text") or "")
    if raw_bytes and not text:
        content_hash = "sha256:" + sha256(raw_bytes).hexdigest()
        size_bytes = len(raw_bytes)
    else:
        encoded = text.encode("utf-8")
        content_hash = "sha256:" + sha256(encoded).hexdigest()
        size_bytes = len(encoded)
    text = text[:max_text_chars]
    truncated = bool(fetch.get("text")) and len(str(fetch.get("text"))) > max_text_chars

    return _with_resource_policy(
        {
            "schema_version": _SCHEMA_VERSION,
            "identifier_or_url": value,
            "retrieval_url": final_url,
            "requested_url": target_url,
            "content_type": content_type,
            "text": text,
            "text_truncated": truncated,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "fetched_at": fetched_at,
            "resolved": resolution,
            "warnings": warnings,
            "evidence_role": "contextual_literature_signal",
            "provenance": {
                "retrieval_url": final_url,
                "requested_url": target_url,
                "fetched_at": fetched_at,
                "content_hash": content_hash,
                "open_access_only": True,
                "credential_scope": "none",
            },
            "pdf_read_note": (
                "PDF bytes were fetched but not OCR'd; use tool_server:pdf_reader for page text."
                if "pdf" in content_type.lower() and not text
                else ""
            ),
        }
    )


def literature_resolve(
    identifier: str,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """Resolve a DOI, PMID, arXiv ID, or OpenAlex ID into one normalized record."""
    if not isinstance(identifier, str) or not identifier.strip():
        return _with_resource_policy(
            {"schema_version": _SCHEMA_VERSION, "error": "identifier is required"}
        )
    value = identifier.strip()

    if _looks_like_arxiv_id(value):
        arxiv_id = _normalize_arxiv_id(value)
        payload = arxiv_adapter.arxiv_get(
            arxiv_id,
            http_client_factory=http_client_factory,
            clock=clock,
        )
        if "error" in payload:
            return _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "identifier": value,
                    "error": payload["error"],
                }
            )
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier": value,
                "normalized_identifier": arxiv_id,
                "record": _normalize_arxiv_record(payload["paper"]),
                "evidence_role": "contextual_literature_signal",
            }
        )

    pmid = _extract_pmid(value)
    if pmid is not None:
        payload = _fetch_pubmed_summaries([pmid], http_client_factory=http_client_factory)
        if "error" in payload:
            return _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "identifier": value,
                    "error": payload["error"],
                }
            )
        records = payload.get("records") or []
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier": value,
                "normalized_identifier": pmid,
                "record": records[0] if records else None,
                "evidence_role": "contextual_literature_signal",
            }
        )

    if _looks_like_openalex_id(value):
        openalex_id = _normalize_openalex_id(value)
        url = _openalex_work_url(openalex_id)
        payload = _fetch_json(url, http_client_factory=http_client_factory)
        if "error" in payload:
            return _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "identifier": value,
                    "error": payload["error"],
                }
            )
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier": value,
                "normalized_identifier": openalex_id,
                "record": _normalize_openalex_work(payload["json"]),
                "evidence_role": "contextual_literature_signal",
            }
        )

    doi = _extract_doi(value)
    if doi:
        url = f"{_CROSSREF_WORK_URL}/{quote(doi, safe='')}"
        payload = _fetch_json(url, http_client_factory=http_client_factory)
        if "error" in payload:
            return _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "identifier": value,
                    "error": payload["error"],
                }
            )
        message = (payload.get("json") or {}).get("message") or {}
        return _with_resource_policy(
            {
                "schema_version": _SCHEMA_VERSION,
                "identifier": value,
                "record": _normalize_crossref_work(message),
                "evidence_role": "contextual_literature_signal",
            }
        )

    return _with_resource_policy(
        {
            "schema_version": _SCHEMA_VERSION,
            "identifier": value,
            "error": "identifier must look like a DOI, PMID, arXiv ID, or OpenAlex work ID",
        }
    )


def literature_source_guide(domain: str = "", objective: str = "") -> dict[str, Any]:
    """Return source-selection guidance for task-local research context gathering."""
    domain_key = _domain_key(domain)
    sources = {
        "machine_learning": [
            "arxiv",
            "OpenAlex",
            "Crossref",
            "Semantic Scholar metadata when rate limits allow",
            "Papers with Code or project pages via web search",
            "benchmark documentation and official repositories",
        ],
        "biology": [
            "PubMed",
            "PubMed Central open access",
            "Europe PMC",
            "OpenAlex",
            "UniProt, PDB, AlphaFold, GEO, ENCODE, ClinVar when task-relevant",
        ],
        "chemistry": [
            "PubChem",
            "ChEMBL",
            "BindingDB",
            "OpenAlex",
            "publisher abstracts or open access copies",
        ],
        "medicine": [
            "PubMed",
            "ClinicalTrials.gov",
            "Europe PMC",
            "openFDA",
            "Guidelines and systematic reviews from official sources",
        ],
        "robotics_control": [
            "arxiv",
            "OpenAlex",
            "benchmark/simulator docs",
            "official repositories for environment and policy baselines",
        ],
        "physics_materials": [
            "arxiv",
            "OpenAlex",
            "materials databases documented by the task owner",
            "official simulation package docs",
        ],
        "generic": [
            "OpenAlex",
            "Crossref",
            "arxiv when preprints are common",
            "PubMed for biomedical topics",
            "official datasets, benchmarks, standards, and repositories",
        ],
    }
    checks = [
        "Prefer primary sources, official benchmark docs, and reproducible code/data pages.",
        "Separate literature/context claims from measured task performance.",
        "Record source URL, identifier, retrieval date, and how the source changes the task plan.",
        "Use public no-key sources first; only request credentials when the task owner explicitly requires a licensed source.",
        "Do not download new datasets, install new packages, or provision new environments during a run just because a source mentions them; adapt ideas to the task's existing local resources.",
    ]
    return _with_resource_policy(
        {
            "schema_version": _SCHEMA_VERSION,
            "domain": domain or "generic",
            "objective": objective or "",
            "recommended_sources": sources.get(domain_key, sources["generic"]),
            "verification_checks": checks,
            "available_no_key_tools": [
                "literature_search",
                "literature_resolve",
                "literature_open_access_text",
                "scientific_database_search",
            ],
            "evidence_role": "research_planning_guidance",
        }
    )


def _with_resource_policy(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("resource_policy", _CURRENT_ENVIRONMENT_RESOURCE_POLICY)
    return payload


def handle_literature_search(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest handler wrapper for direct tool execution."""
    max_results = _handler_max_results(args.get("max_results", _DEFAULT_MAX_RESULTS))
    if max_results is None:
        return _text_result(
            _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "error": "max_results must be an integer",
                    "result_count": 0,
                    "results": [],
                }
            )
        )
    return _text_result(
        literature_search(
            str(args.get("query", "")),
            args.get("sources"),
            max_results,
        )
    )


def handle_literature_resolve(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest handler wrapper for direct tool execution."""
    return _text_result(literature_resolve(str(args.get("identifier", ""))))


def handle_literature_open_access_text(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest handler wrapper for direct tool execution."""
    return _text_result(
        literature_open_access_text(
            str(args.get("identifier_or_url", args.get("identifier", ""))),
            args.get("max_chars", _DEFAULT_MAX_TEXT_CHARS),
        )
    )


def handle_scientific_database_search(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest handler wrapper for direct tool execution."""
    max_results = _handler_max_results(args.get("max_results", _DEFAULT_MAX_RESULTS))
    if max_results is None:
        return _text_result(
            _with_resource_policy(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "error": "max_results must be an integer",
                    "result_count": 0,
                    "results": [],
                }
            )
        )
    return _text_result(
        scientific_database_search(
            str(args.get("query", "")),
            args.get("sources"),
            max_results,
        )
    )


def handle_literature_source_guide(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest handler wrapper for direct tool execution."""
    return _text_result(
        literature_source_guide(str(args.get("domain", "")), str(args.get("objective", "")))
    )


def _normalize_sources(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw is None or raw == "":
        values = list(_DEFAULT_SOURCES)
    elif isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    else:
        values = [str(item).strip() for item in raw]
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        source = _SOURCE_ALIASES.get(value.lower(), value.lower())
        if source not in normalized:
            normalized.append(source)
    return normalized or list(_DEFAULT_SOURCES)


def _coerce_max_results(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_RESULTS
    return max(1, min(value, _MAX_RESULTS_CAP))


def _round_robin_records(
    groups: list[list[dict[str, Any]]], max_results: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if max_results <= 0:
        return records
    index = 0
    while len(records) < max_results:
        added = False
        for group in groups:
            if index >= len(group):
                continue
            records.append(group[index])
            added = True
            if len(records) >= max_results:
                break
        if not added:
            break
        index += 1
    return records


def _handler_max_results(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    if raw is None:
        return _DEFAULT_MAX_RESULTS
    if not isinstance(raw, int | str):
        return None
    try:
        int(raw)
    except ValueError:
        return None
    try:
        return _coerce_max_results(raw)
    except Exception:  # noqa: BLE001 - malformed MCP inputs must fail closed.
        return None


def _coerce_max_text_chars(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _DEFAULT_MAX_TEXT_CHARS
    return max(500, min(value, _MAX_TEXT_CHARS_CAP))


def _search_arxiv(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    payload = arxiv_adapter.arxiv_search(
        query,
        max_results=max_results,
        sort_by="relevance",
        http_client_factory=http_client_factory,
        clock=clock,
    )
    if "error" in payload:
        return {"error": payload["error"]}
    return {"records": [_normalize_arxiv_record(item) for item in payload.get("results") or []]}


def _search_openalex(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _OPENALEX_WORKS_URL,
        params={"search": query, "per-page": max_results},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    data = payload.get("json") or {}
    works = data.get("results") if isinstance(data, dict) else []
    if not isinstance(works, list):
        return {"records": [], "warnings": ["openalex: malformed results"]}
    return {"records": [_normalize_openalex_work(item) for item in works if isinstance(item, dict)]}


def _search_pubmed(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    search_payload = _fetch_json(
        _PUBMED_ESEARCH_URL,
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        },
        http_client_factory=http_client_factory,
    )
    if "error" in search_payload:
        return search_payload
    data = search_payload.get("json") or {}
    ids = (
        (((data.get("esearchresult") or {}).get("idlist")) or []) if isinstance(data, dict) else []
    )
    ids = [str(item) for item in ids[:max_results]]
    if not ids:
        return {"records": []}
    return _fetch_pubmed_summaries(ids, http_client_factory=http_client_factory)


def _search_crossref(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _CROSSREF_WORK_URL,
        params={"query": query, "rows": max_results},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    message = (payload.get("json") or {}).get("message") or {}
    items = message.get("items") if isinstance(message, dict) else []
    if not isinstance(items, list):
        return {"records": [], "warnings": ["crossref: malformed items"]}
    return {"records": [_normalize_crossref_work(item) for item in items if isinstance(item, dict)]}


def _search_semantic_scholar(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _SEMANTIC_SCHOLAR_SEARCH_URL,
        params={
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,externalIds,venue,openAccessPdf",
        },
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    raw_json = payload.get("json") or {}
    data = raw_json.get("data") if isinstance(raw_json, dict) and "data" in raw_json else []
    if not isinstance(data, list):
        return {"records": [], "warnings": ["semantic_scholar: malformed data"]}
    return {
        "records": [
            _normalize_semantic_scholar_paper(item) for item in data if isinstance(item, dict)
        ]
    }


def _search_europepmc(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _EUROPEPMC_SEARCH_URL,
        params={"query": query, "format": "json", "pageSize": max_results},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    data = payload.get("json") or {}
    result_container = data.get("resultList") if isinstance(data, dict) else {}
    result_list = result_container.get("result") if isinstance(result_container, dict) else None
    if result_list is None:
        result_list = []
    if not isinstance(result_list, list):
        return {"records": [], "warnings": ["europepmc: malformed results"]}
    return {
        "records": [
            _normalize_europepmc_record(item) for item in result_list if isinstance(item, dict)
        ]
    }


def _search_uniprot(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _UNIPROT_SEARCH_URL,
        params={"query": query, "format": "json", "size": max_results},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    data = payload.get("json") or {}
    results = data.get("results") if isinstance(data, dict) and "results" in data else []
    if not isinstance(results, list):
        return {"records": [], "warnings": ["uniprot: malformed results"]}
    return {
        "records": [_normalize_uniprot_record(item) for item in results if isinstance(item, dict)]
    }


def _search_clinicaltrials(
    query: str,
    max_results: int,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _CLINICALTRIALS_STUDIES_URL,
        params={"query.term": query, "pageSize": max_results},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    data = payload.get("json") or {}
    studies = data.get("studies") if isinstance(data, dict) and "studies" in data else []
    if not isinstance(studies, list):
        return {"records": [], "warnings": ["clinicaltrials: malformed studies"]}
    return {
        "records": [
            _normalize_clinicaltrials_study(item) for item in studies if isinstance(item, dict)
        ]
    }


def _fetch_pubmed_summaries(
    ids: list[str],
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    payload = _fetch_json(
        _PUBMED_ESUMMARY_URL,
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        http_client_factory=http_client_factory,
    )
    if "error" in payload:
        return payload
    data = payload.get("json") or {}
    result = data.get("result") if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        return {"records": [], "warnings": ["pubmed: malformed summary"]}
    records = [
        _normalize_pubmed_summary(result[pmid])
        for pmid in ids
        if isinstance(result.get(pmid), dict)
    ]
    return {"records": records}


def _fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    if http_client_factory is None:  # pragma: no cover - production-only
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS)

        factory: Any = _factory
        request_error: tuple[type[Exception], ...] = (httpx.RequestError,)
    else:
        factory = http_client_factory
        request_error = (Exception,)

    try:
        with factory() as client:
            response = client.get(url, params=params or {})
    except request_error as exc:  # pragma: no cover - network error path
        return {"schema_version": _SCHEMA_VERSION, "error": f"request failed: {exc}"}

    status = getattr(response, "status_code", None)
    if status != 200:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"returned status {status}",
            "status_code": status,
        }
    try:
        return {"schema_version": _SCHEMA_VERSION, "json": response.json()}
    except Exception:
        text = getattr(response, "text", "")
        try:
            return {"schema_version": _SCHEMA_VERSION, "json": json.loads(text)}
        except Exception as exc:  # noqa: BLE001 - malformed JSON is a remote failure.
            return {"schema_version": _SCHEMA_VERSION, "error": f"invalid JSON: {exc}"}


def _fetch_text_or_pdf(
    url: str,
    *,
    http_client_factory: Any = None,
) -> dict[str, Any]:
    if http_client_factory is None:  # pragma: no cover - production-only
        import httpx  # type: ignore[import-not-found]

        def _factory() -> Any:
            return httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS, follow_redirects=False)

        factory: Any = _factory
        request_error: tuple[type[Exception], ...] = (httpx.RequestError,)
    else:
        factory = http_client_factory
        request_error = (Exception,)

    try:
        with factory() as client:
            response = client.get(url)
            status = getattr(response, "status_code", None)
            if status in {301, 302, 303, 307, 308}:
                location = _redirect_location(url, getattr(response, "headers", {}) or {})
                if not location:
                    return {
                        "schema_version": _SCHEMA_VERSION,
                        "error": "redirect response did not include a Location header",
                    }
                if not _is_allowed_open_access_url(location):
                    return {
                        "schema_version": _SCHEMA_VERSION,
                        "error": "open-access fetch redirected to a non-approved or non-OA host",
                        "final_url": location,
                    }
                response = client.get(location)
    except request_error as exc:  # pragma: no cover - network error path
        return {"schema_version": _SCHEMA_VERSION, "error": f"request failed: {exc}"}
    status = getattr(response, "status_code", None)
    if status != 200:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"returned status {status}",
            "status_code": status,
        }
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "")
    raw = getattr(response, "content", None)
    if raw is None:
        raw = str(getattr(response, "text", "") or "").encode("utf-8")
    if not isinstance(raw, bytes):
        raw = bytes(str(raw), encoding="utf-8")
    if len(raw) > _MAX_FULL_TEXT_BYTES:
        return {
            "schema_version": _SCHEMA_VERSION,
            "error": f"response exceeded {_MAX_FULL_TEXT_BYTES} bytes",
        }
    if b"%PDF" in raw[:1024] or "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return {
            "schema_version": _SCHEMA_VERSION,
            "content_type": "application/pdf",
            "bytes": raw,
            "final_url": str(getattr(response, "url", url) or url),
        }
    text = getattr(response, "text", "")
    if not isinstance(text, str) or not text:
        text = raw.decode("utf-8", errors="ignore")
    return {
        "schema_version": _SCHEMA_VERSION,
        "content_type": content_type or "text/html",
        "text": _strip_html_text(text),
        "final_url": str(getattr(response, "url", url) or url),
    }


def _normalize_arxiv_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "arxiv",
        "title": _compact(item.get("title")),
        "authors": list(item.get("authors") or []),
        "year": _year(item.get("submitted_date")),
        "published": item.get("submitted_date") or "",
        "updated": item.get("updated_date") or "",
        "abstract": _compact(item.get("abstract")),
        "url": item.get("html_url") or "",
        "pdf_url": item.get("pdf_url") or "",
        "identifiers": {"arxiv_id": item.get("arxiv_id") or ""},
    }


def _normalize_openalex_work(item: dict[str, Any]) -> dict[str, Any]:
    authorships = item.get("authorships") if isinstance(item, dict) else []
    authors: list[str] = []
    if isinstance(authorships, list):
        for authorship in authorships[:12]:
            author = (authorship or {}).get("author") if isinstance(authorship, dict) else {}
            name = author.get("display_name") if isinstance(author, dict) else None
            if name:
                authors.append(str(name))
    doi = item.get("doi") or ""
    primary_location = item.get("primary_location") if isinstance(item, dict) else {}
    open_access = item.get("open_access") if isinstance(item, dict) else {}
    open_access_url = ""
    is_open_access = bool(open_access.get("is_oa")) if isinstance(open_access, dict) else None
    if isinstance(open_access, dict) and is_open_access:
        open_access_url = str(open_access.get("oa_url") or "")
    if not open_access_url and is_open_access and isinstance(primary_location, dict):
        open_access_url = str(primary_location.get("pdf_url") or "")
    return {
        "source": "openalex",
        "title": _compact(item.get("title") or item.get("display_name")),
        "authors": authors,
        "year": item.get("publication_year"),
        "published": item.get("publication_date") or "",
        "abstract": _abstract_from_openalex(item),
        "url": item.get("id") or _openalex_landing_page(item),
        "open_access_url": open_access_url,
        "is_open_access": is_open_access,
        "doi": doi,
        "identifiers": {
            "openalex_id": item.get("id") or "",
            "doi": doi,
        },
    }


def _normalize_pubmed_summary(item: dict[str, Any]) -> dict[str, Any]:
    authors_raw = item.get("authors") or []
    authors = [
        str(author.get("name"))
        for author in authors_raw[:12]
        if isinstance(author, dict) and author.get("name")
    ]
    doi = _extract_doi(" ".join(str(item.get(key, "")) for key in ("elocationid", "articleids")))
    pmid = str(item.get("uid") or "")
    return {
        "source": "pubmed",
        "title": _compact(item.get("title")),
        "authors": authors,
        "year": _year(item.get("pubdate") or item.get("epubdate")),
        "published": item.get("pubdate") or item.get("epubdate") or "",
        "abstract": "",
        "venue": item.get("fulljournalname") or item.get("source") or "",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "doi": doi or "",
        "identifiers": {"pmid": pmid, "doi": doi or ""},
    }


def _normalize_crossref_work(item: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(part for part in [author.get("given"), author.get("family")] if part)
        if name:
            authors.append(name)
    date_parts = (
        (item.get("published-print") or item.get("published-online") or {}).get("date-parts")
    ) or [[]]
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    return {
        "source": "crossref",
        "title": _compact(
            (item.get("title") or [""])[0]
            if isinstance(item.get("title"), list)
            else item.get("title")
        ),
        "authors": authors[:12],
        "year": year,
        "published": str(year or ""),
        "abstract": _compact(item.get("abstract")),
        "venue": (item.get("container-title") or [""])[0]
        if isinstance(item.get("container-title"), list)
        else item.get("container-title", ""),
        "url": item.get("URL") or "",
        "doi": item.get("DOI") or "",
        "identifiers": {"doi": item.get("DOI") or ""},
    }


def _normalize_semantic_scholar_paper(item: dict[str, Any]) -> dict[str, Any]:
    authors = [
        str(author.get("name"))
        for author in item.get("authors") or []
        if isinstance(author, dict) and author.get("name")
    ]
    external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    open_pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
    return {
        "source": "semantic_scholar",
        "title": _compact(item.get("title")),
        "authors": authors[:12],
        "year": item.get("year"),
        "published": str(item.get("year") or ""),
        "abstract": _compact(item.get("abstract")),
        "venue": item.get("venue") or "",
        "url": item.get("url") or "",
        "open_access_url": open_pdf.get("url") if isinstance(open_pdf, dict) else "",
        "is_open_access": bool(open_pdf.get("url")) if isinstance(open_pdf, dict) else False,
        "doi": external.get("DOI") or "",
        "identifiers": {
            "semantic_scholar_paper_id": item.get("paperId") or "",
            "doi": external.get("DOI") or "",
            "pmid": external.get("PubMed") or "",
            "arxiv_id": external.get("ArXiv") or "",
        },
    }


def _normalize_europepmc_record(item: dict[str, Any]) -> dict[str, Any]:
    pmcid = item.get("pmcid") or ""
    doi = item.get("doi") or ""
    return {
        "source": "europepmc",
        "record_type": "publication",
        "title": _compact(item.get("title")),
        "authors": [
            part.strip() for part in str(item.get("authorString") or "").split(",") if part.strip()
        ][:12],
        "year": _year(item.get("pubYear") or item.get("firstPublicationDate")),
        "published": item.get("firstPublicationDate") or item.get("pubYear") or "",
        "abstract": _compact(item.get("abstractText")),
        "venue": item.get("journalTitle") or "",
        "url": f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
        "open_access_url": f"https://europepmc.org/articles/{pmcid}" if pmcid else "",
        "is_open_access": bool(pmcid),
        "doi": doi,
        "identifiers": {
            "pmid": item.get("pmid") or "",
            "pmcid": pmcid,
            "doi": doi,
        },
    }


def _normalize_uniprot_record(item: dict[str, Any]) -> dict[str, Any]:
    accession = str(item.get("primaryAccession") or "") or str(item.get("uniProtkbId") or "")
    protein = (
        item.get("proteinDescription") if isinstance(item.get("proteinDescription"), dict) else {}
    )
    recommended = protein.get("recommendedName") if isinstance(protein, dict) else {}
    full_name = ""
    if isinstance(recommended, dict):
        full = recommended.get("fullName")
        if isinstance(full, dict):
            full_name = str(full.get("value") or "")
    organism = item.get("organism") if isinstance(item.get("organism"), dict) else {}
    return {
        "source": "uniprot",
        "record_type": "protein",
        "title": full_name or accession,
        "organism": organism.get("scientificName") if isinstance(organism, dict) else "",
        "url": f"https://www.uniprot.org/uniprotkb/{accession}/entry" if accession else "",
        "identifiers": {"uniprot_accession": accession},
        "summary": _compact(item.get("uniProtkbId") or full_name),
    }


def _normalize_clinicaltrials_study(item: dict[str, Any]) -> dict[str, Any]:
    protocol = item.get("protocolSection") if isinstance(item.get("protocolSection"), dict) else {}
    identification = protocol.get("identificationModule") if isinstance(protocol, dict) else {}
    status = protocol.get("statusModule") if isinstance(protocol, dict) else {}
    design = protocol.get("designModule") if isinstance(protocol, dict) else {}
    conditions = protocol.get("conditionsModule") if isinstance(protocol, dict) else {}
    nct_id = identification.get("nctId") if isinstance(identification, dict) else ""
    return {
        "source": "clinicaltrials",
        "record_type": "clinical_trial",
        "title": identification.get("briefTitle") if isinstance(identification, dict) else "",
        "status": status.get("overallStatus") if isinstance(status, dict) else "",
        "phase": design.get("phases") if isinstance(design, dict) else [],
        "conditions": conditions.get("conditions") if isinstance(conditions, dict) else [],
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "identifiers": {"nct_id": nct_id or ""},
    }


def _abstract_from_openalex(item: dict[str, Any]) -> str:
    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, str(word)))
    return _compact(" ".join(word for _idx, word in sorted(positions)))


def _compact(value: Any, *, limit: int = 2000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _year(value: Any) -> int | None:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _extract_doi(value: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", value)
    if not match:
        return None
    return match.group(0).rstrip(".,);]")


def _identifier_from_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return ""
    if _is_allowed_open_access_url(value):
        return ""
    doi = _extract_doi(value)
    if doi:
        return doi
    pmid = _extract_pmid(value)
    if pmid is not None:
        return pmid
    arxiv_id = _extract_arxiv_id(value)
    if arxiv_id is not None:
        return arxiv_id
    openalex_id = _normalize_openalex_id(value)
    return openalex_id


def _looks_like_pmid(value: str) -> bool:
    return _extract_pmid(value) is not None


def _extract_pmid(value: str) -> str | None:
    stripped = value.strip()
    exact = re.fullmatch(r"(?:PMID:?\s*)?(\d{4,12})", stripped, flags=re.IGNORECASE)
    if exact:
        return exact.group(1)
    url_match = re.search(
        r"(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d{4,12})(?:[/?#]|$)",
        stripped,
        flags=re.IGNORECASE,
    )
    return url_match.group(1) if url_match else None


def _looks_like_arxiv_id(value: str) -> bool:
    return _extract_arxiv_id(value) is not None


def _extract_arxiv_id(value: str) -> str | None:
    stripped = value.strip()
    prefix_match = re.fullmatch(r"arxiv:\s*(.+)", stripped, flags=re.IGNORECASE)
    if prefix_match:
        stripped = prefix_match.group(1).strip()
    url_match = re.search(
        r"arxiv\.org/(?:abs|pdf|html)/([^?#\s]+)",
        stripped,
        flags=re.IGNORECASE,
    )
    if url_match:
        stripped = url_match.group(1).strip()
    stripped = re.sub(r"\.pdf$", "", stripped, flags=re.IGNORECASE)
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", stripped):
        return stripped
    if re.fullmatch(r"[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?", stripped):
        return stripped
    return None


def _normalize_arxiv_id(value: str) -> str:
    return _extract_arxiv_id(value) or value.strip()


def _looks_like_openalex_id(value: str) -> bool:
    return _normalize_openalex_id(value) != ""


def _normalize_openalex_id(value: str) -> str:
    stripped = value.strip()
    match = re.search(r"(?:openalex\.org|api\.openalex\.org/works)/(W\d+)", stripped)
    if match:
        return match.group(1)
    return stripped if re.fullmatch(r"W\d+", stripped) else ""


def _openalex_work_url(value: str) -> str:
    if value.startswith("https://openalex.org/"):
        value = value.rstrip("/").rsplit("/", 1)[-1]
    return f"{_OPENALEX_WORKS_URL}/{value}"


def _openalex_doi_url(doi: str) -> str:
    return f"{_OPENALEX_WORKS_URL}/{quote('https://doi.org/' + doi, safe=':/')}"


def _openalex_landing_page(item: dict[str, Any]) -> str:
    location = item.get("primary_location")
    if not isinstance(location, dict):
        return ""
    return str(location.get("landing_page_url") or "")


def _best_open_access_url(record: dict[str, Any]) -> str:
    for key in ("open_access_url", "pdf_url"):
        value = record.get(key)
        if (
            isinstance(value, str)
            and value.startswith(("http://", "https://"))
            and _is_allowed_open_access_url(value)
        ):
            return value
    identifiers = record.get("identifiers")
    if isinstance(identifiers, dict):
        arxiv_id = str(identifiers.get("arxiv_id") or "")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}"
    return ""


def _is_allowed_open_access_url(url: str) -> bool:
    if not _is_public_https_url(url):
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower().strip(".")
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in _DIRECT_OPEN_ACCESS_HOST_SUFFIXES
    )


def _is_public_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower().strip(".")
    if not host or host == "localhost":
        return False
    try:
        addr = ip_address(host)
    except ValueError:
        return True
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _redirect_location(base_url: str, headers: dict[str, Any]) -> str:
    location = str(headers.get("location") or headers.get("Location") or "").strip()
    if not location:
        return ""
    return urljoin(base_url, location)


def _strip_html_text(raw: str) -> str:
    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._skip_depth = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"script", "style", "noscript"}:
                self._skip_depth += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data: str) -> None:
            if self._skip_depth == 0 and data.strip():
                self.parts.append(data.strip())

    stripper = _Stripper()
    stripper.feed(raw)
    return _compact(" ".join(stripper.parts), limit=_MAX_TEXT_CHARS_CAP)


def _now_iso_utc(clock: Any = None) -> str:
    if clock is None:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    value = clock[0]() if isinstance(clock, tuple) and callable(clock[0]) else clock()
    if not isinstance(value, (int, float, str)):
        return str(value)
    try:
        return (
            datetime.fromtimestamp(float(value), UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        return str(value)


def _domain_key(domain: str) -> str:
    text = domain.lower()
    if any(token in text for token in ("ml", "machine", "ai", "llm", "vision", "learning")):
        return "machine_learning"
    if any(token in text for token in ("bio", "gene", "protein", "genomic", "omics")):
        return "biology"
    if any(token in text for token in ("chem", "drug", "molecule", "compound")):
        return "chemistry"
    if any(token in text for token in ("medical", "medicine", "clinical", "health")):
        return "medicine"
    if any(token in text for token in ("robot", "control", "slam", "landing")):
        return "robotics_control"
    if any(token in text for token in ("physics", "material", "fusion", "plasma")):
        return "physics_materials"
    return "generic"


def _text_result(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def create_literature_lookup_server() -> Any:  # pragma: no cover - requires claude_agent_sdk
    """Create the MCP server exposing public literature lookup tools."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")

    async def _handle_search(args: dict[str, Any]) -> dict[str, Any]:
        return handle_literature_search(args)

    async def _handle_resolve(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(literature_resolve(str(args.get("identifier", ""))))

    async def _handle_source_guide(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            literature_source_guide(str(args.get("domain", "")), str(args.get("objective", "")))
        )

    async def _handle_open_access_text(args: dict[str, Any]) -> dict[str, Any]:
        return _text_result(
            literature_open_access_text(
                str(args.get("identifier_or_url", args.get("identifier", ""))),
                args.get("max_chars", _DEFAULT_MAX_TEXT_CHARS),
            )
        )

    async def _handle_database_search(args: dict[str, Any]) -> dict[str, Any]:
        return handle_scientific_database_search(args)

    search_tool = tool(
        "literature_search",
        (
            "Search no-key public literature sources. sources is a comma-separated "
            "list using arxiv, openalex, pubmed, crossref, semantic_scholar; "
            "default searches compact public indexes."
        ),
        {"query": str, "sources": str, "max_results": int},
    )(_handle_search)
    resolve_tool = tool(
        "literature_resolve",
        "Resolve a DOI, PMID, arXiv ID, or OpenAlex work ID into normalized metadata.",
        {"identifier": str},
    )(_handle_resolve)
    guide_tool = tool(
        "literature_source_guide",
        "Return task-agnostic source-selection guidance for a research domain.",
        {"domain": str, "objective": str},
    )(_handle_source_guide)
    open_access_tool = tool(
        "literature_open_access_text",
        (
            "Fetch open-access HTML/XML text or PDF provenance for a DOI, PMID, "
            "arXiv/OpenAlex identifier, or URL. Never bypasses paywalls."
        ),
        {"identifier_or_url": str, "max_chars": int},
    )(_handle_open_access_text)
    database_tool = tool(
        "scientific_database_search",
        (
            "Search no-key public scientific databases such as Europe PMC, "
            "UniProt, and ClinicalTrials.gov; returns contextual database signals."
        ),
        {"query": str, "sources": str, "max_results": int},
    )(_handle_database_search)
    return create_sdk_mcp_server(
        "literature-lookup",
        tools=[search_tool, resolve_tool, guide_tool, open_access_tool, database_tool],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint exposing the literature lookup tool server descriptor."""
    return {
        "tool_server_ref": "tool_server:literature_lookup",
        "server_name": "literature-lookup",
        "factory": (
            "praxist.plugins.tools.literature_lookup.adapter:create_literature_lookup_server"
        ),
        "tool_names": [
            "literature_search",
            "literature_resolve",
            "literature_source_guide",
            "literature_open_access_text",
            "scientific_database_search",
        ],
        "visibility": ["peer", "panel"],
        "required_capability": "tool_server.literature_lookup",
        "handlers": {
            "literature_search": (
                "praxist.plugins.tools.literature_lookup.adapter:handle_literature_search"
            ),
            "literature_resolve": (
                "praxist.plugins.tools.literature_lookup.adapter:handle_literature_resolve"
            ),
            "literature_source_guide": (
                "praxist.plugins.tools.literature_lookup.adapter:handle_literature_source_guide"
            ),
            "literature_open_access_text": (
                "praxist.plugins.tools.literature_lookup.adapter:handle_literature_open_access_text"
            ),
            "scientific_database_search": (
                "praxist.plugins.tools.literature_lookup.adapter:handle_scientific_database_search"
            ),
        },
    }
