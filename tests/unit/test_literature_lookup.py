"""Unit tests for the no-key literature lookup tool server."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from praxist.plugins.tools.literature_lookup import adapter as literature


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        text: str = "",
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        url: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = content if content is not None else self.text.encode("utf-8")
        self.headers = headers or {}
        self.url = url

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params or {}), "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("test scripted no further responses")
        return self._responses.pop(0)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _RaisingClient:
    def get(self, *_args: Any, **_kwargs: Any) -> _FakeResponse:
        raise RuntimeError("network down")

    def __enter__(self) -> _RaisingClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _factory_for(client: _FakeClient):
    @contextmanager
    def _inner():
        yield client

    return _inner


def _raising_factory():
    @contextmanager
    def _inner():
        yield _RaisingClient()

    return _inner


def _zero_clock() -> tuple[Any, Any]:
    return (lambda: 0.0, lambda _s: None)


_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Efficient Control Search</title>
    <summary>Abstract text.</summary>
    <author><name>Ada</name></author>
    <category term="cs.LG"/>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2401.00001v1"/>
    <link rel="related" type="application/pdf" href="http://arxiv.org/pdf/2401.00001v1"/>
  </entry>
</feed>
"""


class LiteratureLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        from praxist.plugins.tools.arxiv import adapter as arxiv

        arxiv._reset_rate_limit_for_tests()

    def test_empty_query_returns_error(self) -> None:
        result = literature.literature_search(" ")

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["error"], "query is required")

    def test_malformed_max_results_uses_default_cap(self) -> None:
        client = _FakeClient([_FakeResponse(200, {"results": []})])

        result = literature.literature_search(
            "x",
            sources="openalex",
            max_results="bad",  # type: ignore[arg-type]
            http_client_factory=_factory_for(client),
        )

        self.assertEqual(result["result_count"], 0)
        self.assertEqual(client.calls[0]["params"]["per-page"], literature._DEFAULT_MAX_RESULTS)
        self.assertEqual(literature._coerce_max_results(None), literature._DEFAULT_MAX_RESULTS)
        self.assertEqual(literature._coerce_max_results(0), 1)
        self.assertEqual(literature._coerce_max_results(999), literature._MAX_RESULTS_CAP)

    def test_search_combines_no_key_sources(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(200, text=_ARXIV_XML),
                _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "id": "https://openalex.org/W123",
                                "title": "OpenAlex Work",
                                "publication_year": 2025,
                                "publication_date": "2025-05-01",
                                "doi": "https://doi.org/10.1234/example",
                                "authorships": [
                                    {"author": {"display_name": "Grace Hopper"}},
                                ],
                                "abstract_inverted_index": {
                                    "Normalized": [0],
                                    "record": [1],
                                },
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    {"esearchresult": {"idlist": ["123456"]}},
                ),
                _FakeResponse(
                    200,
                    {
                        "result": {
                            "123456": {
                                "uid": "123456",
                                "title": "PubMed Work",
                                "pubdate": "2023 Jan",
                                "fulljournalname": "Journal",
                                "authors": [{"name": "Linus"}],
                            }
                        }
                    },
                ),
                _FakeResponse(
                    200,
                    {
                        "message": {
                            "items": [
                                {
                                    "title": ["Crossref Work"],
                                    "author": [{"given": "Ada", "family": "L"}],
                                    "published-online": {"date-parts": [[2022]]},
                                    "DOI": "10.5555/cross",
                                }
                            ]
                        }
                    },
                ),
            ]
        )

        result = literature.literature_search(
            "control search",
            sources="arxiv,openalex,pubmed,crossref",
            max_results=10,
            http_client_factory=_factory_for(client),
            clock=_zero_clock(),
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["evidence_role"], "contextual_literature_signal")
        self.assertIn("Current-environment only", result["resource_policy"])
        self.assertIn("do not download", result["resource_policy"])
        self.assertEqual(result["result_count"], 4)
        self.assertEqual(
            [entry["source"] for entry in result["results"]],
            ["arxiv", "openalex", "pubmed", "crossref"],
        )
        self.assertEqual(client.calls[0]["params"]["search_query"], "control search")
        self.assertEqual(client.calls[1]["params"]["search"], "control search")
        self.assertEqual(client.calls[2]["params"]["db"], "pubmed")

    def test_arxiv_source_uses_https_and_returns_records_without_301_warning(self) -> None:
        client = _FakeClient([_FakeResponse(200, text=_ARXIV_XML)])

        result = literature.literature_search(
            "S4 structured state space model HiPPO",
            sources="arxiv",
            max_results=3,
            http_client_factory=_factory_for(client),
            clock=_zero_clock(),
        )

        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["source"], "arxiv")
        self.assertEqual(client.calls[0]["url"], "https://export.arxiv.org/api/query")
        self.assertFalse(any("status 301" in warning for warning in result["warnings"]))

    def test_search_interleaves_multi_source_results(self) -> None:
        with (
            patch.object(
                literature,
                "_search_arxiv",
                return_value={
                    "records": [
                        {"source": "arxiv", "title": "a1"},
                        {"source": "arxiv", "title": "a2"},
                        {"source": "arxiv", "title": "a3"},
                    ]
                },
            ),
            patch.object(
                literature,
                "_search_openalex",
                return_value={
                    "records": [
                        {"source": "openalex", "title": "o1"},
                        {"source": "openalex", "title": "o2"},
                    ]
                },
            ),
            patch.object(
                literature,
                "_search_pubmed",
                return_value={"records": [{"source": "pubmed", "title": "p1"}]},
            ),
        ):
            result = literature.literature_search(
                "control",
                sources="arxiv,openalex,pubmed",
                max_results=4,
            )

        self.assertEqual(
            [(item["source"], item["title"]) for item in result["results"]],
            [
                ("arxiv", "a1"),
                ("openalex", "o1"),
                ("pubmed", "p1"),
                ("arxiv", "a2"),
            ],
        )

    def test_source_failure_becomes_warning_not_exception(self) -> None:
        client = _FakeClient([_FakeResponse(503, text="down")])

        result = literature.literature_search(
            "x",
            sources="openalex",
            http_client_factory=_factory_for(client),
        )

        self.assertEqual(result["results"], [])
        self.assertTrue(any("openalex" in warning for warning in result["warnings"]))

    def test_unknown_source_and_request_exception_become_warnings(self) -> None:
        result = literature.literature_search(
            "x",
            sources=("unknown", "openalex", ""),
            max_results=50,
            http_client_factory=_raising_factory(),
        )

        self.assertEqual(result["results"], [])
        self.assertEqual(result["sources"], ["unknown", "openalex"])
        self.assertTrue(any("unknown source ignored" in warning for warning in result["warnings"]))
        self.assertTrue(any("network down" in warning for warning in result["warnings"]))

    def test_resolve_arxiv_id(self) -> None:
        client = _FakeClient([_FakeResponse(200, text=_ARXIV_XML)])

        result = literature.literature_resolve(
            "2401.00001v1",
            http_client_factory=_factory_for(client),
            clock=_zero_clock(),
        )

        self.assertEqual(result["record"]["source"], "arxiv")
        self.assertEqual(result["record"]["identifiers"]["arxiv_id"], "2401.00001v1")

    def test_resolve_bad_identifier_returns_error(self) -> None:
        result = literature.literature_resolve("not-an-id")

        self.assertIn("identifier must look like", result["error"])

    def test_resolve_empty_identifier_returns_error(self) -> None:
        result = literature.literature_resolve("")

        self.assertEqual(result["error"], "identifier is required")

    def test_resolve_pmid_openalex_and_doi_records(self) -> None:
        pmid_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "result": {
                            "123456": {
                                "uid": "123456",
                                "title": "PMID title",
                                "pubdate": "2022",
                                "authors": [{"name": "Ada"}],
                                "articleids": "doi:10.1000/pmid",
                            }
                        }
                    },
                )
            ]
        )
        pmid = literature.literature_resolve(
            "123456",
            http_client_factory=_factory_for(pmid_client),
        )
        self.assertEqual(pmid["record"]["identifiers"]["pmid"], "123456")
        self.assertEqual(pmid["record"]["doi"], "10.1000/pmid")

        prefixed_pmid_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "result": {
                            "123456": {
                                "uid": "123456",
                                "title": "Prefixed PMID title",
                                "pubdate": "2022",
                            }
                        }
                    },
                )
            ]
        )
        prefixed_pmid = literature.literature_resolve(
            "PMID: 123456",
            http_client_factory=_factory_for(prefixed_pmid_client),
        )
        self.assertEqual(prefixed_pmid["normalized_identifier"], "123456")
        self.assertEqual(prefixed_pmid["record"]["identifiers"]["pmid"], "123456")
        self.assertEqual(prefixed_pmid_client.calls[0]["params"]["id"], "123456")

        pubmed_url_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "result": {
                            "123456": {
                                "uid": "123456",
                                "title": "PubMed URL title",
                                "pubdate": "2022",
                            }
                        }
                    },
                )
            ]
        )
        pubmed_url = literature.literature_resolve(
            "https://pubmed.ncbi.nlm.nih.gov/123456/",
            http_client_factory=_factory_for(pubmed_url_client),
        )
        self.assertEqual(pubmed_url["normalized_identifier"], "123456")
        self.assertEqual(pubmed_url_client.calls[0]["params"]["id"], "123456")

        with patch.object(
            literature.arxiv_adapter,
            "arxiv_get",
            return_value={
                "paper": {
                    "arxiv_id": "2401.00001",
                    "title": "Arxiv URL title",
                    "authors": ["Ada"],
                    "submitted_date": "2024-01-01",
                    "abstract": "Abstract",
                    "html_url": "https://arxiv.org/abs/2401.00001",
                    "pdf_url": "https://arxiv.org/pdf/2401.00001",
                }
            },
        ) as arxiv_get:
            arxiv_url = literature.literature_resolve("https://arxiv.org/pdf/2401.00001.pdf")
        self.assertEqual(arxiv_url["normalized_identifier"], "2401.00001")
        self.assertEqual(arxiv_url["record"]["identifiers"]["arxiv_id"], "2401.00001")
        self.assertEqual(arxiv_get.call_args.args[0], "2401.00001")

        openalex_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "OA title",
                        "publication_year": 2024,
                        "authorships": [{"author": {"display_name": "Grace"}}],
                    },
                )
            ]
        )
        openalex = literature.literature_resolve(
            "W123",
            http_client_factory=_factory_for(openalex_client),
        )
        self.assertEqual(openalex["record"]["source"], "openalex")
        self.assertEqual(openalex["record"]["authors"], ["Grace"])

        openalex_api_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W123",
                        "display_name": "OA API title",
                        "publication_year": 2024,
                        "authorships": [],
                    },
                )
            ]
        )
        openalex_api = literature.literature_resolve(
            "https://api.openalex.org/works/W123",
            http_client_factory=_factory_for(openalex_api_client),
        )
        self.assertEqual(openalex_api["normalized_identifier"], "W123")
        self.assertEqual(openalex_api_client.calls[0]["url"], "https://api.openalex.org/works/W123")

        crossref_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "message": {
                            "title": ["CR title"],
                            "author": [{"given": "Linus", "family": "T"}],
                            "published-online": {"date-parts": [[2021]]},
                            "container-title": ["Venue"],
                            "URL": "https://doi.org/10.1000/cr",
                            "DOI": "10.1000/cr",
                        }
                    },
                )
            ]
        )
        crossref = literature.literature_resolve(
            "https://doi.org/10.1000/cr",
            http_client_factory=_factory_for(crossref_client),
        )
        self.assertEqual(crossref["record"]["source"], "crossref")
        self.assertEqual(crossref["record"]["authors"], ["Linus T"])

    def test_resolve_source_errors_are_structured(self) -> None:
        for identifier in ("PMID: 123456", "W123", "10.1000/fail"):
            with self.subTest(identifier=identifier):
                client = _FakeClient([_FakeResponse(500, text="bad")])
                result = literature.literature_resolve(
                    identifier,
                    http_client_factory=_factory_for(client),
                )
                self.assertIn("returned status 500", result["error"])

    def test_source_guide_is_task_agnostic(self) -> None:
        result = literature.literature_source_guide("robotics control", "improve policy search")

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["evidence_role"], "research_planning_guidance")
        self.assertTrue(any("arxiv" in source.lower() for source in result["recommended_sources"]))
        self.assertIn("scientific_database_search", result["available_no_key_tools"])
        self.assertIn("existing local resources", " ".join(result["verification_checks"]))
        self.assertIn("do not download", result["resource_policy"])

    def test_scientific_database_search_combines_public_databases(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "resultList": {
                            "result": [
                                {
                                    "title": "Europe PMC paper",
                                    "authorString": "A One, B Two",
                                    "pubYear": "2024",
                                    "source": "MED",
                                    "id": "1",
                                    "pmid": "1",
                                    "pmcid": "PMC1",
                                    "doi": "10.1/pmc",
                                }
                            ]
                        }
                    },
                ),
                _FakeResponse(
                    200,
                    {
                        "results": [
                            {
                                "primaryAccession": "P12345",
                                "uniProtkbId": "PROT_HUMAN",
                                "proteinDescription": {
                                    "recommendedName": {"fullName": {"value": "Protein X"}}
                                },
                                "organism": {"scientificName": "Homo sapiens"},
                            }
                        ]
                    },
                ),
                _FakeResponse(
                    200,
                    {
                        "studies": [
                            {
                                "protocolSection": {
                                    "identificationModule": {
                                        "nctId": "NCT00000001",
                                        "briefTitle": "Trial X",
                                    },
                                    "statusModule": {"overallStatus": "RECRUITING"},
                                    "designModule": {"phases": ["PHASE1"]},
                                    "conditionsModule": {"conditions": ["condition"]},
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        result = literature.scientific_database_search(
            "protein trial",
            sources="europepmc,uniprot,clinicaltrials",
            max_results=5,
            http_client_factory=_factory_for(client),
        )

        self.assertEqual(result["evidence_role"], "contextual_database_signal")
        self.assertIn("current-environment", result["resource_policy"].lower())
        self.assertEqual(result["result_count"], 3)
        self.assertEqual(
            [record["source"] for record in result["results"]],
            ["europepmc", "uniprot", "clinicaltrials"],
        )
        self.assertEqual(client.calls[0]["params"]["format"], "json")
        self.assertEqual(client.calls[1]["params"]["format"], "json")
        self.assertEqual(client.calls[2]["params"]["query.term"], "protein trial")

    def test_scientific_database_search_interleaves_sources(self) -> None:
        with (
            patch.object(
                literature,
                "_search_europepmc",
                return_value={
                    "records": [
                        {"source": "europepmc", "title": "e1"},
                        {"source": "europepmc", "title": "e2"},
                        {"source": "europepmc", "title": "e3"},
                    ]
                },
            ),
            patch.object(
                literature,
                "_search_uniprot",
                return_value={
                    "records": [
                        {"source": "uniprot", "title": "u1"},
                        {"source": "uniprot", "title": "u2"},
                    ]
                },
            ),
            patch.object(
                literature,
                "_search_clinicaltrials",
                return_value={"records": [{"source": "clinicaltrials", "title": "c1"}]},
            ),
        ):
            result = literature.scientific_database_search(
                "protein",
                sources="europepmc,uniprot,clinicaltrials",
                max_results=4,
            )

        self.assertEqual(
            [(item["source"], item["title"]) for item in result["results"]],
            [
                ("europepmc", "e1"),
                ("uniprot", "u1"),
                ("clinicaltrials", "c1"),
                ("europepmc", "e2"),
            ],
        )

    def test_scientific_database_search_handles_non_database_source(self) -> None:
        result = literature.scientific_database_search(
            "x",
            sources="arxiv",
            http_client_factory=_raising_factory(),
        )

        self.assertEqual(result["results"], [])
        self.assertIn("not a scientific database connector", result["warnings"][0])

    def test_scientific_database_search_malformed_and_remote_errors_are_warnings(self) -> None:
        europe_error = literature.scientific_database_search(
            "x",
            sources="europepmc",
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(503, text="down")])),
        )
        self.assertEqual(europe_error["results"], [])
        self.assertTrue(any("returned status 503" in item for item in europe_error["warnings"]))

        malformed_cases = [
            ("europepmc", {"resultList": {"result": {}}}, "europepmc: malformed results"),
            ("uniprot", {"results": {}}, "uniprot: malformed results"),
            ("clinicaltrials", {"studies": {}}, "clinicaltrials: malformed studies"),
        ]
        for source, payload, warning in malformed_cases:
            with self.subTest(source=source):
                result = literature.scientific_database_search(
                    "x",
                    sources=source,
                    http_client_factory=_factory_for(_FakeClient([_FakeResponse(200, payload)])),
                )
                self.assertEqual(result["results"], [])
                self.assertIn(warning, result["warnings"])

        raising = literature.scientific_database_search(
            "x",
            sources=("europepmc", ""),
            http_client_factory=_raising_factory(),
        )
        self.assertEqual(raising["sources"], ["europepmc"])
        self.assertTrue(any("network down" in item for item in raising["warnings"]))

    def test_literature_search_semantic_scholar_and_malformed_source_payloads(self) -> None:
        semantic = literature.literature_search(
            "policy",
            sources="semantic_scholar",
            http_client_factory=_factory_for(
                _FakeClient(
                    [
                        _FakeResponse(
                            200,
                            {
                                "data": [
                                    {
                                        "paperId": "S2",
                                        "title": "Semantic title",
                                        "authors": [{"name": "Ada"}],
                                        "year": 2024,
                                        "abstract": "Summary",
                                        "url": "https://semanticscholar.org/paper/S2",
                                        "venue": "Venue",
                                        "externalIds": {
                                            "DOI": "10.1/s2",
                                            "PubMed": "123",
                                            "ArXiv": "2401.00001",
                                        },
                                        "openAccessPdf": {
                                            "url": "https://arxiv.org/pdf/2401.00001"
                                        },
                                    },
                                    "bad",
                                ]
                            },
                        )
                    ]
                )
            ),
        )
        self.assertEqual(semantic["result_count"], 1)
        self.assertEqual(semantic["results"][0]["source"], "semantic_scholar")
        self.assertTrue(semantic["results"][0]["is_open_access"])

        malformed_cases = [
            ("crossref", {"message": {"items": {}}}, "crossref: malformed items"),
            ("semantic_scholar", {"data": {}}, "semantic_scholar: malformed data"),
        ]
        for source, payload, warning in malformed_cases:
            with self.subTest(source=source):
                result = literature.literature_search(
                    "x",
                    sources=source,
                    http_client_factory=_factory_for(_FakeClient([_FakeResponse(200, payload)])),
                )
                self.assertEqual(result["results"], [])
                self.assertIn(warning, result["warnings"])

    def test_open_access_text_fetches_html_with_provenance(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    text="<html><script>x()</script><body><h1>Title</h1><p>Full text.</p></body></html>",
                    headers={"content-type": "text/html"},
                )
            ]
        )

        result = literature.literature_open_access_text(
            "https://arxiv.org/html/2401.00001",
            max_chars=20,
            http_client_factory=_factory_for(client),
            clock=lambda: 0,
        )

        self.assertEqual(result["evidence_role"], "contextual_literature_signal")
        self.assertEqual(result["retrieval_url"], "https://arxiv.org/html/2401.00001")
        self.assertIn("do not download", result["resource_policy"])
        self.assertIn("Title Full text", result["text"])
        self.assertTrue(result["content_hash"].startswith("sha256:"))
        self.assertEqual(result["provenance"]["credential_scope"], "none")

    def test_open_access_text_records_pdf_without_ocr(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    content=b"%PDF-1.7\ncontent",
                    headers={"content-type": "application/pdf"},
                )
            ]
        )

        result = literature.literature_open_access_text(
            "https://arxiv.org/pdf/2401.00001",
            http_client_factory=_factory_for(client),
            clock=lambda: 0,
        )

        self.assertEqual(result["content_type"], "application/pdf")
        self.assertEqual(result["text"], "")
        self.assertIn("pdf_reader", result["pdf_read_note"])

    def test_open_access_text_resolves_openalex_doi_url(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(500, text="crossref unavailable"),
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W1",
                        "title": "OA Work",
                        "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/html/oa"},
                    },
                ),
                _FakeResponse(
                    200,
                    text="<article>Open text</article>",
                    headers={"content-type": "text/html"},
                ),
            ]
        )

        result = literature.literature_open_access_text(
            "10.1234/oa",
            http_client_factory=_factory_for(client),
        )

        self.assertEqual(result["retrieval_url"], "https://arxiv.org/html/oa")
        self.assertIn("Open text", result["text"])
        self.assertEqual(
            result["resolved"]["record"]["open_access_url"], "https://arxiv.org/html/oa"
        )

    def test_open_access_text_treats_identifier_urls_as_resolvable_ids(self) -> None:
        doi_client = _FakeClient(
            [
                _FakeResponse(500, text="crossref unavailable"),
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W4",
                        "title": "DOI URL OA",
                        "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/html/doi-url"},
                    },
                ),
                _FakeResponse(
                    200,
                    text="<article>DOI URL text</article>",
                    headers={"content-type": "text/html"},
                ),
            ]
        )
        doi_result = literature.literature_open_access_text(
            "https://doi.org/10.1234/doi-url",
            http_client_factory=_factory_for(doi_client),
        )
        self.assertEqual(doi_result["retrieval_url"], "https://arxiv.org/html/doi-url")
        self.assertIn("DOI URL text", doi_result["text"])
        self.assertNotIn("direct URL fetch is limited", doi_result.get("error", ""))

        pubmed_client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    {
                        "result": {
                            "123456": {
                                "uid": "123456",
                                "title": "PMID",
                                "articleids": "doi:10.1234/pubmed-url",
                            }
                        }
                    },
                ),
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W5",
                        "title": "PubMed URL OA",
                        "open_access": {
                            "is_oa": True,
                            "oa_url": "https://arxiv.org/html/pubmed-url",
                        },
                    },
                ),
                _FakeResponse(
                    200,
                    text="<article>PubMed URL text</article>",
                    headers={"content-type": "text/html"},
                ),
            ]
        )
        pubmed_result = literature.literature_open_access_text(
            "https://pubmed.ncbi.nlm.nih.gov/123456/",
            http_client_factory=_factory_for(pubmed_client),
        )
        self.assertEqual(pubmed_result["retrieval_url"], "https://arxiv.org/html/pubmed-url")
        self.assertIn("PubMed URL text", pubmed_result["text"])
        self.assertEqual(pubmed_result["resolved"]["normalized_identifier"], "123456")
        self.assertEqual(
            pubmed_result["resolved"]["open_access_record"]["open_access_url"],
            "https://arxiv.org/html/pubmed-url",
        )
        self.assertNotIn("direct URL fetch is limited", pubmed_result.get("error", ""))

        with patch.object(
            literature.arxiv_adapter,
            "arxiv_get",
            return_value={
                "paper": {
                    "arxiv_id": "2401.00001",
                    "title": "HTTP arXiv",
                    "authors": [],
                    "submitted_date": "2024-01-01",
                    "abstract": "Abstract",
                    "html_url": "https://arxiv.org/abs/2401.00001",
                    "pdf_url": "https://arxiv.org/pdf/2401.00001",
                }
            },
        ):
            arxiv_result = literature.literature_open_access_text(
                "http://arxiv.org/abs/2401.00001",
                http_client_factory=_factory_for(
                    _FakeClient(
                        [
                            _FakeResponse(
                                200,
                                content=b"%PDF-1.7\ncontent",
                                headers={"content-type": "application/pdf"},
                            )
                        ]
                    )
                ),
            )
        self.assertEqual(arxiv_result["retrieval_url"], "https://arxiv.org/pdf/2401.00001")
        self.assertEqual(arxiv_result["content_type"], "application/pdf")

    def test_open_access_text_resolution_and_fetch_error_paths(self) -> None:
        bad_identifier = literature.literature_open_access_text(
            "not-an-id",
            http_client_factory=_raising_factory(),
        )
        self.assertIn("identifier must look like", bad_identifier["error"])

        no_open_access = literature.literature_open_access_text(
            "10.1234/no-oa",
            http_client_factory=_factory_for(
                _FakeClient(
                    [
                        _FakeResponse(500, text="crossref unavailable"),
                        _FakeResponse(500, text="openalex unavailable"),
                    ]
                )
            ),
        )
        self.assertEqual(no_open_access["error"], "no open-access URL found without credentials")
        self.assertTrue(any("primary resolver" in item for item in no_open_access["warnings"]))
        self.assertTrue(any("openalex DOI lookup" in item for item in no_open_access["warnings"]))

        fetch_error = literature.literature_open_access_text(
            "https://arxiv.org/html/2401.00001",
            max_chars="bad",  # type: ignore[arg-type]
            http_client_factory=_factory_for(_FakeClient([_FakeResponse(404, text="missing")])),
        )
        self.assertEqual(fetch_error["error"], "returned status 404")
        self.assertEqual(fetch_error["retrieval_url"], "https://arxiv.org/html/2401.00001")

    def test_open_access_text_rejects_metadata_oa_url_outside_allowlist(self) -> None:
        result = literature.literature_open_access_text(
            "10.1234/oa-final",
            http_client_factory=_factory_for(
                _FakeClient(
                    [
                        _FakeResponse(500, text="crossref unavailable"),
                        _FakeResponse(
                            200,
                            {
                                "id": "https://openalex.org/W2",
                                "title": "OA final",
                                "open_access": {
                                    "is_oa": True,
                                    "oa_url": "https://example.org/open-copy",
                                },
                            },
                        ),
                    ]
                )
            ),
        )

        self.assertEqual(result["error"], "no open-access URL found without credentials")
        self.assertEqual(
            result["resolved"]["record"]["open_access_url"], "https://example.org/open-copy"
        )

    def test_open_access_text_rejects_general_url_fetch(self) -> None:
        localhost = literature.literature_open_access_text(
            "http://127.0.0.1/latest/meta-data",
            http_client_factory=_raising_factory(),
        )
        publisher = literature.literature_open_access_text(
            "https://nature.com/articles/not-necessarily-open",
            http_client_factory=_raising_factory(),
        )

        self.assertIn("direct URL fetch is limited", localhost["error"])
        self.assertIn("direct URL fetch is limited", publisher["error"])

    def test_open_access_text_rejects_redirect_to_non_open_host(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    200,
                    text="<html>redirect target</html>",
                    headers={"content-type": "text/html"},
                    url="https://nature.com/articles/paywalled",
                )
            ]
        )

        result = literature.literature_open_access_text(
            "https://arxiv.org/html/2401.00001",
            http_client_factory=_factory_for(client),
        )

        self.assertIn("redirected to a non-approved", result["error"])
        self.assertEqual(result["retrieval_url"], "https://arxiv.org/html/2401.00001")
        self.assertEqual(result["final_url"], "https://nature.com/articles/paywalled")

    def test_open_access_text_rejects_metadata_oa_redirect_to_non_open_host(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(500, text="crossref unavailable"),
                _FakeResponse(
                    200,
                    {
                        "id": "https://openalex.org/W3",
                        "title": "OA redirect",
                        "open_access": {
                            "is_oa": True,
                            "oa_url": "https://arxiv.org/html/oa-redirect",
                        },
                    },
                ),
                _FakeResponse(
                    200,
                    text="<html>redirect target</html>",
                    headers={"content-type": "text/html"},
                    url="https://nature.com/articles/paywalled",
                ),
            ]
        )

        result = literature.literature_open_access_text(
            "10.1234/oa-redirect",
            http_client_factory=_factory_for(client),
        )

        self.assertIn("redirected to a non-approved", result["error"])
        self.assertEqual(result["retrieval_url"], "https://arxiv.org/html/oa-redirect")
        self.assertNotIn("provenance", result)

    def test_text_fetch_rejects_redirect_to_private_host_before_second_get(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    302,
                    text="",
                    headers={"Location": "https://127.0.0.1/latest/meta-data"},
                )
            ]
        )

        result = literature._fetch_text_or_pdf(
            "https://arxiv.org/html/2401.00001",
            http_client_factory=_factory_for(client),
        )

        self.assertIn("redirected to a non-approved", result["error"])
        self.assertEqual(result["final_url"], "https://127.0.0.1/latest/meta-data")
        self.assertEqual(len(client.calls), 1)

    def test_source_guide_domain_families(self) -> None:
        domains = [
            ("genomics", "biology"),
            ("molecule search", "chemistry"),
            ("clinical trial", "medicine"),
            ("plasma physics", "physics_materials"),
            ("unknown topic", "generic"),
        ]
        for domain, expected_key in domains:
            with self.subTest(domain=domain):
                self.assertEqual(literature._domain_key(domain), expected_key)
                guide = literature.literature_source_guide(domain)
                self.assertEqual(guide["evidence_role"], "research_planning_guidance")

    def test_descriptor_lists_handlers(self) -> None:
        from praxist.core.tool_servers import (
            LITERATURE_LOOKUP_MCP_TOOL_NAMES,
            LITERATURE_LOOKUP_TOOL_NAMES,
            allowed_mcp_tool_names,
            tool_server_for_ref,
        )

        plugin = literature.create_tool_plugin()

        self.assertEqual(plugin["tool_server_ref"], "tool_server:literature_lookup")
        self.assertEqual(plugin["server_name"], "literature-lookup")
        self.assertEqual(tuple(plugin["tool_names"]), LITERATURE_LOOKUP_TOOL_NAMES)
        self.assertEqual(
            tool_server_for_ref("tool_server:literature_lookup").tool_names,
            LITERATURE_LOOKUP_TOOL_NAMES,
        )
        handlers = plugin["handlers"]
        assert isinstance(handlers, dict)
        self.assertEqual(set(handlers), set(LITERATURE_LOOKUP_TOOL_NAMES))
        allowed_names = allowed_mcp_tool_names(
            ["tool_server:literature_lookup"],
            local_mode=True,
            multi_pi_enabled=True,
        )
        for tool_name in LITERATURE_LOOKUP_MCP_TOOL_NAMES:
            self.assertIn(tool_name, allowed_names)

    def test_mcp_server_handlers_use_manifest_validation(self) -> None:
        handlers: dict[str, Any] = {}

        def fake_tool(name: str, _description: str, _schema: dict[str, Any]):
            def _decorator(fn: Any) -> Any:
                handlers[name] = fn
                return {"name": name, "handler": fn}

            return _decorator

        with (
            patch.object(literature, "tool", fake_tool),
            patch.object(
                literature,
                "create_sdk_mcp_server",
                side_effect=lambda name, tools: {"name": name, "tools": tools},
            ),
            patch.object(
                literature,
                "literature_search",
                side_effect=AssertionError("malformed MCP input should not reach lookup"),
            ),
            patch.object(
                literature,
                "scientific_database_search",
                side_effect=AssertionError("malformed MCP input should not reach lookup"),
            ),
        ):
            server = literature.create_literature_lookup_server()

            search = asyncio.run(
                handlers["literature_search"]({"query": "x", "max_results": "bad"})
            )
            database = asyncio.run(
                handlers["scientific_database_search"]({"query": "x", "max_results": "bad"})
            )

        self.assertEqual(server["name"], "literature-lookup")
        search_payload = json.loads(search["content"][0]["text"])
        database_payload = json.loads(database["content"][0]["text"])
        self.assertEqual(search_payload["error"], "max_results must be an integer")
        self.assertEqual(database_payload["error"], "max_results must be an integer")

    def test_codex_mcp_mapping_uses_direct_sdk_configuration(self) -> None:
        from praxist.plugins.agent_runtimes.codex_sdk._mcp import (
            MCP_STDIO_MODULE,
            mcp_configuration,
        )

        result = mcp_configuration([{"server_name": "literature-lookup"}])
        server = result.config["mcp_servers"]["literature-lookup"]

        self.assertEqual(server["command"], sys.executable)
        self.assertEqual(server["args"][:2], ["-m", MCP_STDIO_MODULE])
        self.assertIn(
            "literature_lookup.adapter:create_literature_lookup_server",
            server["args"][2],
        )

    def test_core_direct_handler_executes_source_guide_without_network(self) -> None:
        from praxist.core.tool_servers import execute_legacy_tool_handler

        result = execute_legacy_tool_handler(
            "tool_server:literature_lookup",
            "literature_source_guide",
            {"domain": "chemistry", "objective": "find molecule baselines"},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["evidence_role"], "research_planning_guidance")
        self.assertEqual(result.failover_reason, "none")

    def test_manifest_handlers_return_mcp_text_envelopes(self) -> None:
        search = literature.handle_literature_search({"query": "", "max_results": 1})
        malformed_search = literature.handle_literature_search({"query": "x", "max_results": "bad"})
        resolve = literature.handle_literature_resolve({"identifier": ""})
        guide = literature.handle_literature_source_guide({"domain": "control"})
        open_text = literature.handle_literature_open_access_text({"identifier_or_url": ""})
        database = literature.handle_scientific_database_search({"query": ""})

        payloads = [
            json.loads(item["content"][0]["text"])
            for item in (search, malformed_search, resolve, guide, open_text, database)
        ]
        self.assertEqual(payloads[0]["error"], "query is required")
        self.assertEqual(payloads[1]["result_count"], 0)
        self.assertEqual(payloads[2]["error"], "identifier is required")
        self.assertEqual(payloads[3]["evidence_role"], "research_planning_guidance")
        self.assertIn("do not download", payloads[3]["resource_policy"])
        self.assertEqual(payloads[4]["error"], "identifier_or_url is required")
        self.assertEqual(payloads[5]["error"], "query is required")

    def test_public_lookup_related_tool_manifests_resolve(self) -> None:
        from praxist.core.tool_servers import tool_server_for_ref

        refs = [
            "tool_server:literature_lookup",
            "tool_server:arxiv",
            "tool_server:browser",
            "tool_server:pdf_reader",
            "tool_server:brave_search",
        ]

        resolved = [tool_server_for_ref(ref).server_name for ref in refs]

        self.assertEqual(
            resolved,
            ["literature-lookup", "arxiv", "browser", "pdf-reader", "brave-search"],
        )

    def test_private_fetch_and_normalization_edge_cases(self) -> None:
        valid_text_client = _FakeClient([_FakeResponse(200, payload=None, text='{"ok": true}')])
        valid_text = literature._fetch_json(
            "https://example.test/json",
            http_client_factory=_factory_for(valid_text_client),
        )
        self.assertEqual(valid_text["json"], {"ok": True})

        invalid_text_client = _FakeClient([_FakeResponse(200, payload=None, text="{bad")])
        invalid_text = literature._fetch_json(
            "https://example.test/json",
            http_client_factory=_factory_for(invalid_text_client),
        )
        self.assertIn("invalid JSON", invalid_text["error"])

        self.assertEqual(
            literature._search_openalex(
                "x",
                1,
                http_client_factory=_factory_for(
                    _FakeClient([_FakeResponse(200, {"results": {}})])
                ),
            )["warnings"],
            ["openalex: malformed results"],
        )
        self.assertEqual(
            literature._search_pubmed(
                "x",
                1,
                http_client_factory=_factory_for(
                    _FakeClient([_FakeResponse(200, {"esearchresult": {"idlist": []}})])
                ),
            ),
            {"records": []},
        )
        self.assertEqual(
            literature._fetch_pubmed_summaries(
                ["1"],
                http_client_factory=_factory_for(_FakeClient([_FakeResponse(200, {"result": []})])),
            )["warnings"],
            ["pubmed: malformed summary"],
        )
        with patch.object(literature.arxiv_adapter, "arxiv_search", return_value={"error": "bad"}):
            self.assertEqual(literature._search_arxiv("x", 1), {"error": "bad"})

        self.assertEqual(
            literature._fetch_json(
                "https://example.test/json",
                http_client_factory=_factory_for(_FakeClient([_FakeResponse(418, text="teapot")])),
            )["error"],
            "returned status 418",
        )
        no_content = literature._fetch_text_or_pdf(
            "https://arxiv.org/html/no-content",
            http_client_factory=_factory_for(
                _FakeClient(
                    [
                        _FakeResponse(
                            200,
                            text="fallback text",
                            content=None,
                            headers={"Content-Type": "text/plain"},
                        )
                    ]
                )
            ),
        )
        self.assertEqual(no_content["content_type"], "text/plain")
        self.assertIn("fallback text", no_content["text"])

        non_bytes = literature._fetch_text_or_pdf(
            "https://arxiv.org/html/non-bytes",
            http_client_factory=_factory_for(
                _FakeClient([_FakeResponse(200, text="", content="abc")])  # type: ignore[arg-type]
            ),
        )
        self.assertEqual(non_bytes["text"], "abc")

        too_large = literature._fetch_text_or_pdf(
            "https://arxiv.org/html/large",
            http_client_factory=_factory_for(
                _FakeClient(
                    [
                        _FakeResponse(
                            200,
                            content=b"x" * (literature._MAX_FULL_TEXT_BYTES + 1),
                        )
                    ]
                )
            ),
        )
        self.assertIn("response exceeded", too_large["error"])

        self.assertEqual(
            literature._fetch_text_or_pdf(
                "https://arxiv.org/html/missing",
                http_client_factory=_factory_for(_FakeClient([_FakeResponse(503, text="down")])),
            )["error"],
            "returned status 503",
        )

        landing = literature._normalize_openalex_work(
            {
                "display_name": "Fallback title",
                "primary_location": {"landing_page_url": "https://landing"},
                "abstract_inverted_index": {"B": [1], "A": [0], "skip": ["x"]},
            }
        )
        self.assertEqual(landing["url"], "https://landing")
        self.assertEqual(landing["open_access_url"], "")
        self.assertEqual(landing["abstract"], "A B")
        closed_pdf = literature._normalize_openalex_work(
            {
                "display_name": "Closed PDF",
                "primary_location": {"pdf_url": "https://sciencedirect.com/paper.pdf"},
                "open_access": {"is_oa": False},
            }
        )
        self.assertEqual(closed_pdf["open_access_url"], "")

        self.assertEqual(literature._abstract_from_openalex({}), "")
        self.assertEqual(
            literature._openalex_work_url("https://openalex.org/W55"),
            literature._OPENALEX_WORKS_URL + "/W55",
        )
        self.assertEqual(
            literature._openalex_doi_url("10.1/x"),
            literature._OPENALEX_WORKS_URL + "/https://doi.org/10.1/x",
        )
        self.assertEqual(literature._openalex_landing_page({}), "")
        self.assertEqual(literature._best_open_access_url({"url": "https://u"}), "")
        self.assertEqual(
            literature._best_open_access_url({"open_access_url": "https://arxiv.org/html/x"}),
            "https://arxiv.org/html/x",
        )
        self.assertEqual(
            literature._best_open_access_url(
                {
                    "open_access_url": "https://nature.com/articles/oa",
                    "is_open_access": True,
                }
            ),
            "",
        )
        self.assertEqual(
            literature._best_open_access_url({"open_access_url": "https://nature.com/articles/x"}),
            "",
        )
        self.assertIsNone(literature._extract_doi("no doi"))
        self.assertTrue(literature._looks_like_arxiv_id("hep-th/9901001v2"))
        self.assertTrue(literature._looks_like_openalex_id("https://openalex.org/W123"))

        crossref = literature._normalize_crossref_work(
            {
                "title": "Scalar title",
                "author": ["bad", {"family": "Solo"}],
                "published-print": {"date-parts": [[]]},
                "container-title": "Scalar venue",
                "DOI": "10.1/x",
            }
        )
        self.assertEqual(crossref["title"], "Scalar title")
        self.assertEqual(crossref["authors"], ["Solo"])
        self.assertIsNone(crossref["year"])

        self.assertFalse(literature._is_public_https_url("http://example.org"))
        self.assertFalse(literature._is_public_https_url("https://127.0.0.1/x"))
        self.assertFalse(literature._is_public_https_url("https://localhost/x"))
        self.assertFalse(literature._is_allowed_open_access_url("https://example.org/x"))
        self.assertEqual(
            literature._strip_html_text("<style>x</style><p>A</p><noscript>B</noscript>"), "A"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
