"""arxiv — provider-agnostic search + metadata MCP tool (#128 PR-2).

See :mod:`praxist.plugins.tools.arxiv.adapter` for the three
handlers (``arxiv_search`` / ``arxiv_get`` / ``arxiv_recent``) and
:mod:`praxist.plugins.tools.arxiv.__main__` for the stdio
entrypoint that codex_sdk / claude_sdk peers spawn when ``arxiv`` is
in their ``tool_servers`` list.
"""
