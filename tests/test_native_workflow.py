"""Default workflow routing and shared execution boundaries (no API calls)."""
import asyncio
from types import SimpleNamespace

import pytest

from open_deep_research import cli, deep_researcher, utils
from open_deep_research.evidence_runtime import ACTIVE_RUN, BudgetExceeded


@pytest.mark.parametrize('option,expected', [([], 'native'), (['--planner', 'mcsearch'], 'mcsearch')])
def test_cli_planner_selection(monkeypatch, tmp_path, option, expected):
    async def research(args):
        assert args.planner == expected
        assert args.model == 'openai:qwen3.5-flash'
        assert args.max_requests == 16
        return tmp_path / 'report.html'
    monkeypatch.setattr(cli, 'research', research)
    monkeypatch.setattr('sys.argv', ['prism-research', 'research', '--question', 'test', '--image', 'fixture.png', *option])
    cli.main()


@pytest.mark.parametrize('images,documents,expected', [
    ({'image': {}}, {}, {'inspect_registered_image'}),
    ({}, {'pdf': {}}, {'search_document_evidence'}),
])
def test_native_tools_match_registered_inputs(monkeypatch, images, documents, expected):
    async def no_mcp(*args): return []
    monkeypatch.setattr(utils, 'load_mcp_tools', no_mcp)
    token = ACTIVE_RUN.set(SimpleNamespace(images=images, documents=documents, allow_web=False))
    try:
        tools = asyncio.run(utils.get_all_tools({'configurable': {
            'evidence_profile': 'general', 'search_api': 'none', 'mmsearch_enabled': False}}))
    finally:
        ACTIVE_RUN.reset(token)
    names = {tool.name for tool in tools}
    assert names & {'inspect_registered_image', 'search_document_evidence', 'read_source_pages'} == expected
    assert 'multimodal_evidence_search' not in names


def test_budget_exhaustion_cannot_be_converted_to_tool_observation():
    class Tool:
        name = 'fixture'
        async def ainvoke(self, *args): raise BudgetExceeded('fixture budget exhausted')
    events = []
    token = ACTIVE_RUN.set(SimpleNamespace(log=lambda event, **data: events.append((event, data))))
    try:
        with pytest.raises(BudgetExceeded):
            asyncio.run(deep_researcher.execute_tool_safely(Tool(), {}, {}))
    finally:
        ACTIVE_RUN.reset(token)
    assert events[0][0] == 'tool_failed'
    assert events[0][1]['error_type'] == 'BudgetExceeded'
