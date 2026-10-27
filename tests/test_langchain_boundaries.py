"""Check model migration without real API calls or credentials."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from open_deep_research import evidence_tools, bailian_search


@pytest.mark.parametrize('finish,content', [('stop', 'visible legend'), ('length', 'partial'), ('stop', '')])
def test_image_langchain_boundary(tmp_path, monkeypatch, finish, content):
    path = tmp_path / 'image.jpg'
    raw = b'\xff\xd8fixture'
    path.write_bytes(raw)
    reservations, calls = [], []
    run = SimpleNamespace(
        images={'chart': {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'url': 'https://example.test/chart.jpg', 'source_page': 'https://example.test'}},
        reserve=lambda *args: reservations.append(args), log=lambda *args, **kwargs: None,
    )
    class FakeModel:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0

        async def ainvoke(self, messages, config):
            calls.append(messages)
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content[1]['type'] == 'image_url'
            assert config['callbacks'] == []
            return AIMessage(content=content, response_metadata={'finish_reason': finish})

    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fixture')
    monkeypatch.setattr(evidence_tools, 'active_run', lambda: run)
    monkeypatch.setattr(evidence_tools, 'ChatOpenAI', FakeModel)
    monkeypatch.setattr(evidence_tools.Configuration, 'from_runnable_config',
                        lambda config: SimpleNamespace(research_model='openai:qwen3.5-plus'))

    async def invoke():
        return await evidence_tools.inspect_source_image.ainvoke({'asset_id': 'chart', 'question': 'Read legend'})

    if finish == 'stop' and content:
        result = json.loads(asyncio.run(invoke()))
        assert result['observations'] == content
        assert len(run.visual_observations) == 1
    else:
        with pytest.raises(ValueError):
            asyncio.run(invoke())
        assert not hasattr(run, 'visual_observations')
    assert len(reservations) == len(calls) == 1


def test_search_runnable_preserves_sources_and_config(monkeypatch):
    expected = {'sources': [{'url': 'https://example.test'}], 'summary': 'evidence'}

    async def transport(query, config):
        assert query == 'question'
        assert config['configurable']['marker'] == 'fixture'
        return expected

    monkeypatch.setattr(bailian_search, 'bailian_search_runnable', RunnableLambda(transport))
    result = asyncio.run(bailian_search.bailian_search_async('question', {'configurable': {'marker': 'fixture'}}))
    assert result == expected
