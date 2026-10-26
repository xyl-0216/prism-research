"""Offline integration checks. Model/search boundaries are fixtures, not benchmarks."""
import asyncio
import hashlib
import json

import pytest

from open_deep_research.mmsearch_adapter import staged_search, parse_selection, image_block


@pytest.mark.parametrize('value', [
    '{"selected_ids":[0]}', '{"selected_ids":[3]}', '{"selected_ids":[1,1]}',
    '{"selected_ids":[true]}', '{"selected_ids":[]}', '{}', 'not json',
    '{"selected_ids":[1,2]}',
])
def test_bad_selection(value):
    with pytest.raises(ValueError):
        parse_selection(value, 2, 1)


def test_pipeline_preserves_ranking_and_originals(tmp_path):
    path = tmp_path/'fixture.png'
    raw = b'\x89PNG\r\n\x1a\nfixture'
    path.write_bytes(raw)
    asset = {'path':str(path), 'sha256':hashlib.sha256(raw).hexdigest()}
    events, calls, reads = [], [], []

    async def infer(stage, prompt, images):
        calls.append((stage, prompt, images))
        return {'requery':'chart conditions', 'rerank':'{"selected_ids":[2]}',
                'summarize':'Evidence: https://example.test/b'}[stage]

    async def search(query):
        assert query == 'chart conditions'
        return [{'url':'https://example.test/a', 'title':'A'},
                {'url':'https://example.test/a', 'title':'duplicate'},
                {'url':'https://example.test/b', 'title':'B', 'screenshot':asset}]

    async def read(url):
        reads.append(url)
        return {'id':'original-b', 'url':url, 'text':'Original source, not search summary'}

    result = asyncio.run(staged_search('What does this chart show?', asset,
        infer=infer, search=search, read=read, emit=lambda name, **kw:events.append((name, kw))))
    assert reads == ['https://example.test/b']
    assert [c[0] for c in calls] == ['requery','rerank','summarize']
    assert calls[0][2][0]['type'] == 'image_url'
    assert sum(x['type'] == 'image_url' for x in calls[1][2]) == 2
    assert 'Original source, not search summary' in calls[2][1]
    assert result['sources'][0]['id'] == 'original-b'
    assert events[1][1]['candidate_screenshots'] == 1
    path.write_bytes(raw+b'changed')
    with pytest.raises(ValueError, match='hash'):
        image_block(asset)


def test_empty_candidates_do_not_generate_answer():
    stages = []
    async def infer(stage, *_):
        stages.append(stage)
        return 'query'
    async def search(_): return []
    async def read(_): raise AssertionError('must not read')
    with pytest.raises(ValueError, match='No eligible'):
        asyncio.run(staged_search('Question', None, infer=infer, search=search,
            read=read, emit=lambda *a, **k:None))
    assert stages == ['requery']


def test_graph_tool_registration():
    from open_deep_research.utils import get_all_tools
    config = {'configurable':{'mmsearch_enabled':True, 'evidence_profile':'nvidia',
                             'search_api':'bailian'}}
    names = [t.name for t in asyncio.run(get_all_tools(config))]
    assert 'multimodal_evidence_search' in names
    assert 'bailian_web_search' not in names
    assert 'inspect_source_image' in names


def test_live_adapter_budget_and_artifact(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    from open_deep_research import mmsearch_adapter as adapter, bailian_search, evidence_tools
    from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
    ledger = tmp_path/'budget.json'
    ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0}), 'utf-8')
    run = EvidenceRun(tmp_path/'run', ledger)
    replies = iter(['query', '{"selected_ids":[1]}', 'Answer from source'])
    class Client:
        def __init__(self, **kw):
            assert kw['max_retries'] == 0
            assert kw['max_tokens'] == 1600
        async def ainvoke(self, messages, config):
            assert isinstance(messages[0], HumanMessage)
            assert config['callbacks'] == []
            return AIMessage(content=next(replies), response_metadata={'finish_reason':'stop'},
                             usage_metadata={'input_tokens':10,'output_tokens':5,'total_tokens':15})
    async def search(query, config):
        return {'sources':[{'url':'https://www.nvidia.com/test','title':'Test'},
                           {'url':'https://127.0.0.1/private','title':'blocked'}]}
    async def read(url): return {'id':'page-1','url':url,'text':'Original'}
    monkeypatch.setattr(adapter, 'ChatOpenAI', Client)
    monkeypatch.setattr(bailian_search, 'bailian_search_async', search)
    monkeypatch.setattr(evidence_tools, 'archive_page', read)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fixture')
    token = ACTIVE_RUN.set(run)
    try:
        result = json.loads(asyncio.run(adapter.multimodal_evidence_search.ainvoke(
            {'question':'Question'}, {'configurable':{'research_model':'openai:fixture'}})))
    finally:
        ACTIVE_RUN.reset(token)
    assert len(result['sources']) == 1
    assert json.loads(ledger.read_text())['requests'] == 3
    assert (run.directory/'mmsearch_result.json').exists()
    events = [json.loads(line) for line in (run.directory/'events.jsonl').read_text('utf-8').splitlines()]
    assert len([e for e in events if e['event'] == 'request_reserved']) == 3
    assert all(e['usage']['total_tokens'] == 15 for e in events if e['event'] == 'mmsearch_model')


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_langchain_transport_and_inherited_budget(tmp_path, monkeypatch, finish):
    """Exercise real ChatOpenAI serialization with HTTP replaced, including callbacks."""
    import httpx
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_openai import ChatOpenAI
    from open_deep_research import mmsearch_adapter as adapter, bailian_search, evidence_tools
    from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
    ledger = tmp_path/'budget.json'
    ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0}), 'utf-8')
    run = EvidenceRun(tmp_path/'run', ledger)
    image = tmp_path/'fixture.png'
    raw = b'\x89PNG\r\n\x1a\nfixture'
    image.write_bytes(raw)
    run.images['fixture'] = {'path':str(image), 'sha256':hashlib.sha256(raw).hexdigest()}
    requests = []
    replies = iter(['query', '{"selected_ids":[1]}', 'Answer'])
    class OuterBudget(AsyncCallbackHandler):
        async def on_chat_model_start(self, *a, **kw):
            run.reserve('unexpected_duplicate_budget', 0, 1)
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body['messages'][0]['content'][1]['type'] == 'image_url'
        return httpx.Response(200, json={'id':'fixture','object':'chat.completion',
            'created':0, 'model':'fixture', 'choices':[{'index':0,
            'message':{'role':'assistant','content':next(replies)}, 'finish_reason':finish}],
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
    async def search(*a): return {'sources':[{'url':'https://www.nvidia.com/test','title':'Test'}]}
    async def read(url): return {'id':'page-1','url':url,'text':'Original'}
    monkeypatch.setattr(bailian_search, 'bailian_search_async', search)
    monkeypatch.setattr(evidence_tools, 'archive_page', read)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fixture')
    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(adapter, 'ChatOpenAI',
                lambda **kw: ChatOpenAI(**kw, http_async_client=client))
            return await adapter.multimodal_evidence_search.ainvoke(
                {'question':'Question','asset_id':'fixture'},
                {'callbacks':[OuterBudget()], 'configurable':{'research_model':'openai:fixture'}})
    token = ACTIVE_RUN.set(run)
    try:
        if finish == 'length':
            with pytest.raises(ValueError, match='Incomplete MMSearch stage'):
                asyncio.run(execute())
            assert not (run.directory/'mmsearch_result.json').exists()
        else:
            assert json.loads(asyncio.run(execute()))['answer'] == 'Answer'
    finally:
        ACTIVE_RUN.reset(token)
    assert len(requests) == (3 if finish == 'stop' else 1)
    assert json.loads(ledger.read_text())['requests'] == len(requests)
