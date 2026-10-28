"""Regressions observed while using the low-cost model; no paid calls."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage

from open_deep_research import evidence_model, evidence_runtime, evidence_tools, bailian_search
from open_deep_research.visual_tools import ImageObservation


@pytest.mark.parametrize('content,finish,valid', [
    ('{"observations":"visible", "uncertainties":""}', 'stop', True),
    ('```json\n{"observations":"visible", "uncertainties":""}\n```', 'stop', True),
    ('{"observations":[], "uncertainties":""}', 'stop', False),
    ('{"observations":"visible", "uncertainties":""}', 'length', False),
    ('Here is my answer, with no structured evidence.', 'stop', False),
])
def test_content_json_uses_same_schema_without_retry(monkeypatch, content, finish, valid):
    calls=[]
    class Model:
        def __init__(self, **kwargs): pass
        def with_structured_output(self, *args, **kwargs): return self
        async def ainvoke(self, *args, **kwargs):
            calls.append(1)
            return {'raw': AIMessage(content=content, response_metadata={'finish_reason':finish}),
                    'parsed':None, 'parsing_error':None}
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    monkeypatch.setattr(evidence_model,'ChatOpenAI',Model)
    run=SimpleNamespace(reserve=lambda *args:None,log=lambda *args,**kwargs:None)
    infer=evidence_model.make_infer(run, {'configurable':{'research_model':'openai:qwen3.5-flash'}},
                                   schema=ImageObservation)
    if valid:
        assert json.loads(asyncio.run(infer('image','question')))['observations']=='visible'
    else:
        with pytest.raises(ValueError,match='structured'):
            asyncio.run(infer('image','question'))
    assert len(calls)==1


def test_atomic_replace_sharing_violation_has_bounded_retry(monkeypatch, tmp_path):
    original=Path.replace
    calls=[]
    def replace(path, target):
        calls.append(1)
        if len(calls)<3: raise PermissionError('temporary sharing violation')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',replace)
    monkeypatch.setattr(evidence_runtime.time,'sleep',lambda _:None)
    source=tmp_path/'file.tmp';source.write_text('new')
    target=tmp_path/'file.json';target.write_text('old')
    evidence_runtime.replace_file(source,target)
    assert target.read_text()=='new' and len(calls)==3
    def deny(*args): raise PermissionError('permanent denial')
    monkeypatch.setattr(Path,'replace',deny)
    with pytest.raises(PermissionError): evidence_runtime.replace_file(source,target)


def test_guessed_source_is_rejected_before_network(monkeypatch):
    async def fetch(*args): raise AssertionError('No HTTP for guessed URLs')
    monkeypatch.setattr(evidence_tools,'fetch',fetch)
    token=evidence_runtime.ACTIVE_RUN.set(SimpleNamespace(source_urls=set()))
    try:
        with pytest.raises(ValueError,match='do not guess'):
            asyncio.run(evidence_tools.archive_page('https://docs.langchain.com/made-up'))
    finally:
        evidence_runtime.ACTIVE_RUN.reset(token)


def test_qwen_flash_search_endpoint_scope_and_empty_result(monkeypatch,tmp_path):
    run=evidence_runtime.EvidenceRun(tmp_path/'run',tmp_path/'budget.json')
    run.question='Use docs.langchain.com only';run.source_urls=set()
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0}))
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    monkeypatch.setenv('OPENAI_API_BASE','https://dashscope.aliyuncs.com/compatible-mode/v1')
    requests=[]
    def respond(request):
        requests.append(request)
        body=json.loads(request.content)
        assert request.url.path.endswith('/text-generation/generation')
        assert body['model']=='qwen-flash'
        assert isinstance(body['input']['messages'][0]['content'],str)
        assert body['parameters']['search_options']['assigned_site_list']==['docs.langchain.com']
        return httpx.Response(200,json={'output':{'search_info':{'search_results':[]},
            'choices':[{'message':{'content':'No indexed source'},'finish_reason':'stop'}]},
            'usage':{'input_tokens':20,'output_tokens':4}})
    client=httpx.AsyncClient
    monkeypatch.setattr(bailian_search.httpx,'AsyncClient',
        lambda **kwargs:client(**kwargs,transport=httpx.MockTransport(respond)))
    token=evidence_runtime.ACTIVE_RUN.set(run)
    try:
        result=asyncio.run(bailian_search.bailian_search_async('query',{'configurable':{
            'bailian_search_model':'openai:qwen-flash'}}))
    finally:
        evidence_runtime.ACTIVE_RUN.reset(token)
    assert result['status']=='insufficient' and not result['sources']
    assert len(requests)==1
    assert 'search_result' in (run.directory/'events.jsonl').read_text()
