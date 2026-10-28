"""Failures found during bounded live smoke runs, replayed offline."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from open_deep_research import evidence_model, deep_researcher
from open_deep_research.mcsearch_adapter import parse_plan
from open_deep_research.visual_tools import ImageObservation


@pytest.mark.parametrize('text', [
    '<Sub-Question>\nFind conditions\n<Search>\nLocal Text Retrieval: conditions\n<Dependencies>',
    '<Sub-Question>Find conditions</Sub-Question>\n<Search>Local Text Retrieval: conditions</Search>',
    '```xml\n<Sub-Question>\nFind conditions\n<Search>\nLocal Text Retrieval: conditions\n```',
])
def test_presentation_variations_do_not_break_action(text):
    plan = parse_plan(text)
    assert plan['retrieval_query'] == 'conditions' and plan['depends_on'] == []


def test_malformed_dependencies_are_not_silently_dropped():
    with pytest.raises(ValueError):
        parse_plan('<Sub-Question>Q\n<Search>Local Text Retrieval: q\n<Dependencies>step-99')


@pytest.mark.parametrize('valid', [True, False])
def test_image_schema_is_bound_and_parsing_failure_rejected(monkeypatch, valid):
    reservations = []
    class Model:
        def __init__(self, **kwargs): pass
        def with_structured_output(self, schema, **kwargs):
            assert schema is ImageObservation
            assert kwargs == {'include_raw': True, 'method': 'function_calling'}
            return self
        async def ainvoke(self, messages, config):
            assert config['callbacks'] == []
            return {'raw': AIMessage(content='', response_metadata={'finish_reason': 'tool_calls'}),
                    'parsed': ImageObservation(observations='Visible limits', uncertainties='') if valid else None,
                    'parsing_error': None if valid else ValueError('type mismatch')}
    monkeypatch.setattr(evidence_model, 'ChatOpenAI', Model)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fixture')
    run = SimpleNamespace(reserve=lambda *a:reservations.append(a), log=lambda *a, **kw:None)
    infer = evidence_model.make_infer(run, {'configurable':{'research_model':'openai:qwen3.5-test'}},
                                      schema=ImageObservation)
    if valid:
        assert 'Visible limits' in asyncio.run(infer('image_observation', 'Read image'))
    else:
        with pytest.raises(ValueError, match='structured'):
            asyncio.run(infer('image_observation', 'Read image'))
    assert len(reservations) == 1


def test_mc_supervisor_only_exposes_single_delegation(monkeypatch):
    class Model:
        def bind_tools(self, tools, **kwargs):
            assert tools == [deep_researcher.ConductResearch]
            assert kwargs == {'tool_choice':'ConductResearch', 'parallel_tool_calls':False}
            return self
        def with_retry(self, **kwargs): return self
        def with_config(self, *args, **kwargs): return self
        async def ainvoke(self, messages):
            assert len([m for m in messages if isinstance(m,SystemMessage)]) == 1
            assert 'old conflicting instructions' not in str(messages)
            return AIMessage(content='', tool_calls=[{'name':'ConductResearch',
                'args':{'research_topic':'question'}, 'id':'delegate'}])
    monkeypatch.setattr(deep_researcher, 'configurable_model', Model())
    result = asyncio.run(deep_researcher.supervisor({'supervisor_messages':[
        SystemMessage(content='old conflicting instructions'), HumanMessage(content='question')]},
        {'configurable':{'mcsearch_enabled':True}}))
    assert result.goto == 'supervisor_tools'


def test_filtered_search_returns_insufficient_without_fabricating_sources(tmp_path, monkeypatch):
    from open_deep_research import mmsearch_adapter as mm, bailian_search as bs
    from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
    from open_deep_research.mcsearch_adapter import evidence_chain
    run = EvidenceRun(tmp_path/'run', tmp_path/'budget.json')
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
                                      'input_chars':0,'reserved_output_tokens':0}))
    class Model:
        def __init__(self, **kwargs): pass
        async def ainvoke(self, messages, config):
            assert config['tags'][-1] == 'requery'
            return AIMessage(content='query', response_metadata={'finish_reason':'stop'})
    async def search(*args):
        return {'sources':[{'url':'https://blog.example.test/article','title':'not permitted'}]}
    monkeypatch.setattr(mm, 'ChatOpenAI', Model)
    monkeypatch.setattr(bs, 'bailian_search_async', search)
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    replies = iter(['<Sub-Question>Find source\n<Search>Local Text Retrieval: query',
                    '<End>\nFinal Answer: No eligible evidence is available'])
    async def infer(stage, prompt):
        assert stage == 'plan'  # No evidence: do not pay for an assessment.
        return next(replies)
    async def execute(*args):
        value = json.loads(await mm.multimodal_evidence_search.ainvoke({'question':'query'}))
        assert value['status'] == 'insufficient' and not value['sources']
        return value['sources']
    token = ACTIVE_RUN.set(run)
    try:
        trace = asyncio.run(evidence_chain('question', infer=infer, execute=execute,
            save=lambda t:None, image_available=False))
    finally:
        ACTIVE_RUN.reset(token)
    assert trace['status'] == 'incomplete'
    assert trace['steps'][0]['status'] == 'insufficient'
    assert (run.directory/'mmsearch_result.json').exists()
    from open_deep_research.reporting import generate_report
    run.mcsearch_trace = trace
    asyncio.run(generate_report(run, {}))
    report = json.loads((run.directory/'report.json').read_text(encoding='utf-8'))
    assert not report['findings'] and report['unknowns']


def test_invalid_action_is_saved_for_debugging():
    from copy import deepcopy
    from open_deep_research.mcsearch_adapter import evidence_chain
    snapshots = []
    async def infer(*args): return 'invalid action fixture'
    async def execute(*args): raise AssertionError('Must not execute')
    with pytest.raises(ValueError, match='protocol'):
        asyncio.run(evidence_chain('question', infer=infer, execute=execute,
            save=lambda t:snapshots.append(deepcopy(t)), image_available=False))
    assert snapshots[-1]['conversation'][-1]['content'] == 'invalid action fixture'
    assert snapshots[-1]['status'] == 'failed'
