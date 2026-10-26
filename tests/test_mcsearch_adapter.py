"""Offline checks of incremental planning and actual supervisor/tool wiring."""
import asyncio
import json

import pytest

from open_deep_research.mcsearch_adapter import evidence_chain, trace_note, parse_plan


def plan(question='Inspect footnote', modality='image', dependencies=None):
    return {'action':'research','subquestion':question,'modality':modality,
            'depends_on':dependencies or []}


def assessment(status='answered', ids=None):
    return {'status':status, 'intermediate_answer':'Frame generation is enabled',
            'evidence_ids':['image:chart'] if ids is None else ids,
            'missing_evidence':'' if status=='answered' else 'Need matching independent test'}


def drive(responses, *, execute=None, image_available=True, max_steps=3):
    prompts, snapshots, tools = [], [], []
    replies = iter(responses)
    async def infer(stage, prompt):
        prompts.append((stage,prompt))
        value = next(replies)
        return value if isinstance(value,str) else json.dumps(value)
    async def default_execute(modality, question):
        tools.append(modality)
        return [{'id':'image:chart' if modality=='image' else 'page-1','text':'Original'}]
    async def run():
        return await evidence_chain('Question', infer=infer, execute=execute or default_execute,
            save=lambda t:snapshots.append(json.loads(json.dumps(t))),
            image_available=image_available, max_steps=max_steps)
    return run, prompts, snapshots, tools


def test_incremental_cross_modal_chain():
    run, prompts, snapshots, tools = drive([plan(), assessment(),
        plan('Find matched tests','text',['step-1']), assessment(ids=['page-1']),
        {'action':'stop','reason':'All planned questions answered'}])
    result = asyncio.run(run())
    assert tools == ['image','text']
    assert result['status']=='completed'
    assert result['steps'][1]['depends_on']==['step-1']
    assert 'Frame generation is enabled' in prompts[2][1]
    assert snapshots[1]['steps'][0]['status']=='running'
    assert 'page-1' in trace_note(result)


def test_upstream_protocol_drives_real_query_dispatch():
    run, _, _, tools = drive([
        '<Sub-Question>\nInspect footnote\n<Search>\nLocal Image Retrieval with Input Image\n<Dependencies>\n[]',
        assessment(),
        '<Sub-Question>\nFind matching tests\n<Search>\nLocal Text Retrieval: matched settings\n<Dependencies>\n["step-1"]',
        assessment(ids=['page-1']), '<End>\nFinal Answer: Evidence collected',
    ])
    trace = asyncio.run(run())
    assert trace['status']=='completed'
    assert tools==['image','text']
    assert trace['steps'][0]['upstream_action']=='img_input'
    assert trace['steps'][1]['retrieval_query']=='matched settings'


@pytest.mark.parametrize('text',[
    '<Sub-Question>\nQ\n<Search>\nLocal Image Retrieval with Text Query: cat\n<Dependencies>\n[]',
    '<Sub-Question>\nQ\n<Search>\nNo Retrieval\n<Dependencies>\n[]',
    '<Sub-Question>\nQ\n<Search>\nLocal Text Retrieval: q\nLocal Image Retrieval with Input Image\n<Dependencies>\n[]',
    'Final Answer: not framed',
])
def test_upstream_parser_is_wrapped_by_strict_capability_checks(text):
    with pytest.raises(ValueError): parse_plan(text)


def test_gap_reaches_next_planner_and_is_not_answered():
    run, prompts, _, _ = drive([plan(), assessment('insufficient'),
        {'action':'stop','reason':'Need unavailable independent measurements'}])
    result = asyncio.run(run())
    assert 'Need matching independent test' in prompts[2][1]
    assert result['status']=='incomplete'


@pytest.mark.parametrize('decision', [plan(modality='video'),
    plan(dependencies=['step-99']), {'action':'stop','reason':'No work needed'}])
def test_invalid_initial_plan_fails_closed(decision):
    run, _, snapshots, tools = drive([decision])
    with pytest.raises(ValueError): asyncio.run(run())
    assert snapshots[-1]['status']=='failed'
    assert not tools


def test_fabricated_evidence_is_rejected():
    run, _, snapshots, _ = drive([plan(), assessment(ids=['invented'])])
    with pytest.raises(ValueError, match='fabricated'): asyncio.run(run())
    assert snapshots[-1]['steps'][0]['status']=='failed'


def test_insufficient_step_cannot_be_dependency():
    run, _, _, tools = drive([plan(), assessment('insufficient'),
        plan('Next', 'text', ['step-1'])])
    with pytest.raises(ValueError, match='dependency'): asyncio.run(run())
    assert tools==['image']


def test_step_limit_preserves_unfinished_status():
    run, _, _, _ = drive([plan(), assessment()], max_steps=1)
    assert asyncio.run(run())['status']=='budget_exhausted'


def test_no_sources_remains_insufficient():
    async def empty(*args): return []
    run, prompts, _, _ = drive([plan(), {'action':'stop','reason':'No source'}], execute=empty)
    result = asyncio.run(run())
    assert result['steps'][0]['status']=='insufficient'
    assert [s for s,_ in prompts]==['plan','plan']


def test_no_second_text_search():
    run, _, _, tools = drive([plan('First','text'), assessment(ids=['page-1']),
        plan('Second','text')])
    with pytest.raises(ValueError, match='modality'): asyncio.run(run())
    assert tools==['text']


def test_actual_adapter_persists_trace_and_routes_tools(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from open_deep_research import mcsearch_adapter as mc, evidence_tools, mmsearch_adapter
    from open_deep_research.evidence_runtime import ACTIVE_RUN, EvidenceRun
    ledger = tmp_path/'budget.json'
    ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0}), 'utf-8')
    run = EvidenceRun(tmp_path/'run', ledger)
    run.images['nvidia_chart'] = {'id':'nvidia_chart'}
    responses = iter([plan(), assessment(ids=['image:nvidia_chart']),
        plan('Read original','text',['step-1']), assessment(ids=['page-1']),
        {'action':'stop','reason':'Done'}])
    calls = []
    class Model:
        def __init__(self, **kw): pass
        async def ainvoke(self, messages, config):
            assert config['callbacks']==[]
            return AIMessage(content=json.dumps(next(responses)),response_metadata={'finish_reason':'stop'})
    async def image(args, config):
        calls.append(('image',args))
        return json.dumps({'asset_id':'nvidia_chart','image_url':'https://example.test/image',
            'image_sha256':'fixture','observations':'Footnote'})
    async def text(args, config):
        calls.append(('text',args))
        return json.dumps({'sources':[{'id':'page-1','url':'https://example.test/page','text':'Original'}],
                           'answer':'Do not use generated text as original evidence'})
    monkeypatch.setattr(mc,'ChatOpenAI',Model)
    monkeypatch.setattr(evidence_tools,'inspect_source_image',SimpleNamespace(ainvoke=image))
    monkeypatch.setattr(mmsearch_adapter,'multimodal_evidence_search',SimpleNamespace(ainvoke=text))
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    token = ACTIVE_RUN.set(run)
    try:
        trace = asyncio.run(mc.run_mc_research('Question',{'configurable':{
            'mmsearch_enabled':True,'evidence_profile':'nvidia','search_api':'bailian'}}))
    finally:
        ACTIVE_RUN.reset(token)
    assert [c[0] for c in calls]==['image','text']
    assert trace['status']=='completed'
    assert json.loads(ledger.read_text())['requests']==5
    assert json.loads((run.directory/'mcsearch_trace.json').read_text('utf-8'))==trace


def test_supervisor_delegates_to_mc_controller(monkeypatch):
    from langchain_core.messages import AIMessage
    from open_deep_research import deep_researcher as dr, mcsearch_adapter as mc
    async def research(question, config):
        assert question=='Full question'
        return {'steps':[], 'status':'incomplete','stop_reason':'fixture gap'}
    monkeypatch.setattr(mc,'run_mc_research',research)
    message = AIMessage(content='',tool_calls=[{'id':'call-1','name':'ConductResearch',
        'args':{'research_topic':'Substitute question'}}])
    command = asyncio.run(dr.supervisor_tools({'supervisor_messages':[message],
        'research_brief':'Full question'}, {'configurable':{'mcsearch_enabled':True}}))
    assert command.goto=='__end__'
    assert 'fixture gap' in command.update['notes'][0]
