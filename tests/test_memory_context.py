"""Persistence, context continuity, trust boundaries and budget tests. No network."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_deep_research import cli, deep_researcher, context_management as cm
from open_deep_research.evidence_runtime import ACTIVE_RUN, EvidenceRun, BudgetExceeded
from open_deep_research.long_term_memory import MemoryStore, recall_context, save_episode


def run_fixture(tmp_path):
    run=EvidenceRun(tmp_path/'run',tmp_path/'run'/'budget.json',secrets=['SECRET_FIXTURE'])
    run.question='PDF 实验分辨率与帧生成条件'
    return run


def history(pairs=7, start=0, content_size=1000):
    messages=[]
    for i in range(start,start+pairs):
        messages += [AIMessage(content='',tool_calls=[{'name':'read','args':{'url':f'url-{i}'},'id':f't-{i}'}]),
                     ToolMessage(content=f'page-{i}: condition still uncertain. '+('x'*content_size),tool_call_id=f't-{i}')]
    return messages


def assert_pairs(messages):
    pending=set()
    for m in messages:
        if isinstance(m,AIMessage): pending.update(c['id'] for c in m.tool_calls)
        if isinstance(m,ToolMessage):
            assert m.tool_call_id in pending
            pending.remove(m.tool_call_id)
    assert not pending


def test_compaction_pins_task_preserves_pairs_caches_and_keeps_raw_history(tmp_path):
    run=run_fixture(tmp_path)
    manager=cm.ContextCompressor(run,trigger=3000,keep=3)
    messages=[SystemMessage(content='Follow current rules'),HumanMessage(content='Original question'),*history()]
    original=deepcopy(messages)
    prompts=[]
    async def summarize(prompt):
        prompts.append(prompt)
        return '已读取 page-0 等来源；实验条件仍不确定，需保留证据缺口。'
    async def check():
        result=await manager.prepare(messages,scope='test',model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
        assert result[:2]==messages[:2]
        assert len(prompts)==1 and cm.estimate_tokens(result)<cm.estimate_tokens(messages)
        assert_pairs(result)
        same=await manager.prepare(messages,scope='test',model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
        assert same==result and len(prompts)==1
        again=await manager.prepare(messages+history(1,start=7),scope='test',
            model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
        assert len(prompts)==1
        assert_pairs(again)
        # A later summary includes the previous summary, rather than forgetting it.
        extended=messages+history(7,start=7)
        twice=await manager.prepare(extended,scope='test',model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
        assert len(prompts)==2 and '实验条件仍不确定' in prompts[-1]
        assert_pairs(twice)
    asyncio.run(check())
    assert messages==original
    assert len(manager.archives)==2
    first=next(iter(manager.archives.values())).read_text('utf-8')
    assert 'page-0' in first


def test_multimodal_accounting_and_archive_never_serializes_image_bytes(tmp_path):
    block={'type':'image_url','image_url':{'url':'data:image/png;base64,SECRET_FIXTURE'}}
    messages=[HumanMessage(content=[{'type':'text','text':'Read image'},block])]
    assert cm.estimate_tokens(messages)>=2000
    assert 'SECRET_FIXTURE' not in json.dumps(cm.export_messages(messages))
    assert cm.fingerprint(messages)!=cm.fingerprint([HumanMessage(content='Read image')])


def test_parallel_tool_results_never_split(tmp_path):
    run=run_fixture(tmp_path); manager=cm.ContextCompressor(run,trigger=1000,keep=2)
    calls=[{'name':'read','args':{},'id':f'id-{i}'} for i in range(3)]
    messages=[SystemMessage(content='rules'),HumanMessage(content='task'),*history(4,content_size=600),
              AIMessage(content='',tool_calls=calls),
              *[ToolMessage(content='short result',tool_call_id=c['id']) for c in calls]]
    async def summary(prompt): return 'Earlier sources were read; conditions unknown.'
    result=asyncio.run(manager.prepare(messages,scope='parallel',model_factory=lambda:FakeListChatModel(responses=[]),summarize=summary))
    assert_pairs(result)
    assert len([m for m in result if isinstance(m,ToolMessage)])==3


@pytest.mark.parametrize('error',[BudgetExceeded('no budget'),RuntimeError('provider unavailable')])
def test_summary_errors_propagate_without_retry_or_cache_commit(tmp_path,error):
    manager=cm.ContextCompressor(run_fixture(tmp_path),trigger=1000,keep=2)
    calls=[]
    async def fail(prompt): calls.append(1);raise error
    with pytest.raises(type(error)):
        asyncio.run(manager.prepare([HumanMessage(content='task'),*history()],scope='fail',
            model_factory=lambda:FakeListChatModel(responses=[]),summarize=fail))
    assert len(calls)==1 and not manager.cache


def test_small_context_never_constructs_model_and_huge_recent_turn_fails(tmp_path):
    manager=cm.ContextCompressor(run_fixture(tmp_path),trigger=1000,keep=4)
    def no_model(): raise AssertionError('No model needed')
    async def no_summary(prompt): raise AssertionError('No summary needed')
    tiny=[HumanMessage(content='task'),AIMessage(content='ok')]
    assert asyncio.run(manager.prepare(tiny,scope='tiny',model_factory=no_model,summarize=no_summary))==tiny
    with pytest.raises(cm.ContextLimitExceeded):
        asyncio.run(manager.prepare([HumanMessage(content='task'),*history(1,content_size=9000)],scope='large',
            model_factory=lambda:FakeListChatModel(responses=[]),summarize=no_summary))


def test_archive_reader_only_registered_id_and_paged_text(tmp_path):
    run=run_fixture(tmp_path);run.context_manager=cm.ContextCompressor(run)
    path=run.directory/'old.json';path.write_text('a'*7000)
    run.context_manager.archives['known']=path
    token=ACTIVE_RUN.set(run)
    try:
        first=json.loads(asyncio.run(cm.read_context_archive.ainvoke({'archive_id':'known'})))
        assert len(first['untrusted_history'])==6000 and first['next_offset']==6000
        with pytest.raises(ValueError):
            asyncio.run(cm.read_context_archive.ainvoke({'archive_id':'../other'}))
    finally: ACTIVE_RUN.reset(token)


def test_mc_action_observation_pairs_and_opening_image_remain(tmp_path):
    manager=cm.ContextCompressor(run_fixture(tmp_path),trigger=2000,keep=3)
    opening=HumanMessage(content=[{'type':'text','text':'Question'}])
    messages=[SystemMessage(content='MC rules'),opening]
    for i in range(6): messages.extend([AIMessage(content=f'action-{i}'),HumanMessage(content=f'observation-{i} '+('x'*800))])
    async def summary(prompt):return 'Earlier actions completed; source-1 remains uncertain.'
    result=asyncio.run(manager.prepare(messages,scope='mcsearch',model_factory=lambda:FakeListChatModel(responses=[]),summarize=summary))
    assert result[1]==opening
    # Summary followed by complete action/observation pairs, including the newest.
    assert isinstance(result[3],AIMessage) and result[3].content=='action-4'
    assert result[-1].content.startswith('observation-5')


def test_memory_persistence_isolation_update_expiry_and_delete(tmp_path,monkeypatch):
    clock=[1000.0]
    monkeypatch.setattr('open_deep_research.long_term_memory.time.time',lambda:clock[0])
    path=tmp_path/'memory.db';store=MemoryStore(path)
    store.put('project-a','style','preference',{'text':'中文回答'},ttl_days=0)
    store.put('project-a','e1','episode',{'question':'PDF 页面相关度检索'},ttl_days=1)
    store.put('project-b','e1','episode',{'question':'PDF 页面相关度检索 private'})
    reopened=MemoryStore(path)
    assert [r['key'] for r in reopened.recall('project-a','PDF 相关度')]==['e1']
    assert not reopened.recall('project-a','unrelated astronomy')
    assert 'private' not in recall_context(reopened,'project-a','PDF 相关度')[0]
    reopened.put('project-a','style','preference',{'text':'简短中文'},ttl_days=0)
    clock[0]+=86401
    assert not reopened.recall('project-a','PDF 相关度')
    assert '简短中文' in recall_context(reopened,'project-a','anything')[0]
    assert reopened.delete('project-a','style') and not reopened.list('project-a')
    assert reopened.list('project-b')


def test_episode_write_is_grounded_deduplicated_and_redacted(tmp_path):
    run=run_fixture(tmp_path);store=MemoryStore(tmp_path/'memory.db')
    report={'findings':[{'text':'SECRET_FIXTURE 实验条件','evidence_ids':['page-1']}],
            'unknowns':['coverage unclear'],'evidence':{'page-1':{'url':'https://learn.microsoft.com/test'}}}
    key=save_episode(store,'test',run,report)
    assert save_episode(store,'test',run,report)==key and len(store.list('test'))==1
    text,urls,keys=recall_context(store,'test','PDF 实验条件')
    assert key in keys and '不能作为本次报告的证据' in text
    assert 'SECRET_FIXTURE' not in text and '[REDACTED]' in text
    assert urls==['https://learn.microsoft.com/test']
    bad=deepcopy(report);bad['findings'][0]['evidence_ids']=['invented']
    with pytest.raises(ValueError):save_episode(store,'test',run,bad)
    incomplete=deepcopy(report);incomplete['trace']={'status':'incomplete'}
    assert save_episode(store,'test',run,incomplete) is None
    assert save_episode(store,'test',run,{'findings':[],'unknowns':[],'evidence':{}}) is None


def test_cli_two_runs_recall_and_disable_memory_without_network(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    monkeypatch.setenv('OPENAI_API_BASE','https://dashscope.aliyuncs.com/compatible-mode/v1')
    path=tmp_path/'memory.db';store=MemoryStore(path)
    store.put('project','style','preference',{'text':'PREFERENCE_MARKER 中文回答'},ttl_days=0)
    seen=[]
    class Graph:
        async def ainvoke(self,state,config):
            run=ACTIVE_RUN.get();seen.append((deepcopy(state),run.memory_context))
            # Prior memory never populates this run's evidence catalog.
            assert not run.pages and not getattr(run,'document_results',[])
            value={'findings':[{'text':'READ_AGAIN_MARKER PDF condition','evidence_ids':['fresh']}],
                   'unknowns':[],'evidence':{'fresh':{'url':'https://learn.microsoft.com/fixture','text':'fresh read'}}}
            (run.directory/'report.json').write_text(json.dumps(value),'utf-8')
            return {'final_report':'fresh report'}
    monkeypatch.setattr(deep_researcher,'deep_researcher',Graph())
    for i in range(3):
        options=SimpleNamespace(question='PDF condition',image=None,pdf=None,web=True,web_screenshot=[],
            model='openai:qwen3.5-flash',planner='native',max_requests=16,output=str(tmp_path/f'run-{i}'),
            env_file=str(tmp_path/'missing.env'),memory_db=str(path),memory_namespace='project',no_memory=i==2)
        asyncio.run(cli.research(options))
    assert 'PREFERENCE_MARKER' in str(seen[0]) and 'READ_AGAIN_MARKER' not in str(seen[0])
    assert 'READ_AGAIN_MARKER' in str(seen[1])
    assert seen[2][1]=='' and 'PREFERENCE_MARKER' not in str(seen[2])


def test_compression_uses_existing_budget_and_no_inherited_callbacks(tmp_path,monkeypatch):
    from open_deep_research import evidence_model
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    run=run_fixture(tmp_path)
    run.context_manager=cm.ContextCompressor(run,trigger=3000,keep=3)
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0,'limits':{
            'requests':1,'searches':0,'input_chars':100000,'output_tokens':10000}}))
    calls=[]
    class Model:
        def __init__(self,**kwargs): pass
        async def ainvoke(self,messages,config):
            assert config['callbacks']==[]
            calls.append(messages)
            return AIMessage(content='Known source page-0; coverage unclear.',
                response_metadata={'finish_reason':'stop'},
                usage_metadata={'input_tokens':100,'output_tokens':20,'total_tokens':120})
    monkeypatch.setattr(evidence_model,'ChatOpenAI',Model)
    token=ACTIVE_RUN.set(run)
    config={'configurable':{'research_model':'openai:qwen3.5-flash'}}
    try:
        asyncio.run(cm.prepare_context([HumanMessage(content='task'),*history()],config,scope='budget'))
        with pytest.raises(BudgetExceeded):
            asyncio.run(cm.prepare_context([HumanMessage(content='task'),*history(14)],config,scope='budget'))
    finally: ACTIVE_RUN.reset(token)
    assert len(calls)==1
    events=[json.loads(line) for line in (run.directory/'events.jsonl').read_text('utf-8').splitlines()]
    assert sum(e['event']=='request_reserved' and e['kind']=='context_summary' for e in events)==1
    assert any(e['event']=='evidence_model' and e['stage']=='context_summary' for e in events)


def test_summary_call_cap_prevents_paid_loop(tmp_path):
    manager=cm.ContextCompressor(run_fixture(tmp_path),trigger=3000,keep=3,max_summaries=1)
    calls=[]
    async def summarize(prompt): calls.append(1);return 'Summary with evidence gaps.'
    async def check():
        await manager.prepare([HumanMessage(content='task'),*history()],scope='cap',
            model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
        # Large unresolved remainder cannot trigger a second paid summary.
        with pytest.raises(cm.ContextLimitExceeded):
            await manager.prepare([HumanMessage(content='task'),*history(30)],scope='cap',
                model_factory=lambda:FakeListChatModel(responses=[]),summarize=summarize)
    asyncio.run(check())
    assert len(calls)==1
