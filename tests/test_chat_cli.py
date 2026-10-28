"""Interactive history, resume, evidence handoff and durable spending boundaries."""
import argparse
import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk

from open_deep_research import cli,chat_cli as chat
from open_deep_research.evidence_runtime import BudgetExceeded,EvidenceRun


def args(tmp_path,**values):
    parser=argparse.ArgumentParser()
    cli.add_research_arguments(parser)
    result=parser.parse_args([])
    result.session_max_requests=10;result.resume=None;result.list=False
    result.session_dir=str(tmp_path/'sessions');result.no_memory=True
    for key,value in values.items():setattr(result,key,value)
    return result


def create(tmp_path,**values):
    options=args(tmp_path,**values)
    return chat.Session.create(options.session_dir,options)


def quiet(*a,**k):pass


async def fake_reply(session,question,run,emit):
    run.reserve('chat_answer',len(question),20)
    answer='已记住条件：'+question
    emit(answer)
    return answer


def test_history_resume_and_budget_are_durable(tmp_path):
    session=create(tmp_path,session_max_requests=2)
    asyncio.run(chat.Chat(session,emit=quiet).turn('预算1000元',reply=fake_reply))
    resumed=chat.Session.load(session.directory.parent,session.data['id'])
    assert resumed.data['turns'][0]['assistant']=='已记住条件：预算1000元'
    asyncio.run(chat.Chat(resumed,emit=quiet).turn('再补充通勤用途',reply=fake_reply))
    assert resumed.used_requests()==2
    with pytest.raises(BudgetExceeded):
        asyncio.run(chat.Chat(resumed,emit=quiet).turn('不应调用',reply=fake_reply))
    assert len(resumed.data['turns'])==2


@pytest.mark.parametrize('error',[RuntimeError('provider secret'),asyncio.CancelledError()])
def test_failure_and_cancellation_still_charge_reserved_requests(tmp_path,error):
    session=create(tmp_path)
    async def fail(session,q,run,emit):
        run.reserve('chat_answer',1,20)
        raise error
    with pytest.raises(type(error)):
        asyncio.run(chat.Chat(session,emit=quiet).turn('任务',reply=fail))
    saved=chat.Session.load(session.directory.parent,session.data['id'])
    assert saved.used_requests()==1
    assert saved.data['turns'][0]['status'] in ('failed','interrupted')
    assert 'provider secret' not in (saved.directory/'session.json').read_text('utf-8')


def test_file_snapshots_and_session_paths(tmp_path):
    import fitz
    source=tmp_path/'my file.pdf'
    with fitz.open() as doc:
        doc.new_page();doc.save(source)
    session=create(tmp_path)
    session.attach('"'+str(source)+'"')
    source.unlink()
    target=Path(session.assets()['pdf'])
    assert target.is_file()
    target.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='changed'):session.assets()
    with pytest.raises(ValueError):session.local('../outside')
    with pytest.raises(ValueError):chat.Session.load(tmp_path,'../outside')


def test_session_lock_and_crash_recovery(tmp_path):
    session=create(tmp_path)
    with chat.session_lock(session.directory):
        with pytest.raises(ValueError,match='already open'):
            with chat.session_lock(session.directory):pass
    with chat.session_lock(session.directory):pass
    session.data['turns'].append({'status':'running','user':'interrupted','directory':'turns/one'})
    session.recover()
    assert session.data['turns'][0]['status']=='interrupted'


def test_research_receives_prior_conversation_and_report_is_reused(tmp_path):
    session=create(tmp_path,web=True)
    asyncio.run(chat.Chat(session,emit=quiet).turn('只考虑开源框架',reply=fake_reply))
    async def researcher(options,emit):
        assert '只考虑开源框架' in options.conversation_context
        assert options.max_requests==9
        root=Path(options.output);root.mkdir(parents=True)
        report={'findings':[{'text':'支持checkpoint','evidence_ids':['page-1']}],
                'unknowns':[],'evidence':{'page-1':{'id':'page-1','url':'https://docs.langchain.com/test','text':'原文checkpoint'}}}
        (root/'report.json').write_text(json.dumps(report),'utf-8')
        (root/'report.md').write_text('支持checkpoint [page-1]','utf-8')
        (root/'report.html').write_text('fixture','utf-8')
        return root/'report.html'
    asyncio.run(chat.Chat(session,emit=quiet).turn('研究它们的持久化',research=True,researcher=researcher))
    context,urls=session.report_context()
    assert '原文checkpoint' in context and urls==['https://docs.langchain.com/test']
    assert session.latest_report().is_file()


def test_actual_streaming_answer_receives_history_and_archived_evidence(tmp_path,monkeypatch):
    import langchain_openai
    session=create(tmp_path,no_context_compression=True)
    asyncio.run(chat.Chat(session,emit=quiet).turn('预算是1200元',reply=fake_reply))
    seen=[]
    class Model:
        def __init__(self,**kw):assert kw['max_retries']==0
        async def astream(self,messages,**kw):
            seen.extend(messages)
            yield AIMessageChunk(content='1200')
            yield AIMessageChunk(content='元',response_metadata={'finish_reason':'stop'},
                                 usage_metadata={'input_tokens':30,'output_tokens':5,'total_tokens':35})
    monkeypatch.setattr(langchain_openai,'ChatOpenAI',Model)
    monkeypatch.setenv('OPENAI_API_KEY','fixture-secret')
    asyncio.run(chat.Chat(session,emit=quiet).turn('我的预算是多少？'))
    assert [m.type for m in seen]==['system','human','ai','human']
    assert seen[1].content=='预算是1200元'
    assert session.data['turns'][-1]['assistant']=='1200元'
    assert session.used_requests()==2


def test_shell_commands_list_and_exit_do_not_call_model(tmp_path,monkeypatch):
    async def forbidden(*a,**kw):pytest.fail('No model expected for shell commands')
    monkeypatch.setattr(chat,'answer',forbidden)
    commands=iter(['/help','/web on','/status','/report','/steps','/history','/exit'])
    output=[]
    chat.run_chat(args(tmp_path),read=lambda _:next(commands),emit=lambda s,**kw:output.append(s))
    path=next((tmp_path/'sessions').glob('*/session.json'))
    data=json.loads(path.read_text('utf-8'))
    assert not data['turns'] and data['settings']['web']
    chat.run_chat(args(tmp_path,list=True),emit=lambda s,**kw:output.append(s))
    assert any(data['id'] in s for s in output)


def test_secrets_are_not_saved_and_failed_turns_are_not_replayed(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','fixture-secret-123')
    session=create(tmp_path)
    asyncio.run(chat.Chat(session,emit=quiet).turn('fixture-secret-123',reply=fake_reply))
    assert 'fixture-secret-123' not in (session.directory/'session.json').read_text('utf-8')
    assert session.data['turns'][0]['user']=='[REDACTED]'


def test_plain_chat_has_no_research_input_requirement_and_research_does(tmp_path):
    session=create(tmp_path)
    asyncio.run(chat.Chat(session,emit=quiet).turn('你好',reply=fake_reply))
    with pytest.raises(ValueError,match='attach'):
        asyncio.run(chat.Chat(session,emit=quiet).turn('查资料',research=True))
    assert len(session.data['turns'])==1


def test_compaction_survives_resume_without_repeating_summary(tmp_path,monkeypatch):
    import langchain_openai
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import AIMessage
    session=create(tmp_path,context_trigger=2500)
    for i in range(12):
        session.data['turns'].append({'user':f'条件{i}'+('任务'*150),
            'assistant':'回复'+('研究'*150),'status':'completed','directory':f'turns/history-{i}'})
    session.save()
    summaries=[]
    class Model(FakeListChatModel):
        async def ainvoke(self,messages,**kw):
            summaries.append(1)
            return AIMessage(content='早期对话条件保留，原始资料未变化。',response_metadata={'finish_reason':'stop'})
        async def astream(self,messages,**kw):
            yield AIMessageChunk(content='继续回答',response_metadata={'finish_reason':'stop'})
    monkeypatch.setattr(langchain_openai,'ChatOpenAI',lambda **kw:Model(responses=[]))
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    asyncio.run(chat.Chat(session,emit=quiet).turn('总结一下'))
    assert len(summaries)==1 and session.data['compression']['cache']
    restored=chat.Session.load(session.directory.parent,session.data['id'])
    asyncio.run(chat.Chat(restored,emit=quiet).turn('继续解释'))
    assert len(summaries)==1 and restored.used_requests()==3
    assert len(restored.data['turns'])==14


def test_research_graph_receives_session_background_and_scoped_urls(tmp_path,monkeypatch):
    import fitz
    from open_deep_research import deep_researcher as dr
    from open_deep_research.evidence_runtime import ACTIVE_RUN
    image=tmp_path/'chart.png'
    with fitz.open() as doc:
        doc.new_page(width=100,height=100).get_pixmap().save(image)
    options=args(tmp_path,image=str(image),question='继续解释这张图',output=str(tmp_path/'research'))
    options.conversation_context='用户前一轮指定只解释实验条件'
    options.conversation_urls=['https://docs.langchain.com/test','https://unlisted.invalid/']
    class Graph:
        async def ainvoke(self,state,config):
            assert options.conversation_context in state['messages'][0].content
            run=ACTIVE_RUN.get()
            assert run.source_urls=={'https://docs.langchain.com/test'}
            assert run.question==options.question
            (run.directory/'report.json').write_text(json.dumps({'findings':[],
                'unknowns':['fixture'],'evidence':{}}),'utf-8')
            return {'final_report':'fixture report'}
    monkeypatch.setattr(dr,'deep_researcher',Graph())
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    path=asyncio.run(cli.research(options))
    assert path.is_file()
