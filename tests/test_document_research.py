import asyncio
import json
from pathlib import Path

import fitz
import numpy as np
import pytest

from open_deep_research.demo import create_demo_pdf
from open_deep_research.document_retrieval import index_pdf, rank_pages, render_page, validate_document_answer
from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
from open_deep_research.mpmqa_scoring import late_interaction_score
from open_deep_research.reporting import validate_report, write_report_html


def fixture_run(tmp_path):
    run=EvidenceRun(tmp_path/'run',tmp_path/'budget.json')
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,'input_chars':0,'reserved_output_tokens':0}),'utf-8')
    path=tmp_path/'study.pdf';create_demo_pdf(path)
    return run,index_pdf(run,path)


class FixtureRetriever:
    """Explicit fixed scores for pipeline tests, never used by production."""
    def rank(self, document, question, top_k):
        return [(0.8, document['pages'][1]), (0.1, document['pages'][0])][:top_k]


def test_pdf_ranking_and_region_geometry(tmp_path):
    run,document=fixture_run(tmp_path)
    score,page=rank_pages(document,'resolution frame generation coverage', retriever=FixtureRetriever())[0]
    assert page['page']==2 and score>0
    assert any(r['kind']=='graphic' for r in page['regions'])
    for region in page['regions']:
        a,b,c,d=region['bbox'];assert 0<=a<c<=1 and 0<=b<d<=1
    image=render_page(document,page,run.directory/'images')
    assert Path(image['raw_path']).is_file()
    Path(document['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):render_page(document,page,run.directory/'images')


def test_mpmqa_interaction_scoring():
    q=np.eye(2,dtype=np.float32)
    p=np.array([[1,0],[0,1],[1,0]],dtype=np.float32)
    assert late_interaction_score(q,p)==(1.0,1.0)
    assert late_interaction_score(q,p[:1])==(0.5,1.0)
    with pytest.raises(ValueError):late_interaction_score(q,np.zeros((0,2)))


@pytest.mark.parametrize('ids',[[],['made-up'],['x','x']])
def test_citations_cannot_be_invented(tmp_path,ids):
    _,doc=fixture_run(tmp_path)
    with pytest.raises(ValueError):validate_document_answer({'answer':'text','region_ids':ids},doc['pages'])


def test_report_rejects_uncited_claims_and_escapes_html(tmp_path):
    with pytest.raises(ValueError):validate_report({'findings':[{'text':'claim','evidence_ids':['fake']}],'unknowns':[]},{})
    value={'findings':[{'text':'<script>attack()</script>','evidence_ids':['source']}],
           'unknowns':[],'evidence':{'source':{'text':'<img onerror=evil>','url':'javascript:evil'}}}
    path=tmp_path/'report.html';write_report_html(value,path)
    text=path.read_text('utf-8')
    assert '<script>' not in text and 'href="javascript:' not in text
    assert '&lt;script&gt;' in text


def test_general_image_registration(tmp_path):
    from open_deep_research.assets import register_image
    run,doc=fixture_run(tmp_path)
    rendered=render_page(doc,doc['pages'][0],run.directory/'images')
    asset=register_image(run,rendered['raw_path'])
    assert asset['id']=='input_image' and Path(asset['path']).is_file()
    with pytest.raises(ValueError):register_image(run,doc['path'])


@pytest.mark.parametrize('experimental', [False, True])
def test_full_general_graph_document_path(tmp_path,monkeypatch,experimental):
    from langchain_core.messages import AIMessage, HumanMessage
    from open_deep_research import deep_researcher as dr, mcsearch_adapter as mc, document_retrieval as documents, reporting
    from open_deep_research.state import ResearchQuestion
    from open_deep_research import context_management
    scopes=[]
    async def context_spy(messages,config,*,scope):
        scopes.append(scope)
        return messages
    monkeypatch.setattr(dr,'prepare_context',context_spy)
    monkeypatch.setattr(context_management,'prepare_context',context_spy)
    run,doc=fixture_run(tmp_path);run.allow_web=False
    region=next(r for r in doc['pages'][1]['regions'] if 'Frame generation' in r['text'])
    native_responses=iter([
        AIMessage(content='',tool_calls=[{'name':'ConductResearch','args':{'research_topic':'frame generation'},'id':'delegate'}]),
        AIMessage(content='',tool_calls=[{'name':'search_document_evidence','args':{'document_id':doc['id'],'question':'conditions'},'id':'document'}]),
        AIMessage(content='',tool_calls=[{'name':'ResearchComplete','args':{},'id':'done'}]),
        AIMessage(content='Frame generation enabled'),
        AIMessage(content='',tool_calls=[{'name':'ResearchComplete','args':{},'id':'finish'}]),
    ])
    class GraphModel:
        def __init__(self,schema=None):self.schema=schema
        def with_structured_output(self,schema,**kw):return GraphModel(schema)
        def bind_tools(self,*a,**kw):return self
        def with_retry(self,**kw):return self
        def with_config(self,*a,**kw):return self
        async def ainvoke(self,*a,**kw):
            if self.schema is ResearchQuestion:return ResearchQuestion(research_brief='frame generation conditions')
            if not experimental:
                assert 'LOCAL BOUNDED TEST' not in str(a)
                return next(native_responses)
            return AIMessage(content='',tool_calls=[{'name':'ConductResearch','args':{'research_topic':'frame generation'},'id':'research-1'}])
    responses=iter([
        '<Sub-Question>\nWhat are the conditions?\n<Search>\nLocal Document Retrieval: frame generation conditions\n<Dependencies>\n[]',
        json.dumps({'status':'answered','intermediate_answer':'Frame generation enabled','evidence_ids':[region['id']],'missing_evidence':''}),
        '<End>\nFinal Answer: Conditions identified',
    ])
    class Planner:
        def __init__(self,**kw):pass
        async def ainvoke(self,*a,**kw):return AIMessage(content=next(responses),response_metadata={'finish_reason':'stop'})
    def factory(run,config,**kwargs):
        async def infer(stage,prompt,images=()):
            if stage=='document_answer':
                assert sum(i['type']=='image_url' for i in images)==2
                return json.dumps({'answer':'Frame generation enabled','region_ids':[region['id']]})
            return json.dumps({'findings':[{'text':'Frame generation enabled','evidence_ids':[region['id']]}],'unknowns':[]})
        return infer
    monkeypatch.setattr(dr,'configurable_model',GraphModel())
    monkeypatch.setattr(mc,'ChatOpenAI',Planner)
    monkeypatch.setattr(documents,'make_infer',factory)
    monkeypatch.setattr(documents,'get_retriever',FixtureRetriever)
    monkeypatch.setattr(reporting,'make_infer',factory)
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    config={'configurable':{'evidence_profile':'general','mcsearch_enabled':experimental,'mmsearch_enabled':False,
        'concise_report':True,'allow_clarification':False,'research_model':'openai:fixture',
        'final_report_model':'openai:fixture','search_api':'none'}}
    token=ACTIVE_RUN.set(run)
    try: result=asyncio.run(dr.deep_researcher.ainvoke({'messages':[HumanMessage(content='What are the conditions?')]},config))
    finally: ACTIVE_RUN.reset(token)
    assert region['id'] in result['final_report']
    if experimental:
        assert run.mcsearch_trace['status']=='completed'
        assert 'mcsearch' in scopes
    else:
        assert 'supervisor' in scopes and any(s.startswith('researcher:') for s in scopes)
        assert not getattr(run, 'mcsearch_trace', None)
        saved=json.loads((run.directory/'report.json').read_text('utf-8'))
        assert saved['research']['method']=='open_deep_research'
        assert saved['research']['question']=='frame generation conditions'
    assert (run.directory/'report.json').exists()
