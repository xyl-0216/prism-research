"""Shopping source/price boundaries and full workflow integration, without APIs."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research import shopping as shop, evidence_tools, bailian_search, reporting, deep_researcher as dr, cli
from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
from open_deep_research.state import ResearchQuestion


IDENTITY={'product':'TEST X1','variant':'128GB','visible_identifiers':'TEST X1 128GB',
          'certainty':'explicit','uncertainties':[],'search_query':'TEST X1 128GB'}
TEXT='TEST X1 128GB 全新 标价 CNY 999.00，运费另计。'
PAGE={'id':'page-1','url':'https://item.jd.com/123.html','title':'TEST X1 128GB',
      'text':TEXT,'fetched_at':'2026-10-28T12:00:00+00:00','sha256':'fixture-hash'}
OFFER={'page_id':'page-1','product':'TEST X1','variant':'128GB','match':'candidate',
       'price_text':'CNY 999.00','price_kind':'listed','currency':'CNY','conditions':'运费另计','quote':TEXT}


def fixture_run(tmp_path):
    run=EvidenceRun(tmp_path/'run',tmp_path/'run'/'budget.json')
    run.shopping_enabled=True;run.shopping_sites={'item.jd.com','www.apple.com.cn'}
    run.allowed_hosts=evidence_tools.ALLOWED_HOSTS | run.shopping_sites
    run.shopping_preferences={'market':'中国大陆','currency':'CNY','budget':1200,'usage':'通勤'}
    run.max_searches=2;run.source_urls=set();run.allow_web=True;run.question='TEST X1 research'
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,'input_chars':0,
        'reserved_output_tokens':0,'limits':{'requests':16,'searches':2,'input_chars':300000,'output_tokens':50000}}))
    return run


def test_marketplace_allowlist_is_scoped_and_redirect_safe(tmp_path,monkeypatch):
    with pytest.raises(ValueError): evidence_tools.validate_url(PAGE['url'])
    run=fixture_run(tmp_path);token=ACTIVE_RUN.set(run)
    try:
        assert evidence_tools.validate_url(PAGE['url'])==PAGE['url']
        for url in ['https://item.jd.com.evil.test/1','http://item.jd.com/1','https://127.0.0.1/',
                    'https://item.jd.com:444/1','https://user:pass@item.jd.com/1']:
            with pytest.raises(ValueError): evidence_tools.validate_url(url)
        client=httpx.AsyncClient
        def redirect(request):return httpx.Response(302,headers={'location':'https://evil.test/'})
        monkeypatch.setattr(evidence_tools.httpx,'AsyncClient',lambda **kw:client(**kw,transport=httpx.MockTransport(redirect)))
        with pytest.raises(ValueError):asyncio.run(evidence_tools.fetch(PAGE['url']))
    finally: ACTIVE_RUN.reset(token)
    with pytest.raises(ValueError): evidence_tools.validate_url(PAGE['url'])


def test_product_jsonld_extracted_without_script_execution():
    raw='<script>evil()</script><script type="application/ld+json">'+json.dumps({
        '@graph':[{'@type':'Product','name':'TEST X1','offers':{'price':'999','priceCurrency':'CNY'}},
                  {'@type':'Person','name':'ignore'}]})+'</script>'
    result=json.loads(shop.product_metadata(raw))
    assert len(result)==1 and result[0]['offers']['price']=='999'
    assert not shop.product_metadata('<script type="application/ld+json">broken</script>')


@pytest.mark.parametrize('field,value', [('page_id','invented'),('price_text','CNY 1'),('quote','Made up CNY 999.00')])
def test_offer_cannot_invent_quote_price_or_evidence(field,value):
    offer={**OFFER,field:value}
    with pytest.raises(ValueError):shop.verify_offers({'offers':[offer]},[PAGE],IDENTITY)


def test_installment_and_uncertain_variant_never_become_confirmed_total():
    quote='TEST X1 分期每月 CNY 99'
    value={**OFFER,'price_text':'CNY 99','quote':quote}
    result=shop.verify_offers({'offers':[value]},[{**PAGE,'text':quote}],{**IDENTITY,'certainty':'uncertain'})[0]
    assert result['price_kind']=='installment' and result['match']=='uncertain'
    assert result['checkout_verified'] is False
    assert shop.price_number('999','CNY') is None
    assert shop.price_number('CNY 999','USD') is None


def test_single_ear_variant_cannot_be_presented_as_full_set():
    quote='TEST X1 整套 左右耳充电盒 当前规格 单左耳 CNY 99'
    offer={**OFFER,'variant':'全新整套','quote':quote,'price_text':'CNY 99'}
    result=shop.verify_offers({'offers':[offer]},[{**PAGE,'text':quote}],IDENTITY)[0]
    assert result['match']=='different' and '单耳/配件' in result['variant']
    assert '不作为整套价格' in result['conditions']


def test_installment_phrase_keeps_label_without_parsing_a_total():
    quote='24 期 0% 费率，每月仅约 RMB 48'
    offer={**OFFER,'quote':quote,'price_text':quote}
    result=shop.verify_offers({'offers':[offer]},[{**PAGE,'text':quote}],IDENTITY)[0]
    assert result['amount'] is None and result['price_kind']=='installment'
    assert result['match']=='uncertain' and '待核验' in result['variant']


def test_report_prices_must_match_cited_offers_and_currency():
    offers=shop.verify_offers({'offers':[OFFER]},[PAGE],IDENTITY)
    offers[0]['evidence_id']='offer:1'
    shopping={'offers':offers,'preferences':{'budget':1200}}
    def report(text,refs):return {'findings':[{'text':text,'evidence_ids':refs}]}
    shop.validate_price_claims(report('标价999元，运费另计。',['offer:1']),shopping)
    shop.validate_price_claims(report('用户预算1200元。',['page-1']),shopping)
    for text,refs in [('当前只要1元',['offer:1']),('标价999元',['wrong']),('USD 999',['offer:1'])]:
        with pytest.raises(ValueError):shop.validate_price_claims(report(text,refs),shopping)


def test_press_release_price_is_reference_only():
    page={**PAGE,'url':'https://www.apple.com.cn/newsroom/2024/09/old-launch/'}
    offers=shop.verify_offers({'offers':[OFFER]},[page],IDENTITY)
    assert offers[0]['price_status']=='reference_only'
    shopping={'offers':offers,'preferences':{}}
    with pytest.raises(ValueError):
        shop.validate_price_claims({'findings':[{'text':'当前售价999元','evidence_ids':['page-1']}]},shopping)
    shop.validate_price_claims({'findings':[{'text':'历史发布价格999元','evidence_ids':['page-1']}]},shopping)


def mock_tool_boundary(monkeypatch,run,*,unavailable=False):
    searches=[];inferences=[]
    def factory(run,config,*,schema=None):
        async def infer(stage,prompt,images=()):
            inferences.append(stage)
            if stage=='product_identity':return json.dumps(IDENTITY)
            return json.dumps({'offers':[OFFER]})
        return infer
    async def search(query,config):
        run.searches+=1;searches.append(query)
        return {'sources':[{'url':PAGE['url'],'title':PAGE['title']},
                           {'url':'https://evil.test/cheap','title':'fake'}]}
    async def read(url):
        if unavailable:raise httpx.HTTPError('blocked')
        run.pages[url]=deepcopy(PAGE)
        return deepcopy(PAGE)
    monkeypatch.setattr(shop,'make_infer',factory)
    monkeypatch.setattr(bailian_search,'bailian_search_async',search)
    monkeypatch.setattr(evidence_tools,'archive_page',read)
    return searches,inferences


@pytest.mark.parametrize('unavailable',[False,True])
def test_purchase_pipeline_budget_cache_and_blocked_sources(tmp_path,monkeypatch,unavailable):
    run=fixture_run(tmp_path)
    searches,inferences=mock_tool_boundary(monkeypatch,run,unavailable=unavailable)
    token=ACTIVE_RUN.set(run)
    try:
        result=json.loads(asyncio.run(shop.research_purchase_options.ainvoke({'question':'TEST X1'},{})))
        again=json.loads(asyncio.run(shop.research_purchase_options.ainvoke({'question':'different question'},{})))
    finally: ACTIVE_RUN.reset(token)
    assert len(searches)==2 and result==again
    assert len(result['purchase_leads'])==1
    assert 'evil.test' not in json.dumps(result)
    if unavailable:
        assert result['status']=='insufficient' and not result['offers']
        assert inferences==['product_identity']
    else:
        assert result['offers'][0]['amount']=='999.00'
        assert result['offers'][0]['evidence_id']=='offer:1'
    assert (run.directory/'shopping_result.json').is_file()


def test_shopping_search_transport_honors_two_call_budget(tmp_path,monkeypatch):
    run=fixture_run(tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    monkeypatch.setenv('OPENAI_API_BASE','https://dashscope.aliyuncs.com/compatible-mode/v1')
    client=httpx.AsyncClient;requests=[]
    def respond(request):
        body=json.loads(request.content);requests.append(body)
        assert set(body['parameters']['search_options']['assigned_site_list'])==run.shopping_sites
        return httpx.Response(200,json={'output':{'search_info':{'search_results':[]},
            'choices':[{'message':{'content':'no results'},'finish_reason':'stop'}]}})
    monkeypatch.setattr(bailian_search.httpx,'AsyncClient',lambda **kw:client(**kw,transport=httpx.MockTransport(respond)))
    config={'configurable':{'bailian_search_model':'openai:qwen-flash'}}
    token=ACTIVE_RUN.set(run)
    try:
        for query in ['q1','q2']:asyncio.run(bailian_search._bailian_native_search(query,config))
        with pytest.raises(RuntimeError,match='budget'):asyncio.run(bailian_search._bailian_native_search('q3',config))
    finally:ACTIVE_RUN.reset(token)
    assert len(requests)==2 and json.loads(run.ledger.read_text())['searches']==2


def test_full_native_shopping_graph_and_report(tmp_path,monkeypatch):
    run=fixture_run(tmp_path);mock_tool_boundary(monkeypatch,run)
    responses=iter([
        AIMessage(content='',tool_calls=[{'name':'ConductResearch','args':{'research_topic':'TEST X1 购买与性价比'},'id':'delegate'}]),
        AIMessage(content='',tool_calls=[{'name':'research_purchase_options','args':{'question':'TEST X1'},'id':'shop'}]),
        AIMessage(content='',tool_calls=[{'name':'ResearchComplete','args':{},'id':'done'}]),
        AIMessage(content='有页面报价，缺少独立性能对照。'),
        AIMessage(content='',tool_calls=[{'name':'ResearchComplete','args':{},'id':'finish'}]),
    ])
    class Model:
        def __init__(self,schema=None):self.schema=schema
        def with_structured_output(self,schema,**kw):return Model(schema)
        def with_config(self,*a,**kw):return self
        def with_retry(self,*a,**kw):return self
        def bind_tools(self,tools,**kw):
            if kw.get('tool_choice')=='any':assert [t.name for t in tools]==['research_purchase_options']
            return self
        async def ainvoke(self,*a,**kw):
            if self.schema is ResearchQuestion:return ResearchQuestion(research_brief='TEST X1 购买与性价比')
            return next(responses)
    def report_factory(run,config,*,schema):
        async def infer(stage,prompt,images=()):
            assert 'purchase_summary' in schema.model_fields
            return json.dumps({'findings':[{'text':'页面标价999元，运费另计','evidence_ids':['offer:1']}],
                'unknowns':['缺少性能对照'],
                'purchase_summary':{'text':'页面标价999元，结算条件待核验','evidence_ids':['offer:1']},
                'value_summary':{'text':'缺少性能对照，不能判断是否最划算','evidence_ids':['page-1']}})
        return infer
    monkeypatch.setattr(dr,'configurable_model',Model())
    monkeypatch.setattr(reporting,'make_infer',report_factory)
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    config={'configurable':{'evidence_profile':'general','concise_report':True,'allow_clarification':False,
        'mmsearch_enabled':True,'search_api':'bailian','research_model':'openai:fixture','final_report_model':'openai:fixture'}}
    token=ACTIVE_RUN.set(run)
    try:result=asyncio.run(dr.deep_researcher.ainvoke({'messages':[HumanMessage(content='TEST X1')]},config))
    finally:ACTIVE_RUN.reset(token)
    assert '购买来源总结' in result['final_report']
    value=json.loads((run.directory/'report.json').read_text('utf-8'))
    reporting.write_report_html(value,run.directory/'report.html')
    html=(run.directory/'report.html').read_text('utf-8')
    assert '性价比分析' in html and 'CNY 999.00' in html and '页面展示价格' in html


def test_shopping_html_escapes_external_titles():
    offer=shop.verify_offers({'offers':[OFFER]},[PAGE],IDENTITY)[0]
    offer['product']='<script>alert(1)</script>'
    result=shop.shopping_html({'identity':IDENTITY,'offers':[offer],'purchase_leads':[
        {'title':'<img src=x onerror=evil()>','url':'javascript:evil()','status':'unavailable'}]})
    assert '<script>' not in result and 'href="javascript:' not in result


def test_cli_shopping_flags(monkeypatch,tmp_path):
    async def run(args):
        assert args.shopping and args.purchase_budget==1200 and args.usage=='通勤'
        assert args.shopping_site==['item.jd.com']
        return tmp_path/'report.html'
    monkeypatch.setattr(cli,'research',run)
    monkeypatch.setattr('sys.argv',['prism-research','research','--shopping','--question','TEST X1',
        '--purchase-budget','1200','--usage','通勤','--shopping-site','item.jd.com'])
    cli.main()


@pytest.mark.parametrize('repair_succeeds',[True,False])
def test_report_repairs_once_or_preserves_only_evidence(tmp_path,monkeypatch,repair_succeeds):
    run=fixture_run(tmp_path);run.pages[PAGE['url']]=deepcopy(PAGE)
    run.shopping_result={'offers':shop.verify_offers({'offers':[OFFER]},[PAGE],IDENTITY)}
    stages=[]
    def factory(run,config,*,schema):
        async def infer(stage,prompt,images=()):
            stages.append(stage)
            valid=repair_succeeds and len(stages)==2
            section={'text':'页面有报价，条件待核实' if valid else '当前价格只有1元',
                     'evidence_ids':['offer:1']}
            return json.dumps({'findings':[section],'unknowns':[],
                'purchase_summary':section,'value_summary':section})
        return infer
    monkeypatch.setattr(reporting,'make_infer',factory)
    asyncio.run(reporting.generate_report(run,{}))
    value=json.loads((run.directory/'report.json').read_text('utf-8'))
    assert stages==['final_report','final_report_repair']
    assert '只有1元' not in json.dumps(value,ensure_ascii=False)
    assert (value.get('generation_status')=='evidence_only') is (not repair_succeeds)
    assert value['shopping']['offers'][0]['amount']=='999.00'


def test_invalid_offer_does_not_discard_valid_sibling(tmp_path,monkeypatch):
    run=fixture_run(tmp_path);mock_tool_boundary(monkeypatch,run)
    def factory(run,config,*,schema=None):
        async def infer(stage,prompt,images=()):
            return json.dumps(IDENTITY if stage=='product_identity' else
                {'offers':[OFFER,{**OFFER,'quote':'Fabricated source text'}]})
        return infer
    monkeypatch.setattr(shop,'make_infer',factory)
    token=ACTIVE_RUN.set(run)
    try:result=json.loads(asyncio.run(shop.research_purchase_options.ainvoke({'question':'TEST X1'},{})))
    finally:ACTIVE_RUN.reset(token)
    assert len(result['offers'])==1 and result['offers'][0]['amount']=='999.00'
    assert any('1 条候选报价' in s for s in result['limitations'])


def test_mc_shopping_routes_text_action_to_purchase_pipeline(tmp_path,monkeypatch):
    from open_deep_research import mcsearch_adapter as mc
    run=fixture_run(tmp_path);searches,_=mock_tool_boundary(monkeypatch,run)
    replies=iter([
        {'action':'research','subquestion':'TEST X1 purchase','modality':'text','depends_on':[]},
        {'status':'answered','intermediate_answer':'页面报价999元','evidence_ids':['page-1'],'missing_evidence':''},
        {'action':'stop','reason':'已读取来源'},
    ])
    class Model:
        def __init__(self,**kw):pass
        async def ainvoke(self,messages,config):
            return AIMessage(content=json.dumps(next(replies)),response_metadata={'finish_reason':'stop'})
    monkeypatch.setattr(mc,'ChatOpenAI',Model)
    monkeypatch.setenv('OPENAI_API_KEY','fixture')
    token=ACTIVE_RUN.set(run)
    try:
        trace=asyncio.run(mc.run_mc_research('TEST X1',{'configurable':{
            'evidence_profile':'general','mmsearch_enabled':True,'search_api':'bailian',
            'research_model':'openai:fixture'}}))
    finally:ACTIVE_RUN.reset(token)
    assert trace['status']=='completed' and len(searches)==2
    assert run.shopping_result['offers'][0]['amount']=='999.00'
    assert trace['steps'][0]['evidence_ids']==['page-1']
