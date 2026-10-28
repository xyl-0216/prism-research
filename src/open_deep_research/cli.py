"""Local command line entry point; demo is offline, research is an explicit API run."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path


class SetupError(ValueError):
    """Local preflight message safe to show without provider response details."""


async def research(args):
    from dotenv import load_dotenv
    load_dotenv(getattr(args,'env_file',None))
    os.environ['LANGSMITH_TRACING']='false'
    if not os.getenv('OPENAI_API_KEY'):
        raise SetupError('Configure OPENAI_API_KEY in .env, or specify --env-file PATH')
    if getattr(args,'shopping',False):
        args.web=True
    if not args.image and not args.pdf and not args.web:
        raise SetupError('Provide --image, --pdf or --web')
    if args.pdf:
        from open_deep_research.pdf_backend import preflight_pdf_backend
        try:
            preflight_pdf_backend(getattr(args, 'pdf_backend', None), args.pdf)
        except (RuntimeError,ValueError,OSError) as error:
            raise SetupError(str(error)) from error
    if args.web:
        from open_deep_research.bailian_search import native_endpoint
        native_endpoint(os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL') or '')
        if not args.model.startswith('openai:qwen3.5-'):
            raise ValueError('Web mode requires a Qwen3.5 model and Bailian endpoint')
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import HumanMessage
    from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
    from open_deep_research.assets import register_image
    from open_deep_research.document_retrieval import index_pdf
    from open_deep_research.reporting import write_report_html
    from open_deep_research.deep_researcher import deep_researcher
    run=EvidenceRun(args.output,Path(args.output)/'budget.json',
                    secrets=[os.environ['OPENAI_API_KEY'], os.getenv('PRISM_RERANK_API_KEY')])
    from open_deep_research.pdf_backend import backend_name
    run.pdf_backend=backend_name(getattr(args, 'pdf_backend', None))
    run.allow_web=args.web
    run.question=args.question
    from open_deep_research.evidence_tools import validate_url, ALLOWED_HOSTS
    from open_deep_research.shopping import SHOPPING_HOSTS
    run.shopping_enabled=getattr(args,'shopping',False)
    run.shopping_sites=set(getattr(args,'shopping_site',None) or SHOPPING_HOSTS)
    if not run.shopping_sites.issubset(SHOPPING_HOSTS): raise SetupError('Unsupported shopping site')
    run.allowed_hosts=ALLOWED_HOSTS | (run.shopping_sites if run.shopping_enabled else set())
    run.max_searches=(getattr(args,'max_searches',None) or (2 if run.shopping_enabled else 1)) if args.web else 0
    run.shopping_preferences={'market':'中国大陆', 'currency':'CNY',
        'budget':getattr(args,'purchase_budget',None), 'usage':getattr(args,'usage','') or ''}
    run.source_urls=set()
    for url in getattr(args,'conversation_urls',[]):
        try: run.source_urls.add(validate_url(url,allowed_hosts=run.allowed_hosts))
        except ValueError: pass
    for url in re.findall(r'https://[^\s<>"\u3000-\u9fff]+', args.question):
        try:
            run.source_urls.add(validate_url(url.rstrip('.,;:)'),allowed_hosts=run.allowed_hosts))
        except ValueError:
            pass
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0,
        'limits':{'requests':args.max_requests,'searches':run.max_searches,
                  'input_chars':300000,'output_tokens':50000}}),'utf-8')
    if args.image: register_image(run,args.image)
    if args.pdf: index_pdf(run,args.pdf)
    from open_deep_research.context_management import ContextCompressor
    if not getattr(args, 'no_context_compression', False):
        run.context_manager=ContextCompressor(run,
            trigger=getattr(args,'context_trigger',12000), keep=4, max_summaries=2)
    memory_store=None
    memory_namespace=getattr(args,'memory_namespace','default')
    run.memory_context=''
    if not getattr(args,'no_memory',False):
        from open_deep_research.long_term_memory import MemoryStore, DEFAULT_DB, recall_context
        memory_store=MemoryStore(getattr(args,'memory_db',None) or DEFAULT_DB)
        run.memory_context, remembered_urls, memory_keys=recall_context(memory_store,memory_namespace,args.question)
        for url in remembered_urls:
            try: run.source_urls.add(validate_url(url,allowed_hosts=run.allowed_hosts))
            except ValueError: pass
        run.log('memory_recalled', namespace=memory_namespace, keys=memory_keys,
                context_chars=len(run.memory_context))
    run.candidate_screenshots={}
    for mapping in args.web_screenshot:
        url,separator,path=mapping.rpartition('=')
        if not separator: raise ValueError('--web-screenshot expects HTTPS_URL=LOCAL_PATH')
        from open_deep_research.evidence_tools import validate_url
        validate_url(url,allowed_hosts=run.allowed_hosts)
        # Screenshots are auxiliary assets, not the input image used by the planner.
        key='candidate_'+str(len(run.candidate_screenshots))
        record=register_image(run,path,key,source_url=url)
        run.images.pop(key)
        run.candidate_screenshots[url]=record
    class GraphBudget(AsyncCallbackHandler):
        raise_error=True
        async def on_chat_model_start(self,serialized,messages,**kw):
            run.reserve('graph_model',sum(len(str(m.content)) for group in messages for m in group),2400)
        async def on_llm_end(self,response,**kw):
            # Record returned usage for cost estimates without saving prompts.
            usage = []
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, 'message', None)
                    if getattr(message, 'usage_metadata', None):
                        usage.append(message.usage_metadata)
            run.log('graph_model_end', model=args.model, usage=usage)
    planner = getattr(args, 'planner', 'native')
    if planner not in ('native', 'mcsearch'):
        raise SetupError('Planner must be native or mcsearch')
    run.log('research_started', planner=planner, model=args.model, max_requests=args.max_requests)
    values={'evidence_profile':'general','concise_report':True,'mcsearch_enabled':planner == 'mcsearch',
        'mmsearch_enabled':args.web,'mcsearch_max_steps':4,'allow_clarification':False,
        'search_api':'bailian' if args.web else 'none','max_concurrent_research_units':1,
        'bailian_search_model':'openai:qwen-flash',
        'max_researcher_iterations':getattr(args, 'research_rounds', 1),
        'max_react_tool_calls':4,
        'max_structured_output_retries':1,'max_retries':0,'timeout':90,
        'research_model':args.model,'summarization_model':args.model,'compression_model':args.model,
        'final_report_model':args.model,'research_model_max_tokens':2400,
        'summarization_model_max_tokens':1200,'final_report_model_max_tokens':1800}
    # Existing upstream configuration prioritizes environment variables; scope this CLI run.
    previous={key.upper():os.environ.get(key.upper()) for key in values}
    for key,value in values.items():
        os.environ[key.upper()]=str(value).lower() if isinstance(value,bool) else str(value)
    token=ACTIVE_RUN.set(run)
    try:
        task=args.question+'\n可用输入：'+json.dumps({'images':list(run.images),'documents':list(getattr(run,'documents',{})),
                                                      'web':args.web},ensure_ascii=False)
        if run.memory_context:
            task+='\n'+run.memory_context
        if getattr(args,'conversation_context',''):
            task+='\n会话历史（用于理解当前追问，不是新的来源证据；当前用户要求优先）：\n'+args.conversation_context
        if run.shopping_enabled:
            from open_deep_research.shopping import shopping_prompt
            task+='\n'+shopping_prompt(run)
        result=await asyncio.wait_for(deep_researcher.ainvoke({'messages':[HumanMessage(content=task)]},
            {'configurable':values,'callbacks':[GraphBudget()],'recursion_limit':45}),timeout=900)
        report=str(result.get('final_report',''))
        if report.startswith('Error') or not (run.directory/'report.json').exists():
            run.log('report_generation_failed', detail=report[:2000])
            raise RuntimeError('Research did not produce a validated report; inspect local events')
        (run.directory/'report.md').write_text(report,'utf-8')
        value=json.loads((run.directory/'report.json').read_text('utf-8'))
        write_report_html(value,run.directory/'report.html')
        if memory_store is not None:
            from open_deep_research.long_term_memory import save_episode
            try:
                key=save_episode(memory_store,memory_namespace,run,value)
                run.log('memory_saved', namespace=memory_namespace, key=key)
            except Exception as error:
                # A completed report must survive a local DB lock or full disk.
                run.log('memory_save_failed', error_type=type(error).__name__)
        return run.directory/'report.html'
    except Exception as error:
        run.log('research_failed',type=type(error).__name__, message=str(error)[:2000])
        raise
    finally:
        ACTIVE_RUN.reset(token)
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


def add_research_arguments(live, *, interactive=False):
    live.add_argument('--image');live.add_argument('--pdf')
    live.add_argument('--pdf-backend',choices=['ura','api-rerank'],default=None,
                      help='PDF ranking backend; otherwise PRISM_PDF_BACKEND or ura')
    live.add_argument('--env-file',help='Explicit dotenv path; credentials are never printed')
    live.add_argument('--web',action='store_true')
    live.add_argument('--shopping',action='store_true',help='Research product, purchase sources and value; enables web')
    from open_deep_research.shopping import SHOPPING_HOSTS
    live.add_argument('--shopping-site',action='append',choices=sorted(SHOPPING_HOSTS),
                      help='Restrict shopping search to selected supported hosts; repeatable')
    live.add_argument('--purchase-budget',type=float,help='User budget in CNY; not API spending budget')
    live.add_argument('--usage',default='',help='Intended use for product value assessment')
    live.add_argument('--max-searches',type=int,choices=range(1,5),default=None,
                      help='Default 1 normally, 2 for shopping; counts against max-requests')
    if not interactive: live.add_argument('--web-screenshot',action='append',default=[])
    live.add_argument('--model',default='openai:qwen3.5-flash',
                      help='Default: low-cost qwen3.5-flash; override for quality comparisons')
    live.add_argument('--planner',choices=['native','mcsearch'],default='native',
                      help='native: Open Deep Research (default); mcsearch: experimental action loop')
    live.add_argument('--max-requests',type=int,choices=range(1,41),default=16)
    live.add_argument('--research-rounds',type=int,choices=range(1,7),default=1,
                      help='Supervisor research rounds; low-cost debugging defaults to one combined task')
    live.add_argument('--no-memory',action='store_true',help='Disable cross-run memory reads and writes')
    live.add_argument('--memory-db',help='Local SQLite memory database path')
    live.add_argument('--memory-namespace',default='default',help='Separate project/user memory namespace')
    live.add_argument('--no-context-compression',action='store_true')
    live.add_argument('--context-trigger',type=int,default=12000,
                      help='Approximate tokens triggering history compression (minimum 1000)')
    if not interactive: live.add_argument('--output',default=None)


def main():
    parser=argparse.ArgumentParser(description='Prism Research — 图像、网页与文档证据研究')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='生成无需密钥的离线证据演示')
    demo.add_argument('--output',default=None)
    live=sub.add_parser('research',help='使用自己的模型 API 执行研究')
    live.add_argument('--question',required=True)
    add_research_arguments(live)
    chat=sub.add_parser('chat',help='交互式多轮对话与研究')
    add_research_arguments(chat,interactive=True)
    chat.add_argument('--resume',help='Resume a saved session ID')
    chat.add_argument('--list',action='store_true',help='List saved sessions without API calls')
    chat.add_argument('--session-dir',help='Session storage root; default .prism/sessions')
    chat.add_argument('--session-max-requests',type=int,default=100,help='Total request ceiling across resumed turns')
    memory=sub.add_parser('memory',help='Inspect or manage local long-term memory; no API calls')
    memory.add_argument('action',choices=['list','remember','forget'])
    memory.add_argument('--memory-db')
    memory.add_argument('--memory-namespace',default='default')
    memory.add_argument('--key',help='Stable key for a preference, or exact key to delete')
    memory.add_argument('--text',help='Explicit user preference to store; never inferred from sources')
    memory.add_argument('--ttl-days',type=int,default=0,help='0 keeps explicit preference until deletion')
    args=parser.parse_args()
    if args.command=='memory':
        from open_deep_research.long_term_memory import MemoryStore, DEFAULT_DB
        if args.action in ('remember','forget') and not args.key:
            parser.error('--key is required')
        if args.action=='remember' and (not args.text or len(args.text)>2000):
            parser.error('--text must contain 1..2000 characters')
        if not 0<=args.ttl_days<=3650:
            parser.error('--ttl-days must be 0..3650')
        store=MemoryStore(args.memory_db or DEFAULT_DB)
        if args.action=='remember':
            store.put(args.memory_namespace,args.key,'preference',{'text':args.text},ttl_days=args.ttl_days)
            print('Memory saved: '+args.key)
        elif args.action=='forget':
            print('Deleted' if store.delete(args.memory_namespace,args.key) else 'Not found')
        else:
            print(json.dumps(store.list(args.memory_namespace),ensure_ascii=False,indent=2))
        return
    if args.command in ('research','chat') and args.context_trigger<1000:
        parser.error('--context-trigger must be >= 1000')
    if args.command in ('research','chat') and args.purchase_budget is not None:
        import math
        if not math.isfinite(args.purchase_budget) or args.purchase_budget<=0:
            parser.error('--purchase-budget must be a positive finite CNY amount')
    if args.command=='chat':
        from open_deep_research.chat_cli import run_chat
        try:
            run_chat(args)
        except (SetupError,ValueError,OSError) as error:
            parser.exit(1,f'Chat setup failed ({type(error).__name__}); check session ID, paths and configuration.\n')
        return
    if args.output is None:
        args.output=str(Path('runs')/(args.command+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    try:
        if args.command=='demo':
            from open_deep_research.demo import run_demo
            path=asyncio.run(run_demo(args.output))
        else:
            path=asyncio.run(research(args))
        print('Report: '+str(path.resolve()))
    except Exception as error:
        # Never echo raw provider exceptions, which may contain request credentials.
        if isinstance(error,SetupError):
            parser.exit(1,str(error)+'\n')
        parser.exit(1,f'{type(error).__name__}: execution failed; check configuration and local run logs.\n')


if __name__=='__main__':
    main()
