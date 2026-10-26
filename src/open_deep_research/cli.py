"""Local command line entry point; demo is offline, research is an explicit API run."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path


async def research(args):
    from dotenv import load_dotenv
    load_dotenv()
    os.environ['LANGSMITH_TRACING']='false'
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError('Configure OPENAI_API_KEY in your local .env')
    if not args.image and not args.pdf and not args.web:
        raise ValueError('Provide --image, --pdf or --web')
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
    run=EvidenceRun(args.output,Path(args.output)/'budget.json',secrets=[os.environ['OPENAI_API_KEY']])
    run.allow_web=args.web
    run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
        'input_chars':0,'reserved_output_tokens':0,
        'limits':{'requests':args.max_requests,'searches':1 if args.web else 0,
                  'input_chars':300000,'output_tokens':50000}}),'utf-8')
    if args.image: register_image(run,args.image)
    if args.pdf: index_pdf(run,args.pdf)
    run.candidate_screenshots={}
    for mapping in args.web_screenshot:
        url,separator,path=mapping.rpartition('=')
        if not separator: raise ValueError('--web-screenshot expects HTTPS_URL=LOCAL_PATH')
        from open_deep_research.evidence_tools import validate_url
        validate_url(url)
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
            run.log('graph_model_end')
    values={'evidence_profile':'general','concise_report':True,'mcsearch_enabled':True,
        'mmsearch_enabled':args.web,'mcsearch_max_steps':4,'allow_clarification':False,
        'search_api':'bailian' if args.web else 'none','max_concurrent_research_units':1,
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
        result=await asyncio.wait_for(deep_researcher.ainvoke({'messages':[HumanMessage(content=task)]},
            {'configurable':values,'callbacks':[GraphBudget()],'recursion_limit':45}),timeout=900)
        report=str(result.get('final_report',''))
        if report.startswith('Error') or not (run.directory/'report.json').exists():
            raise RuntimeError('Research did not produce a validated report; inspect local events')
        (run.directory/'report.md').write_text(report,'utf-8')
        value=json.loads((run.directory/'report.json').read_text('utf-8'))
        write_report_html(value,run.directory/'report.html')
        return run.directory/'report.html'
    except Exception as error:
        run.log('research_failed',type=type(error).__name__)
        raise
    finally:
        ACTIVE_RUN.reset(token)
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


def main():
    parser=argparse.ArgumentParser(description='Prism Research — 图像、网页与文档证据研究')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='生成无需密钥的离线证据演示')
    demo.add_argument('--output',default=None)
    live=sub.add_parser('research',help='使用自己的模型 API 执行研究')
    live.add_argument('--question',required=True)
    live.add_argument('--image');live.add_argument('--pdf')
    live.add_argument('--web',action='store_true')
    live.add_argument('--web-screenshot',action='append',default=[])
    live.add_argument('--model',default='openai:qwen3.5-122b-a10b')
    live.add_argument('--max-requests',type=int,choices=range(1,41),default=24)
    live.add_argument('--output',default=None)
    args=parser.parse_args()
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
        parser.exit(1,f'{type(error).__name__}: execution failed; check configuration and local run logs.\n')


if __name__=='__main__':
    main()
