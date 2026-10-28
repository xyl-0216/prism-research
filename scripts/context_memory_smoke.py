"""One paid Flash summary + local persistence check using synthetic data only.

python scripts/context_memory_smoke.py --env-file PATH --output runs/UNIQUE_NAME
Uses at most one model request. Never writes to the project's real memory DB.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_deep_research.context_management import ContextCompressor, prepare_context, estimate_tokens
from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
from open_deep_research.long_term_memory import MemoryStore, recall_context


async def main(args):
    names=('OPENAI_API_KEY','OPENAI_API_BASE','OPENAI_BASE_URL','RESEARCH_MODEL','LANGSMITH_TRACING')
    previous={k:os.environ.get(k) for k in names}
    settings=dotenv_values(args.env_file)
    root=Path(args.output).resolve()
    token=None
    summary={'synthetic_fixture_only':True,'model':'openai:qwen3.5-flash','success':False}
    try:
        for name in names[:3]:
            if settings.get(name): os.environ[name]=settings[name]
        os.environ['RESEARCH_MODEL']='openai:qwen3.5-flash'
        os.environ['LANGSMITH_TRACING']='false'
        run=EvidenceRun(root,root/'budget.json',secrets=[os.environ.get('OPENAI_API_KEY','')])
        run.ledger.write_text(json.dumps({'approved':True,'requests':0,'searches':0,
            'input_chars':0,'reserved_output_tokens':0,
            'limits':{'requests':1,'searches':0,'input_chars':60000,'output_tokens':1800}}),'utf-8')
        run.context_manager=ContextCompressor(run,trigger=3000,keep=4,max_summaries=1)
        messages=[SystemMessage(content='Synthetic wiring check; do not invent facts.'),
                  HumanMessage(content='Verify experiment conditions and unresolved questions.')]
        for i in range(7):
            fact=('SYNTHETIC page-0: resolution 1440p; frame generation enabled; disabled performance unknown.'
                  if i==0 else f'SYNTHETIC page-{i}: no further performance measurements.')
            messages.extend([AIMessage(content='',tool_calls=[{'name':'read_source',
                'args':{'page':i},'id':f't-{i}'}]),
                ToolMessage(content=fact+'\n'+('Repeated layout metadata, not evidence. '*40),tool_call_id=f't-{i}')])
        token=ACTIVE_RUN.set(run)
        config={'configurable':{'research_model':'openai:qwen3.5-flash'}}
        result=await prepare_context(messages,config,scope='live-fixture')
        repeated=await prepare_context(messages,config,scope='live-fixture')
        events=[json.loads(s) for s in (root/'events.jsonl').read_text('utf-8').splitlines()]
        store=MemoryStore(root/'fixture-memory.sqlite3')
        store.put('fixture-project','style','preference',{'text':'中文回答'},ttl_days=0)
        reopened=MemoryStore(root/'fixture-memory.sqlite3')
        remembered=recall_context(reopened,'fixture-project','experiment')[0]
        checks={'summary_contains_1440p':'1440p' in str(result[2].content),
                'pinned_task':result[:2]==messages[:2],
                'smaller':estimate_tokens(result)<estimate_tokens(messages),
                'one_paid_request':json.loads(run.ledger.read_text())['requests']==1,
                'cache_reused':result==repeated,
                'memory_persisted':'中文回答' in remembered,
                'namespace_isolated':not recall_context(reopened,'another-project','experiment')[0]}
        summary.update(success=all(checks.values()),checks=checks,
            before_estimated_tokens=estimate_tokens(messages),after_estimated_tokens=estimate_tokens(result),
            usage=[e['usage'] for e in events if e['event']=='evidence_model'])
    except Exception as error:
        # Error type only: no provider request body or credentials in stdout.
        summary['error_type']=type(error).__name__
    finally:
        if token is not None: ACTIVE_RUN.reset(token)
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value
    if root.exists() and 'run' in locals():
        (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary['success']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(args)) else 1)
