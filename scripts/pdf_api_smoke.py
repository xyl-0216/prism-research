"""Opt-in PDF API smoke check; uses model quota and an EXISTING synthetic PDF.

First run --mode rank --expected-page 2 against `prism-research demo`'s PDF.
Then --mode agent verifies findings, document citations and backend provenance.
Cost is estimated from provider-returned usage at dated China-region list prices,
not an account bill. No retries, downloads, or large-scale evaluations are run.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from open_deep_research.cli import research
from open_deep_research.document_retrieval import index_pdf
from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN
from open_deep_research.pdf_backend import pdf_retrieval, preflight_pdf_backend


RATES = {'openai:qwen3.5-flash': (0.2, 2.0),
         'openai:qwen3.5-122b-a10b': (0.8, 6.4)}
QUESTION = '请核对这份合成资料：实验分辨率是多少，是否开启帧生成，是否能证明关闭帧生成后所有游戏性能翻倍？引用文档区域，明确证据缺口。'


def usage_cost(events, model):
    input_tokens = output_tokens = rerank_tokens = 0
    missing = []
    for event in events:
        if event['event'] == 'pdf_rerank_usage':
            usage = event.get('usage') or {}
            tokens = usage.get('total_tokens', usage.get('input_tokens', usage.get('prompt_tokens')))
            if tokens is None: missing.append('pdf_rerank')
            else: rerank_tokens += tokens
        elif event['event'] in ('graph_model_end', 'evidence_model'):
            values = event.get('usage') or []
            if isinstance(values, dict): values = [values]
            if not values: missing.append(event['event'])
            for usage in values:
                input_tokens += usage.get('input_tokens', 0)
                output_tokens += usage.get('output_tokens', 0)
    reserved = sum(e['event'] == 'request_reserved' for e in events)
    completed = sum(e['event'] in ('pdf_rerank_usage', 'graph_model_end', 'evidence_model') for e in events)
    unreported = max(0, reserved - completed)
    if unreported: missing.append(f'{unreported} reserved requests returned no usage event')
    rates = RATES.get(model)
    cost = ((input_tokens*rates[0] + output_tokens*rates[1] + rerank_tokens*0.5)/1e6
            if rates else None)
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'rerank_input_tokens': rerank_tokens, 'estimated_cny': cost,
        'missing_usage': missing, 'rates_date': '2026-10-28',
        'rates_scope': 'China list prices, <=128K input per chat request; no free quota/cache discounts',
        'rates_source': 'https://help.aliyun.com/zh/model-studio/model-pricing',
        'not_account_bill': True}


async def main(args):
    load_dotenv(args.env_file)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary = {'mode': args.mode, 'model': args.model, 'success': False,
               'synthetic_fixture_only': True}
    try:
        preflight_pdf_backend('api-rerank', args.pdf)
        if args.mode == 'rank':
            run = EvidenceRun(root/'research', root/'research/budget.json',
                secrets=[os.getenv('OPENAI_API_KEY'), os.getenv('PRISM_RERANK_API_KEY')])
            run.pdf_backend = 'api-rerank'
            run.ledger.write_text(json.dumps({'approved': True, 'requests': 0, 'searches': 0,
                'input_chars': 0, 'reserved_output_tokens': 0,
                'limits': {'requests': 1, 'searches': 0, 'input_chars': 10000, 'output_tokens': 0}}))
            doc = index_pdf(run, args.pdf)
            token = ACTIVE_RUN.set(run)
            try:
                ranked = await pdf_retrieval.ainvoke({'document': doc, 'question': args.question, 'top_k': 2})
            finally:
                ACTIVE_RUN.reset(token)
            summary.update(ranking=[{'page': p['page'], 'score': score} for score, p in ranked])
            if args.expected_page and ranked[0][1]['page'] != args.expected_page:
                raise AssertionError('Top-ranked page did not match expected page')
        else:
            options = SimpleNamespace(question=args.question, image=None, pdf=args.pdf, web=False, no_memory=True,
                web_screenshot=[], model=args.model, planner='native', pdf_backend='api-rerank',
                env_file=args.env_file, max_requests=args.max_requests, output=str(root/'research'))
            path = await research(options)
            report = json.loads(path.with_suffix('.json').read_text('utf-8'))
            if not report['findings'] or not report['evidence']:
                raise AssertionError('Agent returned an empty report')
            pages = sorted({v['page'] for v in report['evidence'].values() if 'page' in v})
            if args.expected_page and args.expected_page not in pages:
                raise AssertionError('Expected evidence page was not cited')
            summary.update(report=str(path), cited_pages=pages, findings=report['findings'],
                           unknowns=report['unknowns'])
        summary['success'] = True
    except Exception as error:
        # Do not print raw SDK/HTTP exceptions, which may contain request data.
        summary.update(error_type=type(error).__name__)
        if isinstance(error, AssertionError): summary['check_failure'] = str(error)
    events_path = root/'research/events.jsonl'
    events = [json.loads(row) for row in events_path.read_text('utf-8').splitlines()] if events_path.exists() else []
    summary['cost'] = usage_cost(events, args.model)
    ledger = root/'research/budget.json'
    if ledger.exists(): summary['requests'] = json.loads(ledger.read_text('utf-8'))['requests']
    summary['retrieval'] = [e['result']['retrieval'] for e in events if e['event'] == 'document_answer']
    summary['retrieval_methods'] = [e['result']['retrieval_method'] for e in events if e['event'] == 'document_answer']
    (root/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary['success']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['rank', 'agent'], default='rank')
    parser.add_argument('--pdf', required=True, help='Existing synthetic PDF, not private data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--model', choices=list(RATES), default='openai:qwen3.5-flash')
    parser.add_argument('--question', default=QUESTION)
    parser.add_argument('--expected-page', type=int)
    parser.add_argument('--max-requests', type=int, choices=range(1,17), default=16)
    raise SystemExit(0 if asyncio.run(main(parser.parse_args())) else 1)
