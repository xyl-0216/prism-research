"""Bounded real Agent smoke checks; opt-in, uses the user's model API quota.

python scripts/live_smoke.py --case image --env-file PATH --output runs/CHECK
Synthetic image content is for wiring checks, not a real product or benchmark.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import fitz
from dotenv import dotenv_values

from open_deep_research.cli import research


def create_image(path):
    with fitz.open() as doc:
        page = doc.new_page(width=720, height=420)
        page.insert_text((35, 55), 'SYNTHETIC TEST FIXTURE - NOT A REAL PRODUCT', fontsize=18)
        page.insert_text((35, 120), 'Model: TEST-A', fontsize=24)
        page.insert_text((35, 185), 'Operating temperature: -10 C to +50 C', fontsize=22)
        page.insert_text((35, 245), 'Storage temperature: -30 C to +70 C', fontsize=22)
        page.insert_text((35, 335), 'Storage conditions do not establish operating conditions.', fontsize=17)
        page.get_pixmap(alpha=False).save(path)


async def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = dotenv_values(args.env_file) if args.env_file else {}
    names = ('OPENAI_API_KEY', 'OPENAI_API_BASE', 'OPENAI_BASE_URL')
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        if not os.getenv(name) and settings.get(name):
            os.environ[name] = settings[name]
    os.environ['LANGSMITH_TRACING'] = 'false'
    summary = {'case': args.case, 'planner': args.planner, 'model':args.model,
               'mode': 'live API smoke check', 'success': False,
               'max_requests': args.max_requests}
    try:
        image = None
        if args.case == 'image':
            image = root / 'synthetic-spec.png'
            create_image(image)
            question = '请读取这张合成设备说明图：TEST-A 能否在零下20摄氏度运行？区分工作温度和储存温度，引用图片证据，不要联网或推测未提供的规格。'
        else:
            question = '请检索 Microsoft Learn 官方文档，说明如何使用 LangChain.js 和 Azure AI Search 构建检索增强生成应用。只引用 learn.microsoft.com 官方原文，证据不足时说明。'
        options = SimpleNamespace(image=str(image) if image else None, pdf=None,
            no_memory=True,
            web=args.case=='web', web_screenshot=[], model=args.model, planner=args.planner,
            question=question, max_requests=args.max_requests, output=str(root/'research'))
        report = await research(options)
        value = json.loads(report.with_suffix('.json').read_text(encoding='utf-8'))
        trace = value.get('trace', {})
        if not value.get('findings') or not value.get('evidence'):
            raise AssertionError('Report has no grounded findings; an empty report is not a successful smoke check')
        if args.planner == 'mcsearch' and trace.get('status') != 'completed':
            raise AssertionError('MC-Search did not complete its evidence chain')
        summary.update(success=True, report=str(report), status=trace.get('status'),
            actions=[{'modality': s['modality'], 'status': s['status'],
                      'evidence_ids': s['evidence_ids']} for s in trace.get('steps', [])],
            findings=value.get('findings'), unknowns=value.get('unknowns'))
    except Exception as error:
        message = str(error)
        for name in names:
            if name.endswith('KEY') and os.getenv(name):
                message = message.replace(os.environ[name], '[REDACTED]')
        message = re.sub(r'(?i)Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', message)
        summary.update(error_type=type(error).__name__, error=message[:3000])
    finally:
        ledger = root/'research/budget.json'
        if ledger.exists():
            budget = json.loads(ledger.read_text(encoding='utf-8'))
            summary['requests'] = budget['requests']
            summary['searches'] = budget['searches']
        (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        for name, value in previous.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary['success']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['image','web'], required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='openai:qwen3.5-flash')
    parser.add_argument('--planner', choices=['native','mcsearch'], default='native')
    parser.add_argument('--max-requests', type=int, choices=range(1,17), default=16)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
