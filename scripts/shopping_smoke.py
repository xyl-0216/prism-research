"""Bounded live shopping workflow check. Synthetic label, real public web sources.

python scripts/shopping_smoke.py --env-file PATH --output runs/UNIQUE_NAME
Consumes API quota; does not write project memory or make purchases.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import fitz
from open_deep_research.cli import research


async def main(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    image=root/'synthetic-product-label.png'
    with fitz.open() as doc:
        page=doc.new_page(width=720,height=360)
        page.insert_text((25,50),'SYNTHETIC LABEL - NOT A PRODUCT PHOTOGRAPH',fontsize=18)
        page.insert_text((25,130),'Apple AirPods 5',fontsize=32)
        page.insert_text((25,200),'Variant: standard charging case',fontsize=22)
        page.insert_text((25,280),'No price or performance claims in this fixture.',fontsize=18)
        page.get_pixmap().save(image)
    options=SimpleNamespace(question=args.question,image=str(image),pdf=None,web=True,
        shopping=True,shopping_site=['www.apple.com.cn','item.jd.com','detail.tmall.com'],
        purchase_budget=1200,usage='日常通勤，使用苹果手机，关注降噪和舒适性',
        max_searches=2,web_screenshot=[],model='openai:qwen3.5-flash',planner='native',
        env_file=args.env_file,no_memory=True,max_requests=16,output=str(root/'research'))
    summary={'success':False,'mode':'live shopping smoke','synthetic_image':True}
    try:
        path=await research(options)
        report=json.loads(path.with_suffix('.json').read_text('utf-8'))
        shopping=report.get('shopping',{})
        checks={'identified':bool(shopping.get('identity')),
            'source_read':bool(shopping.get('sources')),
            'purchase_summary':bool(report.get('purchase_summary')),
            'value_summary':bool(report.get('value_summary')),
            'validated_summary':report.get('generation_status')!='evidence_only',
            'html_exists':path.is_file()}
        summary.update(success=all(checks.values()),checks=checks,report=str(path),
            offers=len(shopping.get('offers',[])),
            observed_purchase_prices=sum(o.get('price_status')=='page_observed' for o in shopping.get('offers',[])),
            limitations=shopping.get('limitations'))
    except Exception as error: summary['error_type']=type(error).__name__
    ledger=root/'research/budget.json'
    if ledger.exists():
        data=json.loads(ledger.read_text('utf-8'))
        summary.update(requests=data['requests'],searches=data['searches'])
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary['success']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--question',default='识别图片中的商品，研究其主要功能和适用人群，查找购买来源与页面价格，并结合预算和用途总结性价比。可参考 https://www.apple.com.cn/shop/buy-airpods/airpods-5 ，同时尝试京东或天猫；无法读取或无法确认同款时说明。')
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(args)) else 1)
