"""Self-contained offline demo with synthetic sources and no model credentials."""
import asyncio
import json
from pathlib import Path

import fitz

from open_deep_research.document_retrieval import index_pdf, render_page, validate_document_answer
from open_deep_research.evidence_runtime import EvidenceRun
from open_deep_research.mcsearch_adapter import evidence_chain
from open_deep_research.reporting import write_report_html


def create_demo_pdf(path):
    with fitz.open() as doc:
        page=doc.new_page(width=595,height=842)
        page.insert_text((48,70),'Prism Research / Synthetic research fixture',fontsize=20)
        page.insert_textbox(fitz.Rect(48,115,540,220),
            'This document is a fabricated example for testing evidence navigation.\n'
            'It is not a real benchmark or a product claim.\n\n'
            'Question: Does this chart prove a 2x gain without frame generation?',fontsize=12)
        page=doc.new_page(width=595,height=842)
        page.insert_text((48,70),'Experiment conditions',fontsize=22)
        page.insert_textbox(fitz.Rect(48,105,545,180),
            'Resolution: 1440p\nFrame generation: enabled\nCoverage: selected games only',fontsize=13)
        page.draw_rect(fitz.Rect(75,230,150,470),color=(.4,.5,.6),fill=(.4,.5,.6))
        page.draw_rect(fitz.Rect(190,150+80,265,470),color=(.3,.55,.8),fill=(.3,.55,.8))
        page.insert_text((70,500),'Baseline',fontsize=12)
        page.insert_text((185,500),'Candidate',fontsize=12)
        page.insert_textbox(fitz.Rect(48,560,545,650),
            'The illustration does not report performance with frame generation disabled.\n'
            'No universal claim about all games can be established from this fixture.',fontsize=12)
        doc.save(path)


async def run_demo(output):
    output=Path(output)
    run=EvidenceRun(output,output/'unused-budget.json')
    pdf=output/'synthetic-study.pdf';create_demo_pdf(pdf)
    doc=index_pdf(run,pdf)
    # Synthetic presentation fixture, not a retrieval benchmark or model fallback.
    page=doc['pages'][1]
    image=render_page(doc,page,output/'page_images')
    region=next(r for r in page['regions'] if 'Frame generation' in r['text'])
    replies=iter([
        '<Sub-Question>\n文档中的实验条件是什么？\n<Search>\nLocal Document Retrieval: frame generation resolution coverage\n<Dependencies>\n[]',
        json.dumps({'status':'insufficient','intermediate_answer':'示例条件启用了帧生成，只覆盖选定游戏。',
                    'evidence_ids':[region['id']],'missing_evidence':'缺少关闭帧生成的配对测试。'},ensure_ascii=False),
        '<End>\nFinal Answer: 保留缺口，需要补充匹配条件的测试。',
    ])
    async def infer(*args): return next(replies)
    async def execute(modality,question):
        return [{'id':region['id'],'text':region['text']}]
    def save(trace):
        (output/'mcsearch_trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),'utf-8')
    trace=await evidence_chain('关闭帧生成后，性能是否翻倍？',infer=infer,execute=execute,
        save=save,image_available=False,document_available=True,text_available=False)
    citation=validate_document_answer({'answer':'演示','region_ids':[region['id']]},[page])[0]
    citation.update(page_image=image['raw_path'],url=f"document:{doc['id']}#page=2",text=region['text'])
    value={'mode':'离线合成演示，非真实模型研究结果',
           'findings':[{'text':'合成资料显示测试启用了帧生成，不能据此确认关闭帧生成后的性能。',
                        'evidence_ids':[region['id']]}],
           'unknowns':['缺少匹配条件的逐项测试。'], 'evidence':{region['id']:citation},'trace':trace}
    (output/'report.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8')
    write_report_html(value,output/'report.html',mode=value['mode'])
    return output/'report.html'
