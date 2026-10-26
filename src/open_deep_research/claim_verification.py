"""Conditional evidence checks and a bounded chart report, without paid I/O.

This is an explicit first-case adapter, not an open-domain entailment judge.
Image transcriptions are VLM observations, not independently verified pixels.
Only source-anchored fields reach the renderer; search summaries do not.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Claim:
    subject: str
    reference: str
    metric: str
    population: str
    minimum_ratio: float
    conditions: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Measurement:
    source_id: str
    subject: str
    reference: str
    metric: str
    population: str
    ratio: float | None
    conditions: dict
    explicit_label: bool = True
    independently_measured: bool = False


def check_claim(claim: Claim, measurements: list[Measurement]) -> dict:
    """Support needs an explicit value and matching scope/conditions.

    Missing observations mean insufficient, never proof that the claim is false.
    Universal claims require universal coverage; finite benchmarks cannot supply it.
    """
    decisions = []
    for item in measurements:
        reasons = []
        for name in ('subject', 'reference', 'metric', 'population'):
            if getattr(claim, name) != getattr(item, name):
                reasons.append(name+'_mismatch')
        for name, value in claim.conditions.items():
            if name not in item.conditions:
                reasons.append('condition_missing:'+name)
            elif item.conditions[name] != value:
                reasons.append('condition_mismatch:'+name)
        if not item.explicit_label or item.ratio is None:
            reasons.append('no_explicit_numeric_measurement')
        decisions.append({'source_id':item.source_id, 'reasons':reasons,
                          'status':'inapplicable' if reasons else
                          ('supports' if item.ratio >= claim.minimum_ratio else 'contradicts'),
                          'independent':item.independently_measured})
    statuses = {item['status'] for item in decisions}
    status = ('conflicting' if {'supports','contradicts'} <= statuses else
              'supported' if 'supports' in statuses else
              'contradicted' if 'contradicts' in statuses else 'insufficient')
    return {'claim':asdict(claim), 'status':status, 'evidence':decisions,
            'independent_validation':any(d['status']=='supports' and d['independent'] for d in decisions)}


def _anchor(source_id, text, quote, modality, **extra):
    start = text.find(quote)
    if not quote or start < 0:
        raise EvidenceError('Evidence quote is absent from archived source')
    return {'source_id':source_id, 'quote':quote, 'start':start, 'end':start+len(quote),
            'modality':modality, **extra}


def _one(pattern, text, label, flags=re.I):
    matches = list(re.finditer(pattern, text, flags))
    if len(matches) != 1:
        raise EvidenceError(f'{label}: expected one unambiguous transcription, got {len(matches)}')
    return matches[0]


def build_chart_evidence(run):
    """Extract only literal, bounded facts from the first NVIDIA case.

    Legacy observations are used as archived VLM output, not as human truth labels.
    Fresh VLM calls keep the same JSON observation schema. Unknown or ambiguous
    transcriptions fail closed instead of being replaced with the expected answer.
    """
    from open_deep_research.evidence_tools import NVIDIA_URL
    page = next((p for p in run.pages.values() if p['url']==NVIDIA_URL), None)
    image = run.images.get('nvidia_chart')
    observations = getattr(run, 'visual_observations', [])
    if not page or not image or not observations:
        raise EvidenceError('Original page, image and actual visual observations are required')
    raw = Path(image['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != image['sha256']:
        raise EvidenceError('Image hash mismatch')
    candidates = [v for v in observations if v.get('asset_id')=='nvidia_chart'
                  and v.get('image_sha256')==image['sha256']
                  and v.get('image_url')==image['url'] and v.get('source_page')==page['url']
                  and v.get('finish_reason')=='stop']
    if len(candidates)!=1:
        raise EvidenceError('Need one successful, provenance-matched visual observation')
    observation = candidates[0]
    try:
        raw_observation = observation['observations'].strip()
        # Accept a single JSON code fence, but never scrape a JSON substring out
        # of arbitrary prose: surrounding instructions or multiple blocks fail.
        fenced = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', raw_observation, re.I)
        parsed = json.loads(fenced[1] if fenced else raw_observation)
    except (ValueError, TypeError) as exc:
        raise EvidenceError('Visual observation must be JSON') from exc
    if not isinstance(parsed, dict):
        raise EvidenceError('Visual observation must be an object')
    # Comparison prose contained estimated bars and is excluded. The legacy
    # uncertainty field explicitly identifies FP4 as the AI mode; only that
    # literal phrase is extracted, not its other unverified statements.
    lines = parsed.get('observations', []) + parsed.get('conditions', []) + parsed.get('uncertainties', [])
    if not all(isinstance(line, str) for line in lines):
        raise EvidenceError('Visual fields must be string lists')
    visual_text = '\n'.join(lines)
    legend = _one(r'Green bar\s*=\s*(RTX\s*\d+),\s*Gray bar\s*=\s*(RTX\s*\d+)', visual_text, 'legend')
    if re.sub(r'\s+', '', legend[1]).upper() != 'RTX5070' or re.sub(r'\s+', '', legend[2]).upper() != 'RTX4070':
        raise EvidenceError('First-case adapter cannot attach RTX 5070 article claims to another comparison')
    footnote = _one(r'\d{3,4}p,\s*Max Settings,\s*DLSS Super Resolution and DLSS Ray Reconstruction on 40 and 50 Series;\s*Frame Gen on 40 Series\.\s*Multi Frame Gen \(4X Mode\) on 50 Series\.', visual_text, 'footnote')
    exception_note = _one(r'Horizon Forbidden West supports DLSS 3\.', visual_text, 'game-specific footnote')
    ai = _one(r"One explicit numerical value visible:\s*['\"]([0-9]+(?:\.[0-9]+)?X)['\"] above the last green bar \(Generative AI / ([^)]+)\)", visual_text, 'explicit AI label')
    ai_mode = _one(r'FP4 mode in Generative AI test', visual_text, 'AI precision condition')
    raster = _one(r'\b(\d+(?:\.\d+)?)% boost in traditional raster performance\b', page['text'], 'raster claim')
    fields = {
        'legend':_anchor('image', visual_text, legend.group(), 'image_transcription',
                         subject=legend[1], reference=legend[2]),
        'conditions':_anchor('image', visual_text, footnote.group(), 'image_transcription',
                             resolution=re.search(r'\d+p', footnote.group()).group()),
        'exception_note':_anchor('image', visual_text, exception_note.group(), 'image_transcription'),
        'ai_label':_anchor('image', visual_text, ai.group(), 'image_transcription',
                          ratio=float(ai[1][:-1]), workload=ai[2]),
        'ai_mode':_anchor('image', visual_text, ai_mode.group(), 'image_transcription', precision='FP4'),
        'raster':_anchor('page', page['text'], raster.group(), 'page_text', percent=float(raster[1])),
    }
    measurements = [
        Measurement('image:gaming', legend[1], legend[2], 'game_fps', 'charted_games', None,
                    {'frame_generation':'enabled', 'resolution':fields['conditions']['resolution']}, False),
        Measurement('image:ai_label', legend[1], legend[2], 'ai_throughput', ai[2],
                    fields['ai_label']['ratio'], {'precision':fields['ai_mode']['precision']}),
        Measurement('page:raster', legend[1], legend[2], 'raster_performance', 'unspecified',
                    1+fields['raster']['percent']/100, {}),
    ]
    claim = Claim(legend[1], legend[2], 'game_fps', 'all_games', 2.0, {'frame_generation':'disabled'})
    sources = {'image':{'url':image['url'], 'sha256':image['sha256'],
                         'transcript_sha256':hashlib.sha256(visual_text.encode()).hexdigest()},
               'page':{'url':page['url'], 'sha256':page['sha256'],
                       'text_sha256':hashlib.sha256(page['text'].encode()).hexdigest()}}
    return {'schema_version':1, 'adapter':'nvidia-first-case-v1', 'fields':fields,
            'sources':sources, 'visual_transcript':visual_text,
            'measurements':[asdict(m) for m in measurements], 'verification':check_claim(claim, measurements),
            'limitations':['VLM transcription not independently verified',
                           'Fixed first-case extraction grammar; not general chart understanding',
                           'Vendor claims are not independent measurements']}


def _cell(text):
    # Source text cannot create new Markdown rows or executable HTML.
    import html
    return html.escape(str(text)).replace('|','&#124;').replace('\n',' ')


def render_chart_report(bundle):
    """No free-form LLM prose can introduce facts after verification."""
    fields, sources, verification = bundle['fields'], bundle['sources'], bundle['verification']
    if verification['status'] != 'insufficient':
        raise EvidenceError('First-case renderer requires an insufficient-evidence finding')
    subject = _cell(fields['legend']['subject'])
    reference = _cell(fields['legend']['reference'])
    return (
        '# NVIDIA 图表条件核验报告\n\n'
        f"[1] {sources['page']['url']}\n\n[2] {sources['image']['url']}\n\n"
        '| 核验项 | 证据 | 适用条件与限制 |\n|---|---|---|\n'
        f'| 比较对象 | 绿色 {subject}；灰色 {reference} [2] | 图例来自视觉转录，未估算柱高 |\n'
        f"| 游戏测试脚注 | {_cell(fields['conditions']['quote'])} [2] | 包含不同代际的帧生成策略；不推广到 AI/视频工作负载 |\n"
        f"| 单项条件 | {_cell(fields['exception_note']['quote'])} [2] | 不把所有游戏概括为相同 DLSS 版本或功能 |\n"
        f"| 明确标注的数值 | {_cell(fields['ai_label']['workload'])}：{fields['ai_label']['ratio']:g}X [2] | AI 工作负载，精度条件 {_cell(fields['ai_mode']['precision'])}；不是游戏 FPS，不能外推到所有游戏 |\n"
        f"| 厂商正文描述 | 传统光栅性能提升 {fields['raster']['percent']:g}% [1] | 来自原文，不是柱高计算；没有逐游戏关闭帧生成的配对结果 |\n\n"
        f'**结论：证据不足，不能证明 {subject} 相对 {reference} 在不使用帧生成时，所有游戏帧率都翻倍。** '
        '现有证据的指标、覆盖范围或帧生成条件与该主张不一致；这不是证明该主张在每个游戏中都为假。'
        '来源是厂商材料，不能视为独立实测。图片内容依赖 VLM 转录，仍需人工核对原图；'
        '本报告由已核验字段确定性生成，未让写作模型补充柱高或因果解释。\n'
    )


def generate_verified_chart_report(run):
    bundle = build_chart_evidence(run)
    run.claim_verification = bundle
    (run.directory/'claim_verification.json').write_text(run.safe(bundle), 'utf-8')
    report = render_chart_report(bundle)
    run.log('claim_verification', status=bundle['verification']['status'],
            evidence_fields=list(bundle['fields']), renderer='deterministic_first_case')
    return report
