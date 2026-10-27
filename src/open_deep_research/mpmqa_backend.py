"""LangChain boundary for the author's URA model in an isolated legacy environment.

The worker imports MQAT5Model from the pinned official checkout. No lexical or
random-weight fallback is allowed. Heavy torch/detectron2 dependencies stay out
of the Agent's Python environment.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import fitz
from langchain_core.runnables import RunnableLambda

AUTHOR_COMMIT = '5226a9aa849fd3b8f35620e7b7d7d10c09d754a2'
METHOD = 'MPMQA URA learned multimodal retrieval (official MQAT5Model)'


@dataclass(frozen=True)
class URASettings:
    python: Path
    repository: Path
    config: Path
    checkpoint: Path
    device: str = 'cuda:0'
    timeout: int = 600

    @classmethod
    def from_env(cls):
        names = ['PYTHON', 'REPO', 'CONFIG', 'CHECKPOINT']
        values = [os.getenv('PRISM_MPMQA_' + name) for name in names]
        if not all(values):
            missing = ', '.join('PRISM_MPMQA_' + n for n, v in zip(names, values) if not v)
            raise RuntimeError('Configure the official MPMQA runtime: ' + missing +
                               '. See docs/runtime.md; lexical fallback is disabled.')
        settings = cls(*(Path(v).expanduser().resolve() for v in values),
                       device=os.getenv('PRISM_MPMQA_DEVICE', 'cuda:0'),
                       timeout=int(os.getenv('PRISM_MPMQA_TIMEOUT', '600')))
        settings.validate()
        return settings

    def validate(self):
        for label, path in [('Python', self.python), ('URA config', self.config),
                            ('consolidated checkpoint', self.checkpoint),
                            ('author model', self.repository / 'models/mqa_model.py'),
                            ('author ROI extractor', self.repository / 'detector/ROIFeatExtractor.py')]:
            if not path.is_file():
                raise RuntimeError('Missing MPMQA ' + label + ': ' + str(path))
        if self.timeout <= 0:
            raise ValueError('MPMQA timeout must be positive')


def prepare_request(document, question, directory):
    """Render unannotated RGB pages; the author worker converts them to BGR.

    Word and region coordinates use the SAME rendered-page pixel coordinate
    system. Layout labels are parser-derived, not PM209 ground-truth labels.
    """
    raw = Path(document['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != document['sha256']:
        raise ValueError('PDF changed after indexing')
    pages = []
    with fitz.open(stream=raw, filetype='pdf') as pdf:
        for page in document['pages']:
            original = pdf[page['page'] - 1]
            scale = min(1.5, 1400 / max(original.rect.width, original.rect.height))
            pix = original.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            path = Path(directory) / ('page-%d.png' % page['page'])
            pix.save(path)

            def pixels(box):
                return [box[0] * pix.width, box[1] * pix.height,
                        box[2] * pix.width, box[3] * pix.height]

            regions = []
            for region in page['regions']:
                # Semantic segmentation is not supplied by arbitrary PDFs.
                # Preserve supplied labels; otherwise use only coarse parser types.
                semantic = region.get('semantic_class') or {
                    'text': 'Text', 'image': 'illustration',
                    'graphic': 'graphic', 'page': 'illustration',
                }[region['kind']]
                regions.append({'id': region['id'], 'structure': semantic,
                                'bbox': pixels(region['bbox']),
                                'words': [{'text': word['text'], 'bbox': pixels(word['bbox'])}
                                          for word in region.get('words', [])]})
            pages.append({'id': page['id'], 'page': page['page'], 'image': str(path),
                          'regions': regions})
    return {'question': question, 'pages': pages, 'document_sha256': document['sha256']}


def validate_scores(value, document):
    rows = value.get('scores') if isinstance(value, dict) else None
    allowed = {page['id']: page for page in document['pages']}
    if not isinstance(rows, list) or len(rows) != len(allowed):
        raise ValueError('URA must return one score per page')
    seen, scored = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid URA score record')
        page_id, score = row.get('id'), row.get('score')
        if (not isinstance(page_id, str) or page_id not in allowed or page_id in seen
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Invalid URA page ID or score')
        seen.add(page_id)
        scored.append((float(score), allowed[page_id]))
    return sorted(scored, key=lambda item: (-item[0], item[1]['page']))


class AuthorURARetriever:
    def __init__(self, settings):
        self.settings = settings

    def rank(self, document, question, top_k):
        self.settings.validate()
        with tempfile.TemporaryDirectory(prefix='prism-ura-') as tmp:
            root = Path(tmp)
            request = prepare_request(document, question, root)
            request.update(config=str(self.settings.config), checkpoint=str(self.settings.checkpoint),
                           repository=str(self.settings.repository), device=self.settings.device)
            request_path, response_path = root / 'request.json', root / 'response.json'
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
            command = [str(self.settings.python), str(Path(__file__).with_name('mpmqa_worker.py')),
                       '--request', str(request_path), '--response', str(response_path)]
            # This local model does not need the Agent's API credentials.
            env = {k: v for k, v in os.environ.items()
                   if not any(s in k.upper() for s in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
            env.update(PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            try:
                result = subprocess.run(command, cwd=self.settings.repository, env=env,
                                        capture_output=True, text=True, encoding='utf-8', errors='replace',
                                        timeout=self.settings.timeout, check=False)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('MPMQA local inference timed out') from error
            if result.returncode != 0:
                raise RuntimeError('MPMQA author runtime failed; check weights and legacy dependencies:\n'
                                   + (result.stderr or result.stdout)[-4000:])
            if not response_path.is_file():
                raise RuntimeError('MPMQA runtime returned no response')
            value = json.loads(response_path.read_text(encoding='utf-8'))
            ranked = validate_scores(value, document)[:top_k]
            document['retrieval_metadata'] = {
                'retrieval_type': value.get('retrieval_type'),
                'layout_source': value.get('layout_source'),
                'pages': value['scores'],
            }
            return ranked


def get_retriever():
    return AuthorURARetriever(URASettings.from_env())


async def _retrieve(request):
    from open_deep_research.document_retrieval import rank_pages
    return await asyncio.to_thread(rank_pages, request['document'],
                                   request['question'], request.get('top_k', 2))


mpmqa_retrieval = RunnableLambda(_retrieve, name='mpmqa_ura_page_retrieval')
