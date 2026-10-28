"""Selectable PDF page ranking: original URA or a bounded DashScope image API.

The API ranks rendered pages directly; it does not reproduce URA or generate
citation coordinates. Original page IDs/regions remain owned by the PDF parser.
"""
import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import fitz
import httpx
from langchain_core.runnables import RunnableLambda

from open_deep_research.evidence_runtime import ACTIVE_RUN, active_run


BACKENDS = ('ura', 'api-rerank')
API_MODEL = 'qwen3-vl-rerank'
API_METHOD = 'DashScope qwen3-vl-rerank (rendered PDF pages; not original URA)'
MAX_API_PAGES = 40


def backend_name(explicit=None):
    run = ACTIVE_RUN.get()
    name = explicit or getattr(run, 'pdf_backend', None) or os.getenv('PRISM_PDF_BACKEND', 'ura')
    if name not in BACKENDS:
        raise ValueError('PRISM_PDF_BACKEND must be ura or api-rerank')
    return name


@dataclass(frozen=True)
class RerankSettings:
    endpoint: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls):
        from open_deep_research.bailian_search import native_endpoint
        base = (os.getenv('PRISM_RERANK_BASE_URL') or os.getenv('OPENAI_API_BASE')
                or os.getenv('OPENAI_BASE_URL') or '')
        # Reuse validation of official HTTPS DashScope/MAAS origins. Never send
        # the model key to a guessed third-party endpoint or follow redirects.
        parsed = urlsplit(native_endpoint(base))
        key = os.getenv('PRISM_RERANK_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError('Configure PRISM_RERANK_API_KEY or OPENAI_API_KEY')
        endpoint = urlunsplit((parsed.scheme, parsed.netloc,
            '/api/v1/services/rerank/text-rerank/text-rerank', '', ''))
        return cls(endpoint, key)


def preflight_pdf_backend(name, path):
    """Fail on missing configuration/page limits before paid Agent planning."""
    name = backend_name(name)
    if name == 'ura':
        from open_deep_research.mpmqa_backend import URASettings
        URASettings.from_env()
        return
    RerankSettings.from_env()
    with fitz.open(path) as pdf:
        if pdf.needs_pass or not 1 <= len(pdf) <= MAX_API_PAGES:
            raise ValueError('API reranking requires an unencrypted PDF with 1..40 pages; '
                             'longer PDFs need an embedding recall stage first')


def validate_results(payload, pages):
    output = payload.get('output') if isinstance(payload, dict) else None
    rows = output.get('results') if isinstance(output, dict) else None
    # Request ALL scores to detect missing/duplicated pages before taking top_k.
    if not isinstance(rows, list) or len(rows) != len(pages):
        raise ValueError('Rerank API must return one score per page')
    seen, ranked = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid rerank result')
        index, score = row.get('index'), row.get('relevance_score')
        if (type(index) is not int or not 0 <= index < len(pages) or index in seen
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Invalid rerank page index or score')
        seen.add(index)
        ranked.append((float(score), pages[index]))
    return sorted(ranked, key=lambda pair: (-pair[0], pair[1]['page']))


class APIReranker:
    method = API_METHOD

    def __init__(self, settings):
        self.settings = settings

    def rank(self, document, question, top_k):
        if not question.strip() or len(question) > 2000 or not 1 <= top_k <= 3:
            raise ValueError('Invalid document query')
        pages = document['pages']
        if not 1 <= len(pages) <= MAX_API_PAGES:
            raise ValueError('API reranking supports 1..40 pages; no pages were silently omitted')
        raw = Path(document['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != document['sha256']:
            raise ValueError('PDF changed after indexing')
        run = active_run()
        key = (document['sha256'], question, self.settings.endpoint, API_MODEL)
        cache = getattr(run, '_pdf_rank_cache', {})
        if key in cache:
            ranked, metadata = cache[key]
            document['retrieval_metadata'] = {**metadata, 'cache_hit': True}
            run.log('pdf_rerank_cache_hit', document_id=document['id'], model=API_MODEL)
            return ranked[:top_k]

        candidates = []
        with fitz.open(stream=raw, filetype='pdf') as pdf:
            for page in pages:
                original = pdf[page['page'] - 1]
                scale = min(1.5, 1120 / max(original.rect.width, original.rect.height))
                pix = original.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                         colorspace=fitz.csRGB, alpha=False)
                encoded = base64.b64encode(pix.tobytes('png')).decode('ascii')
                candidates.append({'image': 'data:image/png;base64,' + encoded})
        body = {'model': API_MODEL, 'input': {'query': {'text': question},
                'documents': candidates}, 'parameters': {'top_n': len(pages),
                                                        'return_documents': False}}
        # Request accounting is owned here, not by inherited model callbacks.
        # input_chars is a text bound, NOT an image-token or monetary estimate.
        run.reserve('pdf_rerank', len(question) * len(pages), 0)
        try:
            with httpx.Client(timeout=httpx.Timeout(90, connect=15), follow_redirects=False) as client:
                response = client.post(self.settings.endpoint, json=body,
                    headers={'Authorization': 'Bearer ' + self.settings.api_key})
        except httpx.TransportError as error:
            raise RuntimeError('PDF rerank transport failed: ' + type(error).__name__) from None
        if response.status_code != 200:
            # Provider bodies can echo credentials or large base64 inputs.
            raise RuntimeError(f'PDF rerank HTTP {response.status_code}; check model access, region and input limits')
        payload = response.json()
        usage = payload.get('usage', {}) if isinstance(payload, dict) else {}
        run.log('pdf_rerank_usage', model=API_MODEL, pages=len(pages), usage=usage)
        ranked = validate_results(payload, pages)
        metadata = {'backend': 'api-rerank', 'model': API_MODEL, 'input_modality': 'page_image',
            'cache_hit': False, 'usage': usage,
            'pages': [{'id': p['id'], 'page': p['page'], 'score': score} for score, p in ranked],
            'limitations': ['Scores rank relevance, not answer correctness; original URA is not used.',
                            'Direct page reranking is limited to 40 pages; no embedding stage yet.']}
        document['retrieval_metadata'] = metadata
        cache[key] = (ranked, metadata)
        run._pdf_rank_cache = cache
        run.log('pdf_rerank_completed', document_id=document['id'], **metadata)
        return ranked[:top_k]


def get_retriever():
    if backend_name() == 'api-rerank':
        return APIReranker(RerankSettings.from_env())
    from open_deep_research.mpmqa_backend import get_retriever as get_ura
    return get_ura()


async def _retrieve(request):
    from open_deep_research.document_retrieval import rank_pages
    return await asyncio.to_thread(rank_pages, request['document'], request['question'],
                                   request.get('top_k', 2))


pdf_retrieval = RunnableLambda(_retrieve, name='pdf_page_retrieval')
