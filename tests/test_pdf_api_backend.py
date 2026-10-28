"""API contract, budget and citation tests; external network is blocked."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from open_deep_research import pdf_backend as backend, document_retrieval as documents
from open_deep_research.demo import create_demo_pdf
from open_deep_research.evidence_runtime import EvidenceRun, ACTIVE_RUN, BudgetExceeded


@pytest.fixture
def context(tmp_path):
    run = EvidenceRun(tmp_path/'run', tmp_path/'budget.json')
    run.pdf_backend = 'api-rerank'
    run.ledger.write_text(json.dumps({'approved': True, 'requests': 0, 'searches': 0,
        'input_chars': 0, 'reserved_output_tokens': 0,
        'limits': {'requests': 2, 'searches': 0, 'input_chars': 10000, 'output_tokens': 2000}}))
    source = tmp_path/'fixture.pdf'
    create_demo_pdf(source)
    document = documents.index_pdf(run, source)
    token = ACTIVE_RUN.set(run)
    try:
        yield run, document
    finally:
        ACTIVE_RUN.reset(token)


def fake_http(monkeypatch, *, status=200, rows=None, error=None):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['follow_redirects'] is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, endpoint, json, headers):
            calls.append(json)
            assert json['model'] == 'qwen3-vl-rerank'
            assert len(json['input']['documents']) == 2
            assert all(d['image'].startswith('data:image/png;base64,') for d in json['input']['documents'])
            assert json['parameters']['top_n'] == 2
            if error: raise error
            return httpx.Response(status, json={
                'output': {'results': rows if rows is not None else [
                    {'index': 0, 'relevance_score': .1}, {'index': 1, 'relevance_score': .9}]},
                'usage': {'total_tokens': 1234}, 'message': 'secret-provider-message'})
    monkeypatch.setattr(backend.httpx, 'Client', Client)
    return calls


def retriever():
    return backend.APIReranker(backend.RerankSettings('https://dashscope.aliyuncs.com/rerank', 'fixture-key'))


def test_real_page_mapping_budget_usage_and_cache(context, monkeypatch):
    run, doc = context
    calls = fake_http(monkeypatch)
    first = documents.rank_pages(doc, 'conditions', 1, retriever=retriever())
    assert first[0][1] is doc['pages'][1]
    assert doc['retrieval_method'] == backend.API_METHOD
    assert doc['retrieval_metadata']['usage'] == {'total_tokens': 1234}
    second = retriever().rank(doc, 'conditions', 2)
    assert [p['page'] for _, p in second] == [2, 1]
    assert doc['retrieval_metadata']['cache_hit']
    assert len(calls) == json.loads(run.ledger.read_text())['requests'] == 1
    events = (run.directory/'events.jsonl').read_text('utf-8')
    assert 'base64' not in events and 'fixture-key' not in events
    # A changed file must be rejected even when this query has a cached result.
    from pathlib import Path
    Path(doc['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        retriever().rank(doc, 'conditions', 2)
    assert len(calls) == 1


@pytest.mark.parametrize('rows', [
    [], [{'index': 0, 'relevance_score': .5}],
    [{'index': 0, 'relevance_score': .5}, {'index': 0, 'relevance_score': .6}],
    [{'index': 0, 'relevance_score': .5}, {'index': 2, 'relevance_score': .6}],
    [{'index': False, 'relevance_score': .5}, {'index': 1, 'relevance_score': .6}],
    [{'index': 0, 'relevance_score': float('nan')}, {'index': 1, 'relevance_score': .6}],
    [{'index': 0, 'relevance_score': True}, {'index': 1, 'relevance_score': .6}],
])
def test_invalid_api_results_rejected(rows):
    with pytest.raises(ValueError):
        backend.validate_results({'output': {'results': rows}}, [{'page': 1}, {'page': 2}])


def test_no_retry_or_ura_fallback_on_http_failure(context, monkeypatch):
    run, doc = context
    calls = fake_http(monkeypatch, status=403)
    with pytest.raises(RuntimeError, match='HTTP 403') as caught:
        retriever().rank(doc, 'conditions', 1)
    assert 'secret-provider-message' not in str(caught.value)
    assert len(calls) == json.loads(run.ledger.read_text())['requests'] == 1
    assert not getattr(run, '_pdf_rank_cache', {})


def test_budget_prevents_network_and_oversized_pdf_is_not_truncated(context, monkeypatch):
    run, doc = context
    calls = fake_http(monkeypatch)
    doc['pages'] = doc['pages'] * 21
    with pytest.raises(ValueError, match='1..40'):
        retriever().rank(doc, 'conditions', 1)
    doc['pages'] = doc['pages'][:2]
    ledger = json.loads(run.ledger.read_text()); ledger['requests'] = 2
    run.ledger.write_text(json.dumps(ledger))
    with pytest.raises(BudgetExceeded):
        retriever().rank(doc, 'conditions', 1)
    assert not calls


def test_api_preflight_and_selection_do_not_require_ura(context, monkeypatch):
    _, doc = context
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture')
    monkeypatch.setenv('OPENAI_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    for name in ('PYTHON', 'REPO', 'CONFIG', 'CHECKPOINT'):
        monkeypatch.delenv('PRISM_MPMQA_' + name, raising=False)
    backend.preflight_pdf_backend('api-rerank', doc['path'])
    assert isinstance(backend.get_retriever(), backend.APIReranker)
    with pytest.raises(RuntimeError, match='Configure the official MPMQA'):
        backend.preflight_pdf_backend('ura', doc['path'])
    assert 'fixture' not in repr(backend.RerankSettings.from_env())
    monkeypatch.setenv('PRISM_RERANK_BASE_URL', 'https://example.com/compatible-mode/v1')
    with pytest.raises(ValueError, match='official HTTPS'):
        backend.RerankSettings.from_env()


def test_api_tool_keeps_region_citations_and_backend_metadata(context, monkeypatch):
    run, doc = context
    fake_http(monkeypatch)
    monkeypatch.setattr(documents, 'get_retriever', retriever)
    region = doc['pages'][1]['regions'][1]
    def infer_factory(*args, **kwargs):
        assert kwargs['schema'] is documents.DocumentAnswer
        async def infer(stage, prompt, images):
            assert stage == 'document_answer'
            assert sum(x['type'] == 'image_url' for x in images) == 2
            return json.dumps({'answer': 'Synthetic conditions', 'region_ids': [region['id']]})
        return infer
    monkeypatch.setattr(documents, 'make_infer', infer_factory)
    result = json.loads(asyncio.run(documents.search_document_evidence.ainvoke({
        'document_id': doc['id'], 'question': 'conditions'})))
    assert result['retrieval'][0]['page'] == 2
    assert result['retrieval_method'] == backend.API_METHOD
    assert result['citations'][0]['bbox'] == region['bbox']
    assert result['citations'][0]['page'] == 2
    assert run.document_calls == 1


def test_report_schema_exposes_only_real_evidence_ids():
    from pydantic import ValidationError
    from open_deep_research.reporting import report_schema
    schema = report_schema({'doc:p2:r1': {}, 'doc:p3:r2': {}})
    good = {'findings': [{'text': 'Visible fact', 'evidence_ids': ['doc:p2:r1']}], 'unknowns': []}
    assert schema.model_validate(good).findings[0].evidence_ids == ['doc:p2:r1']
    good['findings'][0]['evidence_ids'] = ['fabricated']
    with pytest.raises(ValidationError):
        schema.model_validate(good)
