"""Offline adapter tests, not a measurement of pretrained URA retrieval quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import fitz
import pytest

from open_deep_research import mpmqa_backend as backend
from open_deep_research.document_retrieval import index_pdf, rank_pages
from open_deep_research.evidence_runtime import EvidenceRun
from open_deep_research.mpmqa_worker import build_page_tokens, checkpoint_state, load_constants, load_model


@pytest.fixture
def document(tmp_path):
    source = tmp_path / 'manual.pdf'
    with fitz.open() as doc:
        for text in ['Battery temperature', 'Power supply']:
            page = doc.new_page(width=400, height=600)
            page.insert_text((40, 70), text)
        doc.save(source)
    run = EvidenceRun(tmp_path / 'run', tmp_path / 'unused.json')
    return index_pdf(run, source)


def test_native_words_and_render_coordinates(document, tmp_path):
    request = backend.prepare_request(document, 'working conditions', tmp_path)
    page = request['pages'][0]
    words = [w for r in page['regions'] for w in r['words']]
    assert [w['text'] for w in words] == ['Battery', 'temperature']
    image = fitz.Pixmap(page['image'])
    for region in page['regions']:
        for box in [region['bbox']] + [w['bbox'] for w in region['words']]:
            a, b, c, d = box
            assert 0 <= a < c <= image.width
            assert 0 <= b < d <= image.height
    native_word = document['pages'][0]['regions'][0]['words'][0]
    assert words[0]['bbox'][0] == pytest.approx(native_word['bbox'][0] * image.width)
    Path(document['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        backend.prepare_request(document, 'question', tmp_path)


def test_subwords_repeat_word_box_and_segment():
    constants = SimpleNamespace(VRM_SEMANTIC_CLS2TOKEN={'Text': '<Text>'},
                                VRM_SEMANTIC_CLS2ID={'Text': 1})
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return {'<Text>': [50], 'battery': [10, 11], 'limit': [12]}[text]
    regions = [{'structure': 'Text', 'bbox': [0, 0, 100, 100], 'words': [
        {'text': 'battery', 'bbox': [10, 10, 40, 20]},
        {'text': 'limit', 'bbox': [50, 10, 70, 20]},
    ]}]
    ids, boxes, segments, truncated = build_page_tokens(regions, Tokenizer(), constants, 4)
    assert ids == [50, 10, 11, 12]
    assert boxes[1] == boxes[2] == [10, 10, 40, 20]
    assert segments == [1, 1, 1, 1] and not truncated
    ids, boxes, segments, truncated = build_page_tokens(regions, Tokenizer(), constants, 2)
    assert ids == [50, 10] and len(boxes) == len(segments) == 2 and truncated


@pytest.mark.parametrize('rows', [
    [], [{'id': 'fabricated', 'score': 1}, {'id': 'other', 'score': 2}],
    [{'id': 'a', 'score': 1}, {'id': 'a', 'score': 2}],
    [{'id': 'a', 'score': float('nan')}, {'id': 'b', 'score': 0}],
    [{'id': 'a', 'score': True}, {'id': 'b', 'score': 0}],
])
def test_worker_scores_fail_closed(rows):
    with pytest.raises(ValueError):
        backend.validate_scores({'scores': rows}, {'pages': [{'id': 'a'}, {'id': 'b'}]})


def test_scores_sort_semantics_not_lexical_overlap():
    doc = {'pages': [{'id': 'a', 'page': 1, 'text': 'query query'},
                     {'id': 'b', 'page': 2, 'text': 'different wording'}]}
    value = {'scores': [{'id': 'a', 'score': -0.1}, {'id': 'b', 'score': 0.7}]}
    assert backend.validate_scores(value, doc)[0][1]['id'] == 'b'


def test_no_lexical_fallback(document, monkeypatch):
    for name in ('PYTHON', 'REPO', 'CONFIG', 'CHECKPOINT'):
        monkeypatch.delenv('PRISM_MPMQA_' + name, raising=False)
    with pytest.raises(RuntimeError, match='lexical fallback is disabled'):
        rank_pages(document, 'Battery')


def test_process_contract(document, tmp_path, monkeypatch):
    repo = tmp_path / 'author'
    (repo / 'models').mkdir(parents=True)
    (repo / 'detector').mkdir()
    for file in ('models/mqa_model.py', 'detector/ROIFeatExtractor.py', 'config.json', 'weights.pt'):
        (repo / file).write_text('fixture', encoding='utf-8')
    settings = backend.URASettings(Path(sys.executable), repo, repo / 'config.json', repo / 'weights.pt')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-leak')

    def fake_process(command, **kwargs):
        assert kwargs['cwd'] == repo and 'OPENAI_API_KEY' not in kwargs['env']
        request = json.loads(Path(command[3]).read_text(encoding='utf-8'))
        assert request['pages'][0]['regions'][0]['words']
        assert request['checkpoint'] == str(settings.checkpoint)
        Path(command[5]).write_text(json.dumps({'scores': [
            {'id': request['pages'][0]['id'], 'score': 0.1},
            {'id': request['pages'][1]['id'], 'score': 0.9},
        ], 'retrieval_type': 'tokenwise'}), encoding='utf-8')
        return SimpleNamespace(returncode=0, stderr='', stdout='')

    monkeypatch.setattr(backend.subprocess, 'run', fake_process)
    result = backend.AuthorURARetriever(settings).rank(document, 'question', 1)
    assert result[0][1]['page'] == 2
    assert document['retrieval_metadata']['retrieval_type'] == 'tokenwise'

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('fixture', 1)
    monkeypatch.setattr(backend.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='timed out'):
        backend.AuthorURARetriever(settings).rank(document, 'question', 1)


def test_checkpoint_formats():
    assert checkpoint_state({'model': {'x': 1}}) == {'x': 1}
    assert checkpoint_state({'module': {'module.x': 1}}) == {'x': 1}
    with pytest.raises(ValueError):
        checkpoint_state({})


def test_original_model_load_is_strict(tmp_path, monkeypatch):
    # Check integration with author's class interface without pretending to load real weights.
    import types
    repo = tmp_path
    (repo / 'dataset').mkdir()
    (repo / 'dataset/const.py').write_text('VRM_SEMANTIC_TOKENS = ["<Text>"]\n', encoding='utf-8')
    for name in ('t5', 'roi.yaml', 'roi.pth', 'ura.pt'):
        (repo / name).write_text('fixture', encoding='utf-8')
    config = repo / 'config.json'
    config.write_text(json.dumps({'pretrained_dir': 't5', 'roi_config': 'roi.yaml',
                                  'roi_model': 'roi.pth', 'page_contrast_type': 'tokenwise'}))
    calls = []
    class Model:
        def __init__(self, args, pretrained_dir):
            self.tokenizer = SimpleNamespace(add_special_tokens=lambda value: calls.append('tokens'))
        def resize_token_embeddings(self): calls.append('resize')
        def load_state_dict(self, state, strict):
            assert strict is True and state == {'encoder': 1}
            calls.append('weights')
        def to(self, device): return self
        def eval(self): calls.append('eval'); return self
    module = types.ModuleType('models.mqa_model'); module.MQAT5Model = Model
    monkeypatch.setitem(sys.modules, 'models', types.ModuleType('models'))
    monkeypatch.setitem(sys.modules, 'models.mqa_model', module)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    torch = SimpleNamespace(load=lambda *a, **kw: {'model': {'encoder': 1}})
    load_model({'repository': str(repo), 'config': str(config),
                'checkpoint': str(repo / 'ura.pt'), 'device': 'cpu'}, torch)
    assert calls == ['tokens', 'resize', 'weights', 'eval']
