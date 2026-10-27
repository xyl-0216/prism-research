"""Official MPMQA inference worker (Python 3.7+; no Agent imports).

Adapted input construction and tokenwise inference from AIM3-RUC/MPMQA,
commit 5226a9aa849fd3b8f35620e7b7d7d10c09d754a2.
Copyright (c) 2022 Liang Zhang. Adaptations licensed under Apache-2.0;
see docs/third_party/MPMQA-LICENSE.txt. Actual encoder/ROI/scoring functions
are imported from the author's checkout rather than approximated here.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def load_constants(repository):
    spec = importlib.util.spec_from_file_location('mpmqa_constants', str(repository / 'dataset/const.py'))
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    return constants


def build_page_tokens(regions, tokenizer, constants, max_length):
    """Mirror MQAContrastDataset.get_page: region marker then OCR/word tokens.

    Each subword receives the word's box and region's semantic segment ID.
    No regex tokens or one-hot features are used.
    """
    ids, boxes, segments = [], [], []
    truncated = False
    for region in regions:
        semantic = region['structure']
        marker = constants.VRM_SEMANTIC_CLS2TOKEN[semantic]
        segment = constants.VRM_SEMANTIC_CLS2ID[semantic]
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(marker_ids) != 1:
            raise ValueError('URA semantic tokens must be registered before encoding')
        entries = [(marker_ids, region['bbox'])]
        entries.extend((tokenizer.encode(w['text'], add_special_tokens=False), w['bbox'])
                       for w in region['words'])
        for token_ids, box in entries:
            available = max_length - len(ids)
            if len(token_ids) > available:
                truncated = True
            token_ids = token_ids[:available]
            ids.extend(token_ids)
            boxes.extend([box] * len(token_ids))
            segments.extend([segment] * len(token_ids))
            if truncated:
                break
        if truncated:
            break
    if not ids:
        raise ValueError('Page has no encodable regions')
    return ids, boxes, segments, truncated


def checkpoint_state(payload):
    # Author torch checkpoint, consolidated DeepSpeed module, or raw state dict.
    for key in ('model', 'module', 'state_dict'):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Expected a full URA state dict, not a training checkpoint directory')
    if all(k.startswith('module.') for k in payload):
        payload = {k[len('module.'):]: v for k, v in payload.items()}
    return payload


def load_model(request, torch):
    repository = Path(request['repository'])
    sys.path.insert(0, str(repository))
    from models.mqa_model import MQAT5Model
    constants = load_constants(repository)
    values = dict(visual_answer=False, va_module_type='map', va_label_smoothing=0.0,
                  page_contrast=True, page_contrast_type='tokenwise', page_contrast_module_type=None,
                  page_contrast_t=0.01, page_contrast_bidirection=False, max_dec_len=20,
                  roi_bua=True, mask=False, max_page_len=1024)
    values.update(json.loads(Path(request['config']).read_text(encoding='utf-8-sig')))
    for name in ('pretrained_dir', 'roi_config', 'roi_model'):
        if not values.get(name):
            raise ValueError('Official URA config must specify ' + name)
        path = Path(values[name])
        path = path if path.is_absolute() else repository / path
        if not path.exists():
            raise ValueError('Missing official model asset: ' + str(path))
        values[name] = str(path)
    if not values['page_contrast'] or values['page_contrast_type'] not in ('tokenwise', 'global'):
        raise ValueError('Checkpoint configuration must enable URA page contrast')
    args = SimpleNamespace(**values)
    model = MQAT5Model(args, pretrained_dir=args.pretrained_dir)
    special = list(constants.VRM_SEMANTIC_TOKENS)
    if args.mask:
        special.append('<mask>')
    model.tokenizer.add_special_tokens({'additional_special_tokens': special})
    model.resize_token_embeddings()
    # Only load trusted local author weights; no download or partial/random fallback.
    payload = torch.load(request['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint_state(payload), strict=True)
    model.to(request['device']).eval()
    return model, args, constants


def encode_features(model, features, args, torch):
    if args.page_contrast_type == 'global':
        features = features[:, 0]
    if args.page_contrast_module_type is not None:
        features = model.page_contrast_module(features)
    return torch.nn.functional.normalize(features, dim=-1)


def retrieve(request):
    import cv2
    import torch
    model, args, constants = load_model(request, torch)
    device = request['device']
    page_limit = int(args.max_page_len)
    if not 1 <= page_limit <= 4096:
        raise ValueError('max_page_len must be between 1 and 4096')
    tokenizer = model.tokenizer
    # Questions retain the author's tokenizer and EOS handling (no deduplication).
    question = tokenizer(request['question'], return_tensors='pt')
    qids = question['input_ids'].to(device)
    qmask = question['attention_mask'].to(device)
    qsegments = torch.full_like(qids, constants.VRM_SEMANTIC_CLS2ID['Question'])
    scores = []
    with torch.no_grad():
        qhidden = model.encoding_question(qids, qmask, qsegments, return_hidden=True)
        qfeatures = encode_features(model, qhidden, args, torch)
        for page in request['pages']:
            image = cv2.imread(page['image'])  # BGR, as in the author's dataset.
            if image is None:
                raise ValueError('Unable to read rendered page: ' + page['image'])
            ids, boxes, segments, truncated = build_page_tokens(
                page['regions'], tokenizer, constants, page_limit)
            cids = torch.tensor([ids], dtype=torch.long, device=device)
            cmask = torch.ones_like(cids)
            csegments = torch.tensor([segments], dtype=torch.long, device=device)
            bboxes = [torch.tensor(boxes, dtype=torch.float32, device=device)]
            imgs = [torch.from_numpy(image).to(device)]
            # Calls original get_context_embedding: token + segment + ROI + location,
            # followed by the original T5 encoder. Preserve original bbox normalization.
            chidden = model.encoding_context(cids, cmask, imgs, bboxes, csegments, return_hidden=True)
            cfeatures = encode_features(model, chidden, args, torch)
            if args.page_contrast_type == 'tokenwise':
                # Same masks + bidirectional MaxSim as author; no DDP AllGather in inference.
                qc, cq = model.similarity_score(qfeatures, cfeatures, qmask, cmask)
                forward, reverse = qc[0, 0].item(), cq[0, 0].item()
            else:
                forward = reverse = (qfeatures @ cfeatures.T)[0, 0].item()
            scores.append({'id': page['id'], 'score': forward, 'reverse_score': reverse,
                           'tokens': len(ids), 'truncated': truncated})
    return {'scores': scores, 'retrieval_type': args.page_contrast_type,
            'layout_source': 'PyMuPDF regions and words; coarse semantic labels'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    result = retrieve(request)
    Path(args.response).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
