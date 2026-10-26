# Copyright(c) 2022 Liang Zhang
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See docs/third_party/MPMQA-LICENSE.txt for the full license.
"""Adapted from MPMQA MQAT5Model.similarity_score (models/mqa_model.py).

Changes: single query/page pair, NumPy instead of Torch, no masks or distributed
training. Preserves bidirectional token interaction: dot -> max -> mean.
Caller uses lexical one-hot features; this is not the learned URA encoder.
Source commit: 5226a9aa849fd3b8f35620e7b7d7d10c09d754a2.
"""
import numpy as np


def late_interaction_score(question_features, page_features):
    q,p=np.asarray(question_features),np.asarray(page_features)
    if (q.ndim!=2 or p.ndim!=2 or not len(q) or not len(p)
            or q.shape[1]!=p.shape[1] or not np.isfinite(q).all() or not np.isfinite(p).all()):
        raise ValueError('Expected finite nonempty token matrices with matching dimensions')
    interactions=q@p.T
    return float(interactions.max(axis=1).mean()),float(interactions.max(axis=0).mean())
