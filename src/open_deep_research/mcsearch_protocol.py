"""Vendored MC-Search protocol parsers, unchanged function bodies.
Source: src/gpt/agent_local_retrieval_gpt_multi.py
Upstream project: https://mc-search-project.github.io/
Repository: https://github.com/YennNing/MC-Search
Commit: da9fe096592ec3555b3d3c9afe284c939c257458
Full source SHA256: 07c32cc10eed49b3ff9e5fa9efe6f349d85c3808f17848e9ce8626f3d0b89b7d
See docs/third_party/MC-Search-LICENSE.txt.
"""
# MIT License
#
# Copyright (c) 2026 MC-Search contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
from typing import Any, Dict, Optional

def parse_search_block(resp: str) -> Dict[str, Any]:
    """
    Parse <Search>, returns:
      {"type": "text", "query": "..."}  → Local Text Retrieval
      {"type": "img_text", "query": "..."}  → Local Image Retrieval with Text Query
      {"type": "img_input"}  → Local Image Retrieval with Input Image
      {"type": "none"}       → No Retrieval / unrecognized
    """
    # Prefer matching image retrieval with text query first
    m_img_txt = re.search(r"Local Image Retrieval with Text Query\s*:\s*(.+)", resp, flags=re.IGNORECASE)
    if m_img_txt:
        q = m_img_txt.group(1).strip().replace(">", "").strip().strip('"')
        return {"type": "img_text", "query": q}

    # Then match text retrieval
    m_txt = re.search(r"Local Text Retrieval\s*:\s*(.+)", resp, flags=re.IGNORECASE)
    if m_txt:
        q = m_txt.group(1).strip().replace(">", "").strip().strip('"')
        return {"type": "text", "query": q}

    # Match input-image retrieval
    if re.search(r"Local Image Retrieval with Input Image", resp, flags=re.IGNORECASE):
        return {"type": "img_input"}

    # No retrieval
    if re.search(r"No Retrieval", resp, flags=re.IGNORECASE):
        return {"type": "none"}

    return {"type": "none"}

def extract_final_answer(resp: str) -> Optional[str]:
    if "Final Answer" not in resp:
        return None
    return resp.split("Final Answer:")[-1].strip()
