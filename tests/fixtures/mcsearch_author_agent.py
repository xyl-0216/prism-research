"""Author functions for behavioral parity tests; trailing whitespace normalized.
Source: YennNing/MC-Search, src/gpt/agent_local_retrieval_gpt_multi.py.
Commit: da9fe096592ec3555b3d3c9afe284c939c257458.
Copyright (c) 2026 MC-Search contributors. MIT License.
See docs/third_party/MC-Search-LICENSE.txt.
Transport/retrievers are injected by tests; no live network calls.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

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

def vqa_agent_local_gpt(
    client: OpenAI,
    gpt_model: str,
    retriever: LocalRetriever,
    image_retriever: Optional[LocalImageRetriever],
    question: str,
    input_image_urls: Optional[List[str]],   # ← changed to multi-image URL list
    max_turns: int = 8
):
    sys_text = SYS_PROMPT.format(QUESTION=question)
    messages = [{"role": "system", "content": sys_text}]

    # -- Opening logic: only attach sample images when the question mentions images --
    # Keywords: image/images, picture/pictures, photo/photos (case-insensitive)
    pat_any     = re.compile(r"\b(?:image|images|picture|pictures|photo|photos)\b", flags=re.I)
    pat_plural  = re.compile(r"\b(?:images|pictures|photos)\b", flags=re.I)

    question_mentions_image  = bool(pat_any.search(question))
    question_mentions_plural = bool(pat_plural.search(question))

    begin_parts: List[Dict[str, Any]] = [{"type": "text", "text": "Begin."}]

    if question_mentions_image and input_image_urls:
        # If plural and at least two images exist => 2 images, otherwise 1
        max_init_images = 2 if (question_mentions_plural and len(input_image_urls) >= 2) else 1

        for url in input_image_urls[:max_init_images]:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                # begin_parts.append({"type": "image_url", "image_url": {"url": url}})
                b64 = url_to_base64(url)
                if b64:
                    begin_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

    messages.append({"role": "user", "content": begin_parts})

    # First GPT call
    response = gpt_call(client, gpt_model, messages, max_tokens=512)
    messages.append({"role": "assistant", "content": response})

    turns = 0
    last_image_hit: Dict[str, Any] = {}

    while turns < max_turns:
        fa = extract_final_answer(response)
        if fa is not None:
            return fa, messages, last_image_hit

        intent = parse_search_block(response)
        itype = intent.get("type", "none")

        # === Dispatch among three retrieval types ===
        if itype == "text":
            q_txt = intent.get("query", "")
            facts, sids, score = retriever.top1(q_txt)
            if facts:
                messages.append({"role": "user", "content": "Contents of retrieved documents:\n" + facts[0]})
            else:
                messages.append({"role": "user", "content": "No relevant information found."})

        elif itype == "img_input":
            # Feed sample-provided images to the model; if none, provide a hint message
            if not input_image_urls:
                messages.append({"role": "user", "content": "No input images provided. You can conduct a search with text queries for images when needed."})
            else:
                parts: List[Dict[str, Any]] = [{"type": "text", "text": "Content of retrieved images (input images):"}]
                MAX_FEED_IMAGES = 6
                for url in input_image_urls[:MAX_FEED_IMAGES]:
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        b64 = url_to_base64(url)
                        if b64:
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                        if image_retriever:
                            hit = image_retriever.search_with_input_image(url)
                            print(f"[ImageRetriever] Search result for input image {url}: {hit}")
                            if hit:
                                last_image_hit = hit  # record the last hit
                                cap = hit.get("caption", "") or "No caption provided."
                                parts.append({"type": "text", "text": f"Caption: {cap}"})
                messages.append({"role": "user", "content": parts})


        elif itype == "img_text":
            # Text-to-image retrieval (average similarity over image and caption embeddings)
            if image_retriever is None:
                messages.append({"role": "user", "content": "Image retriever is not available."})
            else:
                q_txt = intent.get("query", "")
                hit = image_retriever.search_with_text_query(q_txt)
                if hit:
                    last_image_hit = hit
                    url = hit.get("image_path", "")
                    caption = hit.get("caption", "")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        parts = [{"type": "text", "text": "Contents of retrieved image (from text query):"}]
                        b64 = url_to_base64(url)
                        if b64:
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                            if caption:
                                parts.append({"type": "text", "text": f"Caption: {caption}"})
                        messages.append({"role": "user", "content": parts})
                    else:
                        # If no usable URL, return text description only (rare; URL usually exists)
                        desc = f"Retrieved Image Caption: {caption}"
                        messages.append({"role": "user", "content": "Contents of retrieved image:\n" + desc})
                else:
                    messages.append({"role": "user", "content": "No relevant image found."})

        else:
            messages.append({"role": "user", "content": "No retrieval executed."})

        # Next round
        response = gpt_call(client, gpt_model, messages, max_tokens=512)
        messages.append({"role": "assistant", "content": response})
        turns += 1

    return response, messages, last_image_hit
