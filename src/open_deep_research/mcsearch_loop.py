"""Async adaptation of MC-Search's vqa_agent_local_gpt conversation loop.

Source: YennNing/MC-Search, src/gpt/agent_local_retrieval_gpt_multi.py,
commit da9fe096592ec3555b3d3c9afe284c939c257458 (MIT).
Copyright (c) 2026 MC-Search contributors.
See docs/third_party/MC-Search-LICENSE.txt and mcsearch-provenance.json.

Preserved: system + opening message, assistant action, retrieval observation,
next assistant action using ALL previous messages, final-answer termination.
Adapted: async injected LangChain call, injected tool dispatch, trace hooks,
and checking the final response after the last allowed retrieval. No hidden
reasoning is requested or persisted. See the upstream reference in parity tests.
"""
from copy import deepcopy


async def conversation_search(*, system_prompt, opening, call, parse, retrieve,
                              max_turns, record):
    """max_turns bounds actions; up to max_turns+1 planner calls are possible."""
    if not 1 <= max_turns <= 8:
        raise ValueError('MC-Search max_turns must be 1..8')
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': opening}]
    turns = 0

    async def next_response():
        response = await call(deepcopy(messages))
        messages.append({'role': 'assistant', 'content': response})
        record(messages, turns)
        return parse(response)

    decision = await next_response()
    while True:
        # Author loop checks Final Answer before executing the next retrieval.
        # Also inspect the response generated after the last permitted action.
        if decision.get('action') == 'stop':
            return {'status': 'stopped', 'answer': decision.get('reason'),
                    'turns': turns}
        if turns >= max_turns:
            return {'status': 'budget_exhausted', 'answer': '', 'turns': turns}
        observation = await retrieve(decision)
        if observation is None:
            return {'status': 'budget_exhausted', 'answer': '', 'turns': turns}
        messages.append({'role': 'user', 'content': observation})
        turns += 1
        record(messages, turns)
        decision = await next_response()


def trace_messages(messages):
    """Persist readable action/observation history without duplicating base64 images."""
    result = []
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            content = [block if block.get('type') == 'text' else
                       {'type': 'image_reference', 'description': '已登记的输入图片'}
                       for block in content]
        result.append({'role': message['role'], 'content': content})
    return deepcopy(result)
