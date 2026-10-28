"""Compare the migrated loop with actual extracted author code, without APIs."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_deep_research.mcsearch_adapter import evidence_chain, parse_plan
from open_deep_research.mcsearch_loop import conversation_search


def author_module():
    path = Path(__file__).parent / 'fixtures/mcsearch_author_agent.py'
    spec = importlib.util.spec_from_file_location('mcsearch_author_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SYS_PROMPT = 'Question: {QUESTION}'
    return module


@pytest.mark.parametrize('action', [
    '<Sub-Question>\nFind fact\n<Search>\nLocal Text Retrieval: query',
    '<Sub-Question>\nConsider results\n<Search>\nNo Retrieval',
])
def test_same_conversation_as_author(action):
    upstream = author_module()
    responses = [action, '<End>\nFinal Answer: supported answer']
    upstream_calls = []
    replies = iter(responses)
    def gpt_call(client, model, messages, max_tokens):
        upstream_calls.append(deepcopy(messages))
        return next(replies)
    upstream.gpt_call = gpt_call
    answer, conversation, _ = upstream.vqa_agent_local_gpt(
        None, 'fixture', SimpleNamespace(top1=lambda q: (['raw fact'], ['fact-1'], 0.9)),
        None, 'question', [], max_turns=3)

    adapted_calls, snapshots = [], []
    replies2 = iter(responses)
    async def call(messages):
        adapted_calls.append(deepcopy(messages))
        return next(replies2)
    async def retrieve(decision):
        if decision['action'] == 'no_retrieval':
            return 'No retrieval executed.'
        assert decision['retrieval_query'] == 'query'
        return 'Contents of retrieved documents:\nraw fact'
    result = asyncio.run(conversation_search(system_prompt='Question: question',
        opening=[{'type':'text','text':'Begin.'}], call=call, parse=parse_plan,
        retrieve=retrieve, max_turns=3, record=lambda m,t: snapshots.append(deepcopy(m))))
    assert result['answer'] == answer
    assert adapted_calls == upstream_calls
    assert snapshots[-1] == conversation


def test_same_input_image_conversation_as_author():
    upstream = author_module()
    image = {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,fixture'}}
    upstream.url_to_base64 = lambda url: 'fixture'
    action = '<Sub-Question>\nRead image\n<Search>\nLocal Image Retrieval with Input Image'
    responses = [action, '<End>\nFinal Answer: image answer']
    original_calls = []
    replies = iter(responses)
    def call_original(client, model, messages, max_tokens):
        original_calls.append(deepcopy(messages)); return next(replies)
    upstream.gpt_call = call_original
    upstream.vqa_agent_local_gpt(None, 'fixture', None, None, 'Read the image',
                                  ['https://example.test/image'], max_turns=3)
    adapted_calls = []
    replies2 = iter(responses)
    async def call(messages):
        adapted_calls.append(deepcopy(messages)); return next(replies2)
    async def retrieve(decision):
        assert decision['modality'] == 'image'
        return [{'type':'text','text':'Content of retrieved images (input images):'}, image]
    asyncio.run(conversation_search(system_prompt='Question: Read the image',
        opening=[{'type':'text','text':'Begin.'}, image], call=call, parse=parse_plan,
        retrieve=retrieve, max_turns=3, record=lambda *a:None))
    assert adapted_calls == original_calls


def run_chain(responses, *, max_steps=3, image_available=False, input_images=(), execute=None):
    prompts, snapshots = [], []
    replies = iter(responses)
    async def infer(stage, prompt):
        prompts.append((stage, deepcopy(prompt))); return next(replies)
    async def default_execute(*args):
        return [{'id':'source-1', 'text':'RAW SOURCE: only for model X'}]
    async def run():
        return await evidence_chain('Analyze this image', infer=infer,
            execute=execute or default_execute,
            save=lambda t:snapshots.append(deepcopy(t)), image_available=image_available,
            max_steps=max_steps, input_images=input_images)
    return run, prompts, snapshots


TEXT_ACTION = '<Sub-Question>\nFind conditions\n<Search>\nLocal Text Retrieval: conditions'
ASSESSMENT = json.dumps({'status':'answered', 'intermediate_answer':'model X only',
                        'evidence_ids':['source-1'], 'missing_evidence':''})


def test_raw_observations_and_assistant_actions_reach_next_round():
    run, prompts, _ = run_chain([TEXT_ACTION, ASSESSMENT, '<End>\nFinal Answer: model X only'])
    trace = asyncio.run(run())
    history = prompts[2][1]
    assert [m['role'] for m in history] == ['system','user','assistant','user']
    assert history[2]['content'] == TEXT_ACTION
    assert 'RAW SOURCE: only for model X' in history[3]['content']
    assert trace['final_answer'] == 'model X only' and trace['status'] == 'completed'


def test_last_turn_answer_is_parsed_not_discarded():
    run, _, _ = run_chain([TEXT_ACTION, ASSESSMENT, '<End>\nFinal Answer: model X only'], max_steps=1)
    assert asyncio.run(run())['status'] == 'completed'


def test_no_retrieval_consumes_turns_without_executing_tools():
    action = '<Sub-Question>\nCheck current facts\n<Search>\nNo Retrieval'
    async def unexpected(*args): raise AssertionError('No tool should run')
    run, prompts, _ = run_chain([action, action, TEXT_ACTION], max_steps=2, execute=unexpected)
    trace = asyncio.run(run())
    assert trace['status'] == 'budget_exhausted' and trace['turns'] == 2
    assert not trace['steps'] and len(prompts) == 3


def test_unavailable_image_knowledge_base_fails_explicitly():
    run, _, _ = run_chain(['<Sub-Question>\nFind image\n<Search>\nLocal Image Retrieval with Text Query: device'])
    with pytest.raises(ValueError, match='modality'):
        asyncio.run(run())


def test_input_images_reach_model_but_not_trace_base64():
    image = {'type':'image_url','image_url':{'url':'data:image/png;base64,PRIVATE_IMAGE_BYTES'}}
    action = '<Sub-Question>\nRead image\n<Search>\nLocal Image Retrieval with Input Image'
    run, prompts, snapshots = run_chain([action, ASSESSMENT, '<End>\nFinal Answer: model X only'],
                                        image_available=True, input_images=[image])
    asyncio.run(run())
    assert image in prompts[0][1][1]['content']
    assert image in prompts[2][1][-1]['content']
    assert 'PRIVATE_IMAGE_BYTES' not in json.dumps(snapshots)


def test_no_legacy_dependency_tag_required():
    decision = parse_plan(TEXT_ACTION)
    assert decision['depends_on'] == [] and decision['retrieval_query'] == 'conditions'
