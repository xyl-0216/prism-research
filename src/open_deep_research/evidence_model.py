"""LangChain model boundary with explicit, single-owner request accounting."""
import os
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from open_deep_research.configuration import Configuration
from open_deep_research.model_compat import model_extra_body, structured_output_kwargs


def make_infer(run, config, *, schema=None):
    settings = Configuration.from_runnable_config(config)
    model = ChatOpenAI(model=settings.research_model.split(':',1)[-1],
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL'),
        max_tokens=1800, timeout=90, max_retries=0,
        extra_body=model_extra_body(settings.research_model))
    caller = model.with_structured_output(schema, include_raw=True,
        **structured_output_kwargs(settings.research_model)) if schema is not None else model
    async def infer(stage, prompt, images=()):
        run.reserve(stage, len(prompt), 1800)
        result = await caller.ainvoke([HumanMessage(content=[{'type':'text','text':prompt},*images])],
            config={'callbacks':[], 'tags':['prism',stage]})
        response = result['raw'] if schema is not None else result
        run.log('evidence_model', stage=stage, model=settings.research_model, usage=response.usage_metadata)
        if schema is not None:
            # Some OpenAI-compatible models put valid schema JSON in content
            # despite a named tool choice. Accept that envelope only after the
            # same Pydantic validation; never accept prose or truncated output.
            if (result.get('parsed') is None and not response.tool_calls
                    and not response.invalid_tool_calls
                    and response.response_metadata.get('finish_reason') == 'stop'
                    and isinstance(response.content, str) and response.content.strip()):
                text = response.content.strip()
                match = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
                try:
                    result['parsed'] = schema.model_validate_json(match.group(1) if match else text)
                    result['parsing_error'] = None
                    run.log('model_json_envelope', stage=stage)
                except ValidationError as error:
                    run.log('model_json_validation_error', stage=stage,
                            errors=[{'type': e['type'], 'loc': e['loc']}
                                    for e in error.errors()])
            if (response.response_metadata.get('finish_reason') not in ('stop','tool_calls')
                    or result.get('parsing_error') is not None or result.get('parsed') is None):
                run.log('model_schema_error', stage=stage,
                        finish_reason=response.response_metadata.get('finish_reason'),
                        error_type=type(result.get('parsing_error')).__name__,
                        tool_names=[call.get('name') for call in response.tool_calls])
                raise ValueError('Invalid structured model response: '+stage)
            return result['parsed'].model_dump_json()
        if (response.response_metadata.get('finish_reason')!='stop'
                or not isinstance(response.content,str) or not response.content.strip()):
            raise ValueError('Incomplete model response: '+stage)
        return response.content
    return infer
