"""LangChain model boundary with explicit, single-owner request accounting."""
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from open_deep_research.configuration import Configuration
from open_deep_research.model_compat import model_extra_body


def make_infer(run, config):
    settings = Configuration.from_runnable_config(config)
    model = ChatOpenAI(model=settings.research_model.split(':',1)[-1],
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL'),
        max_tokens=1800, timeout=90, max_retries=0,
        extra_body=model_extra_body(settings.research_model))
    async def infer(stage, prompt, images=()):
        run.reserve(stage, len(prompt), 1800)
        response = await model.ainvoke([HumanMessage(content=[{'type':'text','text':prompt},*images])],
            config={'callbacks':[], 'tags':['prism',stage]})
        if (response.response_metadata.get('finish_reason')!='stop'
                or not isinstance(response.content,str) or not response.content.strip()):
            raise ValueError('Incomplete model response: '+stage)
        run.log('evidence_model', stage=stage, usage=response.usage_metadata)
        return response.content
    return infer
