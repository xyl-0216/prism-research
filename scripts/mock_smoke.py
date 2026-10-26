"""Offline integration test: official graph and Tavily processing, fake I/O only."""
import asyncio
import json
import socket
import time
import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from open_deep_research import deep_researcher as dr, utils
from open_deep_research import bailian_search as bs
from open_deep_research.state import ResearchQuestion, Summary

ROOT = Path(__file__).resolve().parents[1]
EVENTS = []
START = time.perf_counter()
QUESTION = '比较 LangGraph、OpenAI Agents SDK 和 Microsoft Agent Framework 在 Agent 编排、工具调用、状态管理方面的区别，引用官方技术文档生成一份研究报告。'
URLS = ['https://docs.langchain.com/oss/python/langgraph/overview', 'https://openai.github.io/openai-agents-python/', 'https://learn.microsoft.com/en-us/agent-framework/overview/']
REPORT = '''# MOCK 流程报告（不是实际研究结果）

此文件由离线模型替身返回，经官方完整图写入 final_report。未调用真实模型或搜索，下面链接仅为测试夹具，没有在本测试中检索或核实。

## 编排、工具调用和状态管理

本测试不提供三框架的事实比较，只验证研究任务、搜索结果处理、压缩与报告输出的数据通路。

## 来源夹具

''' + '\n'.join(f'- [{i+1}] {url}' for i, url in enumerate(URLS))

def log(event, **data):
    EVENTS.append({'elapsed_s': round(time.perf_counter()-START, 4), 'event': event, 'mode': 'MOCK', **data})

class FakeModel:
    def __init__(self, schema=None, tools=None, config=None):
        self.schema, self.tools, self.config = schema, tools, config or {}
    def with_structured_output(self, schema, **kwargs):
        return FakeModel(schema, self.tools, self.config)
    def bind_tools(self, tools):
        for item in tools:
            convert_to_openai_tool(item)
        return FakeModel(self.schema, tools, self.config)
    def with_retry(self, **kwargs):
        return self
    def with_config(self, config):
        return FakeModel(self.schema, self.tools, config)
    async def ainvoke(self, messages, **kwargs):
        log('model', schema=getattr(self.schema, '__name__', None), messages=len(messages))
        if self.schema is ResearchQuestion:
            return ResearchQuestion(research_brief=QUESTION)
        if self.schema is Summary:
            return Summary(summary='MOCK webpage summary', key_excerpts='MOCK excerpt')
        if self.tools:
            names = [convert_to_openai_tool(t)['function']['name'] for t in self.tools]
            if 'ConductResearch' in names:
                name = 'ResearchComplete' if any(isinstance(m, ToolMessage) for m in messages) else 'ConductResearch'
                args = {'research_topic': QUESTION} if name == 'ConductResearch' else {}
            else:
                name = 'ResearchComplete' if any(isinstance(m, ToolMessage) for m in messages) else ('bailian_web_search' if 'bailian_web_search' in names else 'tavily_search')
                args = {'queries': ['MOCK official agent framework comparison']} if name == 'tavily_search' else ({'query':'MOCK official agent framework comparison'} if name == 'bailian_web_search' else {})
            log('tool_requested', name=name, args=args)
            return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': f'call-{len(EVENTS)}'}])
        if len(messages) == 1:
            assert 'MOCK compressed findings' in str(messages[0].content)
            return AIMessage(content=REPORT)
        assert any(isinstance(m, ToolMessage) and 'MOCK webpage summary' in str(m.content) for m in messages)
        return AIMessage(content='MOCK compressed findings\n' + '\n'.join(URLS))

class FakeTavily:
    def __init__(self, **kwargs):
        pass
    async def search(self, query, **kwargs):
        result = {'query': query, 'results': [{'url': u, 'title': 'MOCK official page', 'content': 'MOCK snippet', 'raw_content': 'MOCK page text'} for u in URLS]}
        log('search_result', name='tavily_search', result=result)
        return result

async def main():
    config = {'recursion_limit': 30, 'configurable': {'allow_clarification': False, 'max_concurrent_research_units': 1, 'max_researcher_iterations': 2, 'max_react_tool_calls': 2, 'max_structured_output_retries': 1}}
    if '--bailian' in sys.argv:
        config['configurable']['search_api'] = 'bailian'
    async def fake_search(query, config):
        result={'query':query,'summary':'MOCK webpage summary','sources':[{'url':u,'title':'MOCK'} for u in URLS]}
        log('search_result', name='bailian_web_search', result=result)
        return result
    with patch.object(dr, 'configurable_model', FakeModel()), patch.object(utils, 'init_chat_model', lambda **kw: FakeModel()), patch.object(utils, 'AsyncTavilyClient', FakeTavily), patch.object(bs, 'bailian_search_async', fake_search), patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled in MOCK')):
        result = await dr.deep_researcher.ainvoke({'messages': [HumanMessage(content=QUESTION)]}, config)
    assert result['final_report'] == REPORT
    assert result['raw_notes'] and 'MOCK webpage summary' in str(result['raw_notes'])
    log('passed', checks=['official full graph', 'tool schema conversion', 'Tavily formatting', 'structured summary fixture', 'compression context', 'final report'])
    (ROOT/'docs/mock_research_report.md').write_text(REPORT, encoding='utf-8')
    print('PASS: offline full-graph integration; no real API calls')

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        log('exception', type=type(exc).__name__, message=str(exc))
        raise
    finally:
        name = 'mock_bailian_events.jsonl' if '--bailian' in sys.argv else 'mock_events.jsonl'
        (ROOT/'docs/logs'/name).write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in EVENTS), encoding='utf-8')
