"""Three bounded real-model chat turns, including process-style session reload.

No web search, personal files or long-term memory. Uses API quota.
"""
import argparse
import json
from pathlib import Path

from open_deep_research.cli import add_research_arguments
from open_deep_research.chat_cli import run_chat,Session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',required=True)
    parser.add_argument('--output',required=True)
    options=parser.parse_args()
    root=Path(options.output).resolve();root.mkdir(parents=True,exist_ok=False)
    common=argparse.ArgumentParser()
    add_research_arguments(common,interactive=True)
    args=common.parse_args(['--env-file',options.env_file,'--no-memory'])
    args.session_dir=str(root/'sessions');args.session_max_requests=6
    args.list=False;args.resume=None
    inputs=iter(['本次讨论只使用 API 模型，不部署本地大模型。调试预算是20元。请简短确认这两个限制。',
                 '刚才我说的预算和部署方式分别是什么？只回答这两项。','/exit'])
    run_chat(args,read=lambda _:next(inputs))
    saved=next((root/'sessions').glob('*/session.json'))
    args.resume=saved.parent.name
    inputs=iter(['继续刚才的话题。请复述我的调试预算和模型部署限制，只回答这两项。','/exit'])
    run_chat(args,read=lambda _:next(inputs))
    session=Session.load(root/'sessions',args.resume)
    turns=session.data['turns']
    checks={'three_completed_turns':len(turns)==3 and all(t['status']=='completed' for t in turns),
            'second_turn_remembers_budget':len(turns)>1 and '20' in turns[1].get('assistant',''),
            'resumed_turn_remembers_budget':len(turns)>2 and '20' in turns[2].get('assistant',''),
            'resumed_turn_mentions_api':len(turns)>2 and 'api' in turns[2].get('assistant','').lower(),
            'within_budget':session.used_requests()<=6}
    result={'success':all(checks.values()),'checks':checks,'requests':session.used_requests(),
            'session_id':args.resume,'session_path':str(saved)}
    (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result['success']


if __name__=='__main__':raise SystemExit(0 if main() else 1)
