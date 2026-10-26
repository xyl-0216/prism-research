"""Scoped evidence runs, durable request ceilings, and secret-safe event logs.

Only enabled by the local verification runner. Does not store prompts or images
in event logs. Request reservations survive failed/interrupted invocations.
"""
import json
import re
import threading
import time
from contextvars import ContextVar
from pathlib import Path

ACTIVE_RUN = ContextVar('evidence_run', default=None)
ROOT = Path(__file__).resolve().parents[2]


class BudgetExceeded(RuntimeError):
    pass


class EvidenceRun:
    def __init__(self, directory, ledger, secrets=()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.ledger = Path(ledger)
        self.secrets = [s for s in secrets if s]
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.pages = {}
        self.images = {}
        self.searches = 0
        self.truncated = []

    def safe(self, value):
        text = json.dumps(value, ensure_ascii=False, default=str)
        for secret in self.secrets:
            text = text.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)Bearer\s+[A-Za-z0-9._-]+', 'Bearer [REDACTED]', text)

    def log(self, event, **data):
        line = self.safe(dict(event=event, elapsed_s=round(time.monotonic()-self.started, 3), **data))
        with self.lock:
            with (self.directory/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(line+'\n')

    def reserve(self, kind, input_chars=0, max_output=0):
        """Single local runner only; lock covers concurrent tools in that runner."""
        with self.lock:
            if not self.ledger.exists():
                raise BudgetExceeded('No authorization ledger; refusing paid request')
            data = json.loads(self.ledger.read_text('utf-8-sig'))
            if data.get('closed', False):
                raise BudgetExceeded('This one-shot authorization is closed; unused quota is not reusable')
            legacy = self.ledger.name == 'authorized_budget_20261019.json'
            if not data.get('approved', legacy):
                raise BudgetExceeded('Additional API budget has not been approved')
            limits = data.get('limits', {'requests':40,'searches':8,'input_chars':600000,'output_tokens':128000})
            if data['requests'] >= limits['requests'] or (kind == 'search' and data['searches'] >= limits['searches']):
                raise BudgetExceeded('Authorized request budget exhausted')
            if data['input_chars']+input_chars > limits['input_chars'] or data['reserved_output_tokens']+max_output > limits['output_tokens']:
                raise BudgetExceeded('Conservative text token budget exhausted')
            data['requests'] += 1
            data['searches'] += kind == 'search'
            data['reserved_output_tokens'] += max_output
            data['input_chars'] += input_chars
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ledger.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')
            tmp.replace(self.ledger)
        self.log('request_reserved', kind=kind, totals=data)


def active_run():
    run = ACTIVE_RUN.get()
    if run is None:
        raise RuntimeError('Evidence tools require a scoped EvidenceRun: prism-research research')
    return run
