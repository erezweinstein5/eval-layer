"""Inspectable judge payloads. Only evaluation data, never credentials or headers."""
from copy import deepcopy
from scripts.jev_judge import build_request as jev_request
from scripts.llm_judge import build_messages


def judge_requests(evidence, rubric, models):
    return {
        'jev': jev_request(deepcopy(evidence), rubric, models['jev']),
        'llm': {'model': models['llm'], 'messages': build_messages(deepcopy(evidence), rubric),
                'max_tokens': 4096, 'temperature': 0},
    }


def evaluation_config(rubric, models, artifacts=(), *, provenance='recorded'):
    templates = judge_requests({'notice': 'Per-case evidence replaces this placeholder.'}, rubric, models)
    return {
        'provenance': provenance,
        'artifacts': deepcopy(list(artifacts)),
        'backends': {
            'jev': {'model': models['jev'], 'parameters': {'max_attempts': 1, 'timeout_s': 30},
                    'request_template': templates['jev']},
            'llm': {'model': models['llm'], 'parameters': {'max_tokens': 4096, 'temperature': 0,
                    'json_mode': False, 'timeout_s': 120}, 'request_template': templates['llm']},
        },
        'notes': [
            'Both judges receive the same saved agent evidence. Judge calls do not rerun the agent.',
            'The generated judge guide documents intent; the request template shows the adapter prompt.',
            'Overall pass requires both the weighted rubric threshold and all execution checks.',
            'Jev maps ordered score criteria to an expected score; the LLM returns scores and explanations.',
            'Generation, agent execution, and judging are separate stages; judge timing excludes the first two.',
        ],
    }
