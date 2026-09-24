"""Inspectable judge payloads. Only evaluation data, never credentials or headers."""
from copy import deepcopy
from scripts.jev_judge import build_request as jev_request
from scripts.llm_judge import build_messages


def judge_requests(evidence, rubric, models, *, llm_options=None):
    options = llm_options or {}
    llm_payload = {'model': models['llm'], 'messages': build_messages(deepcopy(evidence), rubric), 'max_tokens': 4096, 'temperature': 0}
    if options.get('provider') == 'bedrock':
        from scripts.bedrock_judge import build_request
        llm_payload = build_request(deepcopy(evidence), rubric, models['llm'], max_tokens=options.get('max_tokens',8192), effort=options.get('effort','high'))
    return {
        'jev': jev_request(deepcopy(evidence), rubric, models['jev']),
        'llm': llm_payload,
    }


def evaluation_config(rubric, models, artifacts=(), *, provenance='recorded', llm_options=None):
    templates = judge_requests({'notice': 'Per-case evidence replaces this placeholder.'}, rubric, models, llm_options=llm_options)
    options = llm_options or {}
    parameters = {'max_tokens': 4096, 'temperature': 0, 'json_mode': False, 'timeout_s': 120}
    if options.get('provider') == 'bedrock':
        parameters = {'provider':'bedrock', 'region':options['region'], 'endpoint':f"https://bedrock-runtime.{options['region']}.amazonaws.com", 'max_tokens':options.get('max_tokens',8192), 'effort':options.get('effort','high'), 'timeout_s':180, 'max_retries':0}
    return {
        'provenance': provenance,
        'artifacts': deepcopy(list(artifacts)),
        'backends': {
            'jev': {'model': models['jev'], 'parameters': {'max_attempts': 1, 'timeout_s': 30},
                    'request_template': templates['jev']},
            'llm': {'model': models['llm'], 'parameters': parameters, 'request_template': templates['llm']},
        },
        'notes': [
            'Both judges receive the same saved agent evidence. Judge calls do not rerun the agent.',
            'The generated judge guide documents intent; the request template shows the adapter prompt.',
            'Overall pass requires both the weighted rubric threshold and all execution checks.',
            'Jev maps ordered score criteria to an expected score; the LLM returns scores and explanations.',
            'Generation, agent execution, and judging are separate stages; judge timing excludes the first two.',
        ],
    }
