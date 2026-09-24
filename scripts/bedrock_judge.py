"""Claude judge on Bedrock Runtime through the official Anthropic SDK.

Install anthropic[bedrock] only when using this optional provider. Credentials
stay in the SDK environment/credential chain, outside evaluation artifacts.
"""
import os
import re
import time

try:
    from .llm_judge import build_messages, _redact
    from .judge_results import parse_judge_response, validate_judge
except ImportError:
    from llm_judge import build_messages, _redact
    from judge_results import parse_judge_response, validate_judge


def build_request(state, rubric, model_id, *, max_tokens=8192, effort='high'):
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError('An explicit model ID is required')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if effort not in ('low', 'medium', 'high', 'xhigh', 'max'):
        raise ValueError('Unsupported effort level')
    messages = build_messages(state, rubric)
    return {'model': model_id, 'system': messages[0]['content'],
            'messages': messages[1:], 'max_tokens': max_tokens,
            'output_config': {'effort': effort}}


def normalize_usage(raw):
    """Include Anthropic cache reads/writes in total input, preserving unknowns."""
    def count(name):
        value = raw.get(name)
        return value if type(value) is int and value >= 0 else None
    fresh = count('input_tokens')
    read = count('cache_read_input_tokens')
    write = count('cache_creation_input_tokens')
    # Optional cache fields absent from older responses mean no cache accounting
    # was reported; don't invent a total if either cache value is malformed.
    for name in ('cache_read_input_tokens', 'cache_creation_input_tokens'):
        if name not in raw:
            raw[name] = 0
    read, write = count('cache_read_input_tokens'), count('cache_creation_input_tokens')
    usage = {'input_tokens': fresh + read + write if None not in (fresh, read, write) else None,
             'output_tokens': count('output_tokens'), 'cached_input_tokens': read,
             'cache_creation_input_tokens': write}
    return {k: v for k, v in usage.items() if v is not None} or None


def evaluate(state, rubric, *, model_id, region, max_tokens=8192,
             effort='high', timeout_s=180, client=None):
    start = time.perf_counter()
    key = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
    metadata = {'backend': 'llm', 'provider': 'bedrock', 'model_id': model_id,
                'resolved_model_id': None, 'usage': None, 'latency_ms': 0}
    owned_client = None
    try:
        if not isinstance(region, str) or not re.fullmatch(r'[a-z]{2}(?:-gov)?-[a-z]+-\d', region):
            raise ValueError('An explicit AWS region is required')
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 900:
            raise ValueError('timeout_s must be within 0..900 seconds')
        payload = build_request(state, rubric, model_id, max_tokens=max_tokens, effort=effort)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return {**metadata, 'error': 'configuration_error', 'reason': 'Invalid Bedrock judge configuration or rubric'}
    try:
        if client is None:
            from anthropic import AnthropicBedrock
            # Pin the endpoint: never forward the credential to an env URL override.
            owned_client = AnthropicBedrock(aws_region=region,
                base_url=f'https://bedrock-runtime.{region}.amazonaws.com',
                max_retries=0, timeout=timeout_s)
            client = owned_client
        with client.messages.stream(**payload) as stream:
            response = stream.get_final_message()
        raw_usage = response.usage.model_dump()
        metadata.update(resolved_model_id=response.model,
                        usage=normalize_usage(dict(raw_usage)), provider_usage=raw_usage,
                        stop_reason=response.stop_reason)
        if response.stop_reason != 'end_turn':
            result = {'error': 'incomplete_response', 'reason': 'Judge did not finish with end_turn'}
        else:
            text = ''.join(block.text for block in response.content if block.type == 'text')
            result = _redact(parse_judge_response(text), key)
            result.pop('model_id', None)
            result.pop('resolved_model_id', None)
            result = validate_judge({**result, **metadata}, rubric)
    except ImportError:
        result = {'error': 'configuration_error', 'reason': 'Install anthropic[bedrock] for this provider'}
    except Exception as exc:
        # Exception bodies may echo credentials or request contents. Keep only type/status.
        result = {'error': 'provider_error', 'reason': type(exc).__name__}
        status = getattr(exc, 'status_code', None)
        if type(status) is int:
            result['http_status'] = status
    finally:
        if owned_client is not None:
            owned_client.close()
    metadata['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
    return _redact({**result, **metadata}, key)
