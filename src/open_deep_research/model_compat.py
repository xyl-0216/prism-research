"""Small opt-in adaptation for Qwen through OpenAI-compatible endpoints."""

def model_extra_body(model_name):
    if model_name.lower().startswith('openai:qwen3.5-'):
        return {'enable_thinking': False}
    return None


def structured_output_kwargs(model_name):
    if model_name.lower().startswith('openai:qwen3.5-'):
        return {'method': 'function_calling'}
    return {}
