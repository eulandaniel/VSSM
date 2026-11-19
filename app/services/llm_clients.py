import json
import os
import time
import requests

REQUEST_TIMEOUT = int(
    os.getenv('FRAMEWORK_TIMEOUT',
    os.getenv('SVSM_TIMEOUT', '30'))
)

VALID_LABELS = ['positivo', 'negativo', 'neutro', 'misto']

SYSTEM_PROMPT = '''Você é um avaliador semântico especializado em análise de comentários qualitativos.
Classifique o sentimento do comentário em UMA destas classes: positivo, negativo, neutro, misto.
Responda exclusivamente em JSON válido, com as chaves:
label, confidence, justification, evidence.
- label: positivo|negativo|neutro|misto
- confidence: número de 0 a 1
- justification: explicação curta em português
- evidence: trecho do comentário que sustenta a decisão
Não invente informação que não esteja no texto.'''


def classify_comment(provider: str, model: str, text: str, context: str = '') -> dict:
    start = time.time()
    try:
        if provider == 'openai':
            result = _openai_chat(model, text, context)
        elif provider == 'groq':
            result = _groq_chat(model, text, context)
        elif provider == 'anthropic':
            result = _anthropic_chat(model, text, context)
        elif provider == 'gemini':
            result = _gemini_chat(model, text, context)
        else:
            raise ValueError('Provedor não suportado.')
        parsed = _parse_json_result(result)
        parsed['latency'] = round(time.time() - start, 4)
        parsed['raw_response'] = result
        parsed['error'] = None
        return parsed
    except Exception as e:
        return {
            'label': None,
            'confidence': 0,
            'justification': '',
            'evidence': '',
            'latency': round(time.time() - start, 4),
            'raw_response': '',
            'error': str(e)
        }


def _user_prompt(text: str, context: str = ''):
    ctx = f'Contexto/indicador: {context}\n' if context else ''
    return f'{ctx}Comentário: """{text}"""'


def _raise_safe_http_error(response):
    """Raise an operational error without exposing URLs/API keys."""
    if response.ok:
        return
    retry_after = response.headers.get("Retry-After")
    suffix = f" Retry-After={retry_after}s." if retry_after else ""
    if response.status_code == 429:
        raise RuntimeError(f"HTTP 429 rate limit.{suffix}")
    raise RuntimeError(f"HTTP {response.status_code} API request failed.")


def _openai_chat(model, text, context):
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY ausente')
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _user_prompt(text, context)}
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'}
    }
    r = requests.post('https://api.openai.com/v1/chat/completions', headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'
    }, json=payload, timeout=REQUEST_TIMEOUT)
    _raise_safe_http_error(r)
    return r.json()['choices'][0]['message']['content']


def _groq_chat(model, text, context):
    key = os.getenv('GROQ_API_KEY')
    if not key:
        raise RuntimeError('GROQ_API_KEY ausente')
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': _user_prompt(text, context)}
        ],
        'temperature': 0
    }
    r = requests.post('https://api.groq.com/openai/v1/chat/completions', headers={
        'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'
    }, json=payload, timeout=REQUEST_TIMEOUT)
    _raise_safe_http_error(r)
    return r.json()['choices'][0]['message']['content']


def _anthropic_chat(model, text, context):
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        raise RuntimeError('ANTHROPIC_API_KEY ausente')
    payload = {
        'model': model,
        'max_tokens': 500,
        'temperature': 0,
        'system': SYSTEM_PROMPT,
        'messages': [{'role': 'user', 'content': _user_prompt(text, context)}]
    }
    r = requests.post('https://api.anthropic.com/v1/messages', headers={
        'x-api-key': key, 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'
    }, json=payload, timeout=REQUEST_TIMEOUT)
    _raise_safe_http_error(r)
    content = r.json().get('content', [])
    return ''.join(block.get('text', '') for block in content if block.get('type') == 'text')


def _gemini_chat(model, text, context):
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY ausente')
    prompt = SYSTEM_PROMPT + '\n\n' + _user_prompt(text, context)
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0, 'response_mime_type': 'application/json'}
    }
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    _raise_safe_http_error(r)
    candidates = r.json().get('candidates', [])
    return candidates[0]['content']['parts'][0]['text'] if candidates else ''


def _parse_json_result(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        cleaned = cleaned.replace('json\n', '', 1).replace('json', '', 1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start >= 0 and end > start:
            data = json.loads(cleaned[start:end+1])
        else:
            raise
    label = str(data.get('label', '')).strip().lower()
    if label not in VALID_LABELS:
        label = None
    raw_confidence = data.get('confidence', None)
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else None
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            confidence = None
    except (TypeError, ValueError):
        confidence = None

    return {
        'label': label,
        'confidence': confidence,
        'justification': str(data.get('justification', '') or ''),
        'evidence': data.get('evidence', '')
    }
