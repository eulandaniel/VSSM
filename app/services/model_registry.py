import os
import requests
import concurrent.futures


REQUEST_TIMEOUT = int(
    os.getenv(
        "FRAMEWORK_TIMEOUT",
        os.getenv("SVSM_TIMEOUT", "30"),
    )
)


# Fase 3: triagem de capacidade para classificação textual/sentimento.
# Não faz inferência, não testa quota/crédito e não usa a referência humana.
# A decisão usa a modalidade/finalidade identificável no catálogo/ID retornado pelo provedor.

# Famílias de geração/entendimento textual geral capazes de seguir instruções de classificação.



def provider_status():
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
    }


def discover_models():
    return [
        _openai_models(),
        _groq_models(),
        _anthropic_models(),
        _gemini_models(),
    ]


# ---------------------------------------------------------------------------
# Regra única de elegibilidade para TODOS os provedores.
# O provedor só interfere na descoberta/chamada da API; não no critério científico.
# A triagem não executa inferência, não usa quota/crédito e não usa a referência humana.
# ---------------------------------------------------------------------------

INCOMPATIBLE_TERMS = [
    ("embedding", "Embedding/vetorização não produz a classificação textual requerida."),
    ("embed", "Embedding/vetorização não produz a classificação textual requerida."),
    ("whisper", "Transcrição de áudio não corresponde à tarefa de classificação textual."),
    ("transcribe", "Transcrição de áudio não corresponde à tarefa de classificação textual."),
    ("transcription", "Transcrição de áudio não corresponde à tarefa de classificação textual."),
    ("audio", "Modelo especializado em áudio não é candidato ao benchmark textual."),
    ("speech", "Modelo especializado em voz/áudio não é candidato ao benchmark textual."),
    ("tts", "Text-to-speech não realiza a classificação textual requerida."),
    ("orpheus", "Modelo Text-to-Speech não realiza a classificação textual requerida."),
    ("image", "Modelo especializado em imagem não é candidato ao benchmark textual."),
    ("dall-e", "Geração de imagem está fora do escopo do benchmark textual."),
    ("moderation", "Modelo especializado em moderação não é o classificador textual avaliado pelo VSSM."),
    ("guard", "Modelo especializado em segurança/guardrail não é o classificador textual avaliado pelo VSSM."),
    ("safeguard", "Modelo especializado em segurança/guardrail não é o classificador textual avaliado pelo VSSM."),
    ("computer-use", "Modelo especializado em uso de computador está fora da tarefa."),
    ("computer_use", "Modelo especializado em uso de computador está fora da tarefa."),
    ("robotics", "Modelo especializado em robótica está fora da tarefa."),
    ("realtime", "Modelo especializado em interação em tempo real está fora do benchmark textual."),
    ("search", "Modelo/serviço especializado em busca não é o classificador textual avaliado pelo VSSM."),
]

# Evidências positivas de capacidade textual/instrucional.
# A lista é deliberadamente independente do provedor: qualquer modelo retornado por qualquer
# catálogo recebe exatamente os mesmos critérios.
TEXT_CAPABILITY_TERMS = (
    "gpt", "chatgpt", "claude", "gemini", "llama", "mixtral", "mistral",
    "qwen", "gemma", "deepseek", "command", "phi", "falcon", "yi-",
    "olmo", "nemotron", "granite", "ministral", "magistral", "jamba",
    "solar", "glm", "kimi", "minimax", "hermes", "nous", "dbrx",
    "grok", "o1", "o3", "o4", "compound"
)

EXPERIMENTAL_TERMS = ("preview", "beta", "experimental")


def _normalize_name(model_id: str) -> str:
    return (model_id or "").strip()


def _has_textual_capability_evidence(low: str) -> bool:
    """Mesma heurística de evidência textual para qualquer provedor."""
    return any(term in low for term in TEXT_CAPABILITY_TERMS)


def classify_model(model_id: str, provider: str = "") -> dict:
    """Classifica elegibilidade com regra única e agnóstica ao provedor."""
    name = _normalize_name(model_id)
    low = name.lower()

    # 1) Exclusões explícitas por modalidade/finalidade incompatível.
    for term, reason in INCOMPATIBLE_TERMS:
        if term in low:
            return {
                "id": name,
                "eligible": False,
                "status": "excluido",
                "category": "Não elegível",
                "reason": reason,
                "eligibility_basis": "modalidade/finalidade incompatível",
            }

    # 2) Evidência positiva de modelo textual/instrucional.
    if _has_textual_capability_evidence(low):
        experimental = any(term in low for term in EXPERIMENTAL_TERMS)
        return {
            "id": name,
            "eligible": True,
            "status": "experimental" if experimental else "elegivel",
            "category": "Elegível",
            "reason": (
                "Modelo identificado como textual/instrucional e compatível com classificação "
                "por instrução. O mesmo critério é aplicado a todos os provedores. "
                "Quota, crédito e disponibilidade momentânea são tratados somente na execução."
            ),
            "eligibility_basis": "capacidade textual/classificatória",
        }

    # 3) Ausência de evidência suficiente: não presumir elegibilidade nem incompatibilidade.
    return {
        "id": name,
        "eligible": False,
        "status": "revisar",
        "category": "Revisão",
        "reason": (
            "O catálogo retornou o modelo, mas o identificador/metadados disponíveis não fornecem "
            "evidência suficiente para confirmar automaticamente capacidade textual/classificatória. "
            "A mesma regra de revisão é aplicada independentemente do provedor."
        ),
        "eligibility_basis": "capacidade não confirmada",
    }


def _probe_model(provider: str, model_id: str) -> dict:
    """Probe neutro: verifica acesso e geração textual; não usa a base humana."""
    prompt = "Responda somente com a palavra OK."
    def result(resp, endpoint):
        if resp.ok:
            return {"available": True, "probe_status": "verified", "probe_endpoint": endpoint,
                    "probe_reason": f"Acesso e geração textual verificados em {endpoint}."}
        try: detail = str(resp.json())[:900].lower()
        except Exception: detail = (resp.text or "")[:900].lower()
        code=resp.status_code
        if code==429 or code>=500 or any(x in detail for x in ("rate limit","temporarily","timeout")):
            st="temporarily_unavailable"
        elif code in (401,403): st="access_denied"
        elif code==404 or any(x in detail for x in ("model_not_found","not found","does not exist")): st="not_available"
        else: st="incompatible"
        return {"available": False, "probe_status": st, "probe_endpoint": endpoint,
                "probe_reason": f"Verificação não concluída em {endpoint} (HTTP {code}; {st})."}
    try:
        if provider=="openai":
            key=os.getenv("OPENAI_API_KEY"); hdr={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
            x=requests.post("https://api.openai.com/v1/responses",headers=hdr,
                json={"model":model_id,"input":prompt,"max_output_tokens":16},timeout=REQUEST_TIMEOUT)
            rr=result(x,"Responses API")
            if rr["available"] or rr["probe_status"]!="incompatible": return rr
            y=requests.post("https://api.openai.com/v1/chat/completions",headers=hdr,
                json={"model":model_id,"messages":[{"role":"user","content":prompt}],"max_completion_tokens":16},timeout=REQUEST_TIMEOUT)
            return result(y,"Chat Completions API")
        if provider=="groq":
            x=requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {os.getenv('GROQ_API_KEY')}","Content-Type":"application/json"},
                json={"model":model_id,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":8},timeout=REQUEST_TIMEOUT)
            return result(x,"Groq Chat Completions")
        if provider=="anthropic":
            x=requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":os.getenv("ANTHROPIC_API_KEY"),"anthropic-version":"2023-06-01","content-type":"application/json"},
                json={"model":model_id,"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":prompt}]},timeout=REQUEST_TIMEOUT)
            return result(x,"Anthropic Messages API")
        if provider=="gemini":
            x=requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={os.getenv('GEMINI_API_KEY')}",
                headers={"Content-Type":"application/json"},json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0,"maxOutputTokens":8}},timeout=REQUEST_TIMEOUT)
            return result(x,"Gemini generateContent")
        return {"available":False,"probe_status":"unsupported_provider","probe_endpoint":"","probe_reason":"Provedor não suportado."}
    except (requests.Timeout, requests.ConnectionError):
        return {"available":False,"probe_status":"temporarily_unavailable","probe_endpoint":"","probe_reason":"Instabilidade de conexão durante o probe."}
    except Exception as exc:
        return {"available":False,"probe_status":"probe_error","probe_endpoint":"","probe_reason":f"Probe não concluído: {type(exc).__name__}."}

def _verify_operational_models(provider: str, models: list) -> list:
    """Probes independentes em paralelo, com concorrência controlada."""
    if not models: return []
    workers=max(1,min(int(os.getenv("VSSM_PROBE_WORKERS","6")),len(models)))
    def one(model):
        item=dict(model); p=_probe_model(provider,item["id"]); item.update(p)
        item["eligible"]=bool(item.get("eligible") and p.get("available"))
        if not p.get("available"):
            item["pre_probe_status"]=item.get("status"); item["status"]="indisponivel"
            item["category"]="Descoberto, mas não verificado para execução"; item["reason"]=p.get("probe_reason")
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        items=list(ex.map(one,models))
    return sorted(items,key=lambda x:x.get("id",""))

def _build_provider_result(provider: str, configured: bool, models: list, error: str | None = None) -> dict:
    """Descoberta + triagem sem inferência. Preserva o catálogo bruto retornado pela API."""
    raw_models = [str(model).strip() for model in (models or []) if str(model).strip()]
    unique_models = sorted(set(raw_models))
    enriched = [classify_model(model, provider) for model in unique_models]
    eligible = [model for model in enriched if model.get("eligible") is True]
    excluded = [model for model in enriched if model.get("eligible") is not True]
    stable = [model for model in eligible if model.get("status") == "elegivel"]
    experimental = [model for model in eligible if model.get("status") == "experimental"]
    review = [model for model in excluded if model.get("status") == "revisar"]
    blocked = [model for model in excluded if model.get("status") == "excluido"]
    return {
        "provider": provider,
        "configured": configured,
        # Rastreabilidade: resposta do catálogo antes da triagem.
        "raw_models": raw_models,
        "raw_count": len(raw_models),
        "unique_raw_models": unique_models,
        "duplicate_count": max(0, len(raw_models) - len(unique_models)),
        "models": [model["id"] for model in enriched],
        "eligible_models": eligible,
        "recommended_models": stable,
        "experimental_models": experimental,
        "excluded_models": excluded,
        "review_models": review,
        "blocked_models": blocked,
        "unavailable_models": [],
        "counts": {
            "total": len(enriched),
            "discovered": len(unique_models),
            "raw": len(raw_models),
            "eligible": len(eligible),
            "recommended": len(stable),
            "experimental": len(experimental),
            "review": len(review),
            "blocked": len(blocked),
            "unavailable": 0,
            "excluded": len(excluded),
        },
        "error": error,
    }


def verify_selected_models(selected: list) -> tuple[list, list]:
    """Probe mínimo apenas dos modelos escolhidos, antes da validação da base."""
    if not selected:
        return [], []
    workers=max(1,min(int(os.getenv("VSSM_PROBE_WORKERS","4")),len(selected)))
    def one(item):
        provider=str(item.get("provider","")); model=str(item.get("model",""))
        probe=_probe_model(provider, model)
        row={"provider":provider,"model":model,**probe}
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        checked=list(ex.map(one,selected))
    passed=[x for x in checked if x.get("available") is True]
    failed=[x for x in checked if x.get("available") is not True]
    return passed, failed

def _openai_models():
    key = os.getenv("OPENAI_API_KEY")

    if not key:
        return _build_provider_result("openai", False, [], "OPENAI_API_KEY ausente")

    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        models = [model.get("id") for model in data if model.get("id")]

        return _build_provider_result("openai", True, models)

    except Exception as e:
        return _build_provider_result("openai", True, [], str(e))


def _groq_models():
    key = os.getenv("GROQ_API_KEY")

    if not key:
        return _build_provider_result("groq", False, [], "GROQ_API_KEY ausente")

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        models = [model.get("id") for model in data if model.get("id")]

        return _build_provider_result("groq", True, models)

    except Exception as e:
        return _build_provider_result("groq", True, [], str(e))


def _anthropic_models():
    key = os.getenv("ANTHROPIC_API_KEY")

    if not key:
        return _build_provider_result("anthropic", False, [], "ANTHROPIC_API_KEY ausente")

    try:
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        models = [model.get("id") for model in data if model.get("id")]

        return _build_provider_result("anthropic", True, models)

    except Exception as e:
        return _build_provider_result("anthropic", True, [], str(e))


def _gemini_models():
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        return _build_provider_result("gemini", False, [], "GEMINI_API_KEY ausente")

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json().get("models", [])
        models = [
            model.get("name", "").replace("models/", "")
            for model in data
            if "generateContent" in model.get("supportedGenerationMethods", [])
        ]

        return _build_provider_result("gemini", True, models)

    except Exception as e:
        return _build_provider_result("gemini", True, [], str(e))
