from typing import Dict
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
)

LABELS = ['positivo', 'negativo', 'neutro', 'misto']


def classification_metrics(y_true, y_pred) -> Dict:
    """Métricas com médias macro apenas nas classes suportadas pelo ground truth."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    domain_labels = list(LABELS)
    true_set = set(y_true)
    supported_labels = [label for label in domain_labels if label in true_set]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=domain_labels, zero_division=0
    )
    macro = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=supported_labels, average='macro', zero_division=0
        )
        if supported_labels else (0.0, 0.0, 0.0, 0)
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=domain_labels, average='weighted', zero_division=0
    )

    try:
        kappa = cohen_kappa_score(y_true, y_pred, labels=domain_labels)
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        'kappa': float(kappa),
        'macro_precision': float(macro[0]),
        'macro_recall': float(macro[1]),
        'macro_f1': float(macro[2]),
        'weighted_f1': float(weighted[2]),
        'per_class': {
            label: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            } for i, label in enumerate(domain_labels)
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=domain_labels).tolist(),
        'labels': domain_labels,
        'supported_labels': supported_labels,
        'unsupported_labels': [label for label in domain_labels if label not in supported_labels],
    }

def lexical_coherence(row) -> float:
    pred = row.get('model_label')
    justification = str(row.get('model_justification') or '').lower()
    text = str(row.get('text') or '').lower()

    if not pred or pred not in LABELS:
        return 0.0

    score = 0.4
    if pred in justification:
        score += 0.25

    markers = {
        'positivo': ['bom', 'boa', 'ótim', 'excelente', 'aprendi', 'gostei'],
        'negativo': ['ruim', 'confus', 'pouco', 'difícil', 'problema', 'fraco'],
        'neutro': ['neutro', 'não tenho', 'sem opinião', 'regular'],
        'misto': ['mas', 'porém', 'entretanto', 'apesar', 'positivo', 'negativo']
    }

    if any(m in text or m in justification for m in markers[pred]):
        score += 0.25
    if len(justification.split()) >= 8:
        score += 0.10

    return float(min(score, 1.0))


def explainability_score(row) -> float:
    justification = str(row.get('model_justification') or '')
    evidence = row.get('model_evidence') or ''

    score = 0.0
    if len(justification.split()) >= 8:
        score += 0.45
    if evidence and len(str(evidence).strip()) > 3:
        score += 0.35
    if row.get('model_label') in LABELS:
        score += 0.20

    return float(min(score, 1.0))



def sarh_score(metrics: Dict) -> float:
    """Score de Aderência à Referência Humana (SARH).

    Ranking principal pré-especificado: 40% Macro-F1, 30% Cohen's Kappa,
    20% Macro-Recall e 10% Accuracy. Indicadores operacionais e heurísticos
    são diagnósticos complementares e não alteram o SARH.
    """
    score = (
        0.40 * metrics.get('macro_f1', 0) +
        0.30 * metrics.get('kappa', 0) +
        0.20 * metrics.get('macro_recall', 0) +
        0.10 * metrics.get('accuracy', 0)
    )
    # Mantém precisão integral para ordenação; arredondamento é apenas de apresentação.
    return float(score)


def sarh_sensitivity(metrics: Dict) -> Dict:
    """Cenários pré-especificados para análise de sensibilidade do SARH."""
    mf1 = metrics.get('macro_f1', 0)
    kap = metrics.get('kappa', 0)
    mrec = metrics.get('macro_recall', 0)
    acc = metrics.get('accuracy', 0)
    scenarios = {
        'principal': (0.40, 0.30, 0.20, 0.10),
        'A': (0.50, 0.30, 0.10, 0.10),
        'B': (0.35, 0.35, 0.20, 0.10),
        'C': (0.40, 0.40, 0.10, 0.10),
    }
    return {
        name: float(w1*mf1 + w2*kap + w3*mrec + w4*acc)
        for name, (w1, w2, w3, w4) in scenarios.items()
    }


# O ranking final usa exclusivamente SARH. Scores legados foram removidos
# para evitar lógicas concorrentes de recomendação.

def classify_error(error_message: str) -> str:
    msg = str(error_message or '').lower()

    if not msg:
        return 'Resposta inválida ou rótulo ausente.'
    if '401' in msg or 'api key' in msg or 'authentication' in msg or 'unauthorized' in msg:
        return 'Chave de API ausente ou inválida.'
    if '403' in msg or 'permission' in msg or 'forbidden' in msg:
        return 'Sem permissão para usar o modelo.'
    if '404' in msg or 'not found' in msg:
        return 'Modelo indisponível ou não encontrado.'
    if '429' in msg or 'rate limit' in msg or 'quota' in msg:
        return 'Limite de requisições ou quota excedida.'
    if 'timeout' in msg or 'timed out' in msg:
        return 'Tempo de resposta excedido.'
    if 'json' in msg or 'decode' in msg:
        return 'Resposta fora do formato JSON esperado.'
    if 'label' in msg or 'classe' in msg or 'rótulo' in msg:
        return 'Resposta inválida ou fora das classes esperadas.'

    return 'Falha técnica durante a execução do modelo.'


def dominant_failure_reason(errors) -> str:
    reasons = [classify_error(e) for e in errors if str(e or '').strip()]
    if not reasons:
        return 'Resposta inválida ou fora do formato esperado.'
    return Counter(reasons).most_common(1)[0][0]


def diagnostic_summary(y_true, y_pred) -> Dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    total = len(y_true) or 1

    transitions = Counter()
    predicted_counter = Counter(y_pred)

    for h, p in zip(y_true, y_pred):
        if h != p:
            transitions[f'{h} → {p}'] += 1

    return {
        'positive_bias': round(predicted_counter.get('positivo', 0) / total * 100, 1),
        'negative_bias': round(predicted_counter.get('negativo', 0) / total * 100, 1),
        'neutral_bias': round(predicted_counter.get('neutro', 0) / total * 100, 1),
        'mixed_bias': round(predicted_counter.get('misto', 0) / total * 100, 1),
        'top_errors': [
            {'transition': k, 'count': int(v)}
            for k, v in transitions.most_common(8)
        ]
    }
