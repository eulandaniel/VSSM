import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from werkzeug.datastructures import FileStorage

VALID_LABELS = ['positivo', 'negativo', 'neutro', 'misto']
LABEL_MAP = {
    'pos': 'positivo', 'positiva': 'positivo', 'positivo': 'positivo', '+': 'positivo',
    'neg': 'negativo', 'negativa': 'negativo', 'negativo': 'negativo', '-': 'negativo',
    'neu': 'neutro', 'neutral': 'neutro', 'neutra': 'neutro', 'neutro': 'neutro',
    'mixed': 'misto', 'mista': 'misto', 'misto': 'misto',
    'positivo e negativo': 'misto', 'negativo e positivo': 'misto',
    'positivo/negativo': 'misto', 'negativo/positivo': 'misto',
    'positivo, negativo': 'misto', 'negativo, positivo': 'misto',
    'ambivalente': 'misto'
}


def save_upload(file: FileStorage, upload_folder: str) -> str:
    ext = Path(file.filename).suffix.lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        raise ValueError('Formato inválido. Use CSV ou XLSX.')
    filename = f'{uuid.uuid4().hex}{ext}'
    path = str(Path(upload_folder) / filename)
    file.save(path)
    return path


def load_dataframe(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin-1')
    return pd.read_excel(path)


def normalize_label(value):
    """Normaliza um julgamento humano para uma das quatro classes do VSSM.

    Combinações que expressem simultaneamente valência positiva e negativa são
    tratadas como 'misto', inclusive pequenas variações de gênero, número,
    separadores e ordem dos termos. Valores que não representam uma classe de
    sentimento permanecem inválidos (None) e NÃO são descartados silenciosamente
    durante a consolidação.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    if not text:
        return None

    direct = LABEL_MAP.get(text)
    if direct:
        return direct

    # Reconhece variações como "positivos e negativo", "negativa/positiva",
    # "positivo - negativo" etc. sem depender de enumerar cada grafia.
    has_positive = bool(re.search(r'\bpositiv(?:o|a|os|as)\b', text))
    has_negative = bool(re.search(r'\bnegativ(?:o|a|os|as)\b', text))
    if has_positive and has_negative:
        return 'misto'

    return text if text in VALID_LABELS else None


def profile_dataframe(df: pd.DataFrame) -> Dict:
    return {
        'rows': int(len(df)),
        'columns': list(df.columns),
        'empty_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }


def _valid_votes(values):
    """Retorna somente votos normalizados pertencentes às quatro classes válidas."""
    return [v for v in list(values) if v in VALID_LABELS]


def majority_vote(values):
    """Consolida quando pelo menos dois julgamentos válidos concordam."""
    vals = _valid_votes(values)
    if len(vals) < 2:
        return None
    top_label, top_count = Counter(vals).most_common(1)[0]
    return top_label if top_count >= 2 else None


def decision_status(values) -> str:
    """Classifica a decisão humana sem criar novas classes de sentimento.

    - unanimidade: todos os avaliadores selecionados deram votos válidos iguais;
    - maioria: pelo menos dois votos válidos concordam;
    - empate: há pelo menos dois votos válidos, mas nenhum rótulo recebe dois votos;
    - insuficiencia: há menos de dois votos válidos.
    """
    raw = list(values)
    vals = _valid_votes(raw)
    if len(vals) < 2:
        return 'insuficiencia'
    counts = Counter(vals)
    top_count = counts.most_common(1)[0][1]
    if top_count >= 2:
        if len(vals) == len(raw) and top_count == len(raw):
            return 'unanimidade'
        return 'maioria'
    return 'empate'


def tie_score(values):
    return int(decision_status(values) == 'empate')


def incomplete_score(values):
    """Compatibilidade: marca somente insuficiência real de decisão."""
    return int(decision_status(values) == 'insuficiencia')


def agreement_score(values):
    vals = _valid_votes(values)
    if not vals:
        return 0.0
    top = Counter(vals).most_common(1)[0][1]
    return top / len(vals)


def ambiguity_score(values):
    return round(1.0 - agreement_score(values), 4)


def _raw_vote_status(value) -> str:
    """Distingue vazio de rótulo não reconhecido antes da normalização."""
    if pd.isna(value) or not str(value).strip():
        return 'vazio'
    return 'valido' if normalize_label(value) in VALID_LABELS else 'invalido'


def _evaluator_quality(work: pd.DataFrame, label_cols: List[str]) -> Dict:
    """Resume qualidade original e distribuição após normalização por avaliador."""
    result = {}
    canonical = set(VALID_LABELS)
    for col in label_cols:
        raw = work[col]
        norm = raw.apply(normalize_label)
        statuses = raw.apply(_raw_vote_status)
        raw_clean = raw.fillna('').astype(str).str.strip().str.lower()
        canonical_count = int(((statuses == 'valido') & raw_clean.isin(canonical)).sum())
        recognized_count = int((statuses == 'valido').sum())
        raw_values = (
            raw.fillna('VAZIO').astype(str).str.strip().replace('', 'VAZIO')
            .value_counts().to_dict()
        )
        result[col] = {
            'total': int(len(work)),
            'recognized': recognized_count,
            'canonical_original': canonical_count,
            'normalized_variants': int(recognized_count - canonical_count),
            'empty': int((statuses == 'vazio').sum()),
            'invalid': int((statuses == 'invalido').sum()),
            'raw_values': {str(k): int(v) for k, v in raw_values.items()},
            'normalized_distribution': {
                label: int((norm == label).sum()) for label in VALID_LABELS
            },
        }
    return result

def prepare_labeled_data(
    df: pd.DataFrame,
    text_col: str,
    label_cols: List[str] | None = None,
    optional_cols: List[str] | None = None,
    reference_col: str | None = None
) -> Tuple[pd.DataFrame, Dict]:
    label_cols = label_cols or []
    optional_cols = optional_cols or []

    if text_col not in df.columns:
        raise ValueError('Coluna textual não encontrada.')

    original_rows = int(len(df))
    work = df.copy()
    work['_original_index'] = work.index

    before_text = len(work)
    work = work[work[text_col].notna()].copy()
    work[text_col] = work[text_col].astype(str).str.strip()
    work = work[work[text_col] != '']
    empty_text_rows = int(before_text - len(work))

    tie_rows = 0
    incomplete_rows = 0
    invalid_reference_labels = {}
    reference_mode = ''
    label_columns = []
    evaluator_quality = {}
    status_counts = {}
    audit_records = []

    if reference_col:
        if reference_col not in work.columns:
            raise ValueError('Coluna de referência humana não encontrada.')

        work['human_consensus'] = work[reference_col].apply(normalize_label)
        work['human_agreement'] = 1.0
        work['human_ambiguity'] = 0.0

        invalid = work[work['human_consensus'].isna()].copy()
        if len(invalid):
            invalid_reference_labels = (
                invalid[reference_col]
                .fillna('VAZIO')
                .astype(str)
                .str.strip()
                .replace('', 'VAZIO')
                .value_counts()
                .to_dict()
            )

        tie_rows = int(
            work[reference_col]
            .fillna('')
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['empate', 'sem consenso', 'sem consenso humano'])
            .sum()
        )

        reference_mode = 'rotulo_final'
        label_columns = [reference_col]

    else:
        missing = [c for c in label_cols if c not in work.columns]
        if missing:
            raise ValueError(f'Colunas de classificação humana ausentes: {missing}')
        if not label_cols:
            raise ValueError('Informe uma coluna de referência humana ou colunas para consenso por maioria.')

        for col in label_cols:
            work[col + '_norm'] = work[col].apply(normalize_label)

        norm_cols = [c + '_norm' for c in label_cols]
        work['human_status'] = work[norm_cols].apply(decision_status, axis=1)
        work['human_incomplete'] = (work['human_status'] == 'insuficiencia').astype(int)
        work['human_consensus'] = work[norm_cols].apply(majority_vote, axis=1)
        work['human_tie'] = (work['human_status'] == 'empate').astype(int)
        work['human_ambiguity'] = work[norm_cols].apply(ambiguity_score, axis=1)
        work['human_agreement'] = work[norm_cols].apply(agreement_score, axis=1)

        tie_rows = int(work['human_tie'].sum())
        incomplete_rows = int(work['human_incomplete'].sum())
        evaluator_quality = _evaluator_quality(work, label_cols)
        status_counts = {
            status: int((work['human_status'] == status).sum())
            for status in ['unanimidade', 'maioria', 'empate', 'insuficiencia']
        }
        audit_cols = ['_original_index', text_col]
        for col in label_cols:
            audit_cols.extend([col, col + '_norm'])
        audit_cols.extend(['human_consensus', 'human_status', 'human_agreement', 'human_ambiguity'])
        audit_records = work[audit_cols].copy().fillna('').to_dict(orient='records')
        reference_mode = 'votacao_maioria'
        label_columns = label_cols

        invalid_votes = {}
        for col in label_cols:
            normalized = work[col].apply(normalize_label)
            raw_nonempty = work[col].notna() & work[col].astype(str).str.strip().ne('')
            invalid_values = work[normalized.isna() & raw_nonempty][col]
            if len(invalid_values):
                invalid_votes[col] = (
                    invalid_values.fillna('VAZIO').astype(str).str.strip()
                    .replace('', 'VAZIO').value_counts().to_dict()
                )
        invalid_reference_labels = invalid_votes

    removed_reference_rows = int(work['human_consensus'].isna().sum())
    work = work[work['human_consensus'].notna()].copy()

    distribution = (
        work['human_consensus']
        .value_counts()
        .reindex(VALID_LABELS, fill_value=0)
        .to_dict()
    )

    quality = {
        'original_rows': original_rows,
        'valid_rows': int(len(work)),
        'removed_rows': int(original_rows - len(work)),
        'empty_text_rows': empty_text_rows,
        'removed_reference_rows': removed_reference_rows,
        'tie_rows': tie_rows,
        'incomplete_rows': incomplete_rows,
        'insufficient_rows': incomplete_rows,
        'status_counts': status_counts,
        'evaluator_quality': evaluator_quality,
        'audit_records': audit_records,
        'invalid_reference_labels': invalid_reference_labels,
        'reference_mode': reference_mode,
        'reference_column': reference_col,
        'label_columns': label_columns,
        'distribution': distribution,
        'avg_agreement': float(work['human_agreement'].mean()) if len(work) else 0,
        'avg_ambiguity': float(work['human_ambiguity'].mean()) if len(work) else 0,
        'optional_columns': [c for c in optional_cols if c in work.columns],
    }
    return work, quality
