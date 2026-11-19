import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


HISTORY_FILENAME = 'winner_history.json'


def _history_path(results_folder: str) -> Path:
    path = Path(results_folder)
    path.mkdir(parents=True, exist_ok=True)
    return path / HISTORY_FILENAME


def load_winner_history(results_folder: str) -> List[Dict[str, Any]]:
    path = _history_path(results_folder)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def latest_winner(results_folder: str) -> Optional[Dict[str, Any]]:
    history = load_winner_history(results_folder)
    return history[0] if history else None


def save_winner_history(results_folder: str, history: List[Dict[str, Any]]) -> None:
    path = _history_path(results_folder)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')


def append_winner_history(results_folder: str, record: Dict[str, Any]) -> None:
    history = load_winner_history(results_folder)
    run_id = record.get('run_id')

    if run_id and any(item.get('run_id') == run_id for item in history):
        return

    record.setdefault('created_at', datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    history.insert(0, record)
    save_winner_history(results_folder, history)
