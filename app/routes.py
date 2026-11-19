import json
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .services.data_pipeline import (
    load_dataframe,
    prepare_labeled_data,
    profile_dataframe,
    save_upload,
)
from .services.history import append_winner_history, load_winner_history
from .services.llm_clients import classify_comment
from .services.metrics import (
    classification_metrics,
    explainability_score,
    lexical_coherence,
    sarh_score,
    sarh_sensitivity,
)
from .services.model_registry import discover_models, provider_status


bp = Blueprint("main", __name__)

FRAMEWORK_MAX_WORKERS = int(os.getenv("FRAMEWORK_MAX_WORKERS", "4"))
FRAMEWORK_MODEL_FAILURE_LIMIT = int(os.getenv("FRAMEWORK_MODEL_FAILURE_LIMIT", "5"))

RUN_STATUS = {}
RUN_STATUS_LOCK = threading.Lock()


def _session_paths():
    return {
        "uploaded_path": session.get("uploaded_path"),
        "prepared_path": session.get("prepared_path"),
        "results_path": session.get("results_path"),
        "summary_path": session.get("summary_path"),
    }


def _failure_reason(error):
    msg = str(error or "").lower()

    if not msg:
        return ""

    if "401" in msg or "api_key" in msg or ("key" in msg and "ausente" in msg):
        return "Chave de API ausente ou inválida."

    if "403" in msg or "permission" in msg or "permiss" in msg:
        return "Sem permissão para usar o modelo."

    if "404" in msg or "not found" in msg:
        return "Modelo indisponível ou não encontrado."

    if "429" in msg or "rate" in msg or "quota" in msg or "limit" in msg:
        return "Limite de requisições ou quota excedida."

    if "timeout" in msg or "timed out" in msg:
        return "Tempo de resposta excedido."

    if "json" in msg:
        return "Resposta inválida ou fora do formato JSON esperado."

    if "label" in msg or "rótulo" in msg or "rotulo" in msg:
        return "Rótulo ausente ou fora das classes aceitas."

    if "interrompido" in msg:
        return "Modelo interrompido por falhas consecutivas."

    return "Falha técnica durante a execução do modelo."


def _failure_class(error):
    """Classifica a natureza operacional da falha para decidir retry/interrupção."""
    msg = str(error or "").lower()
    if not msg:
        return "none"

    # Permanente/configuração: repetir não tende a resolver nesta execução.
    if any(x in msg for x in ["401", "api_key", "unauthorized", "403", "permission", "forbidden", "404", "not found"]):
        return "permanent"

    # Quota/crédito explicitamente esgotado. 429 isolado não é tratado como quota permanente.
    quota_terms = ["insufficient_quota", "quota exhausted", "quota exceeded", "billing", "credit", "credits exhausted"]
    if any(x in msg for x in quota_terms):
        return "quota_exhausted"

    # Limite operacional de taxa: não é instabilidade nem falha de desempenho do modelo.
    if any(x in msg for x in ["429", "rate limit", "too many requests", "resource_exhausted"]):
        return "rate_limited"

    # Transitório técnico: uma nova tentativa controlada.
    if any(x in msg for x in ["timeout", "timed out", "connection", "connectionerror",
                               "500", "502", "503", "504", "server error", "temporarily unavailable"]):
        return "transient"

    # Saída semanticamente/estruturalmente inválida não recebe retry automático.
    if any(x in msg for x in ["json", "rótulo", "rotulo", "label", "resposta inválida"]):
        return "invalid_response"

    return "unknown"


def _should_retry(error):
    return _failure_class(error) in {"rate_limited", "transient", "unknown"}


def _dominant_failure_reason(errors):
    reasons = [_failure_reason(error) for error in errors if error]
    reasons = [reason for reason in reasons if reason]

    if not reasons:
        return ""

    return pd.Series(reasons).value_counts().idxmax()


def _model_key(provider, model):
    return f"{provider}::{model}"


def _update_run_model(run_id, provider, model, **updates):
    if not run_id:
        return

    key = _model_key(provider, model)

    with RUN_STATUS_LOCK:
        run = RUN_STATUS.get(run_id)

        if not run:
            return

        model_state = run.get("models", {}).get(key)

        if not model_state:
            return

        model_state.update(updates)


def _update_run_status(run_id, **updates):
    if not run_id:
        return

    with RUN_STATUS_LOCK:
        run = RUN_STATUS.get(run_id)

        if not run:
            return

        run.update(updates)


def _get_run_status(run_id):
    with RUN_STATUS_LOCK:
        return RUN_STATUS.get(run_id, {"status": "not_found"})


def _prepare_data_with_reference(df, text_col, reference_col):
    """
    Usa o rótulo humano final como referência principal.
    Mantém compatibilidade caso o data_pipeline ainda esteja na versão antiga.
    """
    try:
        return prepare_labeled_data(
            df,
            text_col=text_col,
            label_cols=[],
            optional_cols=[],
            reference_col=reference_col,
        )
    except TypeError:
        return prepare_labeled_data(df, text_col, [reference_col], [])


def _execute_single_model(
    chosen,
    df_records,
    total_rows,
    context_col=None,
    failure_limit=5,
    run_id=None,
):
    provider = chosen["provider"]
    model = chosen["model"]

    _update_run_model(run_id, provider, model, status="running", message="Em execução")

    model_rows = []
    consecutive_definitive_failures = 0
    stopped = False
    stop_reason = ""
    stop_class = ""
    observed_failures = 0
    not_executed_after_stop = 0

    for position, row in enumerate(df_records):
        idx = row.get("_row_id", position)
        record_id = row.get("record_id") or f"VSSM-{int(idx)+1:04d}"

        if stopped:
            not_executed_after_stop += 1
            item = {
                "row_id": int(idx),
                "record_id": record_id,
                "provider": provider,
                "model": model,
                "text": row.get("text", ""),
                "human_consensus": row.get("human_consensus", ""),
                "human_ambiguity": row.get("human_ambiguity", 0),
                "human_agreement": row.get("human_agreement", 0),
                "model_label": None,
                "model_confidence": None,
                "model_justification": "",
                "model_evidence": "",
                "latency": 0,
                "attempts": 0,
                "execution_status": "not_executed_after_stop",
                "failure_class": stop_class,
                "error": None,
                "error_reason": f"Não executado após interrupção: {stop_reason}",
                "raw_response": "",
                "coherence": 0,
                "explainability": 0,
            }
            model_rows.append(item)
            continue

        context = str(row.get(context_col, "")) if context_col and context_col in row else ""
        text = str(row.get("text", ""))

        attempts = 0
        total_latency = 0.0
        result = {}
        error = None

        # Uma tentativa inicial + no máximo uma repetição para falhas transitórias/desconhecidas.
        while attempts < 2:
            attempts += 1
            result = classify_comment(provider, model, text, context)
            total_latency += float(result.get("latency") or 0)
            error = result.get("error")
            if not result.get("label") and not error:
                error = "Resposta inválida ou rótulo ausente."

            if not error:
                break
            if attempts >= 2 or not _should_retry(error):
                break
            retry_class = _failure_class(error)
            # Rate limit precisa de backoff real; 0,5 s apenas repete o mesmo 429.
            # Valor configurável para adequação ao plano/cota do provedor.
            if retry_class == "rate_limited":
                time.sleep(float(os.getenv("VSSM_RATE_LIMIT_BACKOFF", "15")))
            else:
                time.sleep(0.5)

        failure_class = _failure_class(error)

        if error:
            observed_failures += 1
            # Apenas falhas definitivas contam para o limite de interrupção.
            # Transitórias que persistem após retry interrompem por instabilidade ao atingir o limite,
            # mas não caracterizam reprovação do modelo.
            consecutive_definitive_failures += 1
        else:
            consecutive_definitive_failures = 0

        item = {
            "row_id": int(idx),
            "record_id": record_id,
            "provider": provider,
            "model": model,
            "text": row.get("text", ""),
            "human_consensus": row.get("human_consensus", ""),
            "human_ambiguity": row.get("human_ambiguity", 0),
            "human_agreement": row.get("human_agreement", 0),
            "model_label": result.get("label"),
            "model_confidence": result.get("confidence"),
            "model_justification": result.get("justification"),
            "model_evidence": result.get("evidence"),
            "latency": total_latency,
            "attempts": attempts,
            "execution_status": "failed" if error else "success",
            "failure_class": failure_class if error else "",
            "error": error,
            "error_reason": _failure_reason(error),
            "raw_response": result.get("raw_response"),
        }
        item["coherence"] = lexical_coherence(item)
        item["explainability"] = explainability_score(item)
        model_rows.append(item)

        _update_run_model(
            run_id, provider, model,
            processed=position + 1,
            valid_predictions=sum(1 for rr in model_rows if rr.get("model_label")),
            errors=observed_failures,
        )

        if error and failure_class in {"permanent", "quota_exhausted"}:
            # Falha permanente confirmada: continuar não tende a produzir sucesso.
            stopped = True
            stop_class = failure_class
            stop_reason = _failure_reason(error)
        elif error and consecutive_definitive_failures >= failure_limit:
            stopped = True
            # Cinco falhas transitórias persistentes indicam instabilidade técnica,
            # não reprovação semântica do modelo.
            if failure_class == "rate_limited":
                stop_class = "rate_limited"
                stop_reason = f"Execução interrompida por limitação operacional de taxa após {failure_limit} falhas consecutivas persistentes."
            elif failure_class in {"transient", "unknown"}:
                stop_class = "temporary_instability"
                stop_reason = f"Execução interrompida por instabilidade técnica após {failure_limit} falhas consecutivas persistentes."
            else:
                stop_class = "model_output_failure"
                stop_reason = f"Execução interrompida após {failure_limit} falhas consecutivas de resposta do modelo."

    mdf = pd.DataFrame(model_rows)
    valid = mdf[mdf["model_label"].notna()].copy()

    invalid_count = int(total_rows - len(valid))
    error_count = int((mdf.get("execution_status") == "failed").sum()) if len(mdf) else 0
    error_rate = invalid_count / total_rows if total_rows else 1.0
    coverage = (len(valid) / total_rows) if total_rows else 0.0
    excluded = bool(total_rows == 0 or len(valid) != total_rows)

    if len(valid):
        cls = classification_metrics(valid["human_consensus"], valid["model_label"])
        avg_coherence = float(valid["coherence"].mean())
        avg_exp = float(valid["explainability"].mean())
        avg_latency = float(valid["latency"].mean())
        score = sarh_score(cls) if not excluded else 0
        sensitivity = sarh_sensitivity(cls) if not excluded else {}
    else:
        cls = {}
        avg_coherence = avg_exp = avg_latency = 0
        score = 0
        sensitivity = {}

    if not excluded:
        model_outcome = "completed"
        failure_reason = ""
    elif stop_class == "rate_limited":
        model_outcome = "rate_limited"
        failure_reason = stop_reason + " Modelo não avaliado; recomenda-se retomar após renovação/aumento do limite ou com cadência compatível."
    elif stop_class == "temporary_instability":
        model_outcome = "temporary_instability"
        failure_reason = stop_reason + " Modelo não avaliado; recomenda-se retomar a execução quando o serviço estiver disponível."
    elif stop_class in {"permanent", "quota_exhausted"}:
        model_outcome = "permanent_failure"
        failure_reason = stop_reason
    elif stop_class == "model_output_failure":
        model_outcome = "model_output_failure"
        failure_reason = stop_reason
    else:
        # Cobertura incompleta sem interrupção antecipada.
        classes = [x for x in mdf.get("failure_class", pd.Series(dtype=str)).tolist() if x]
        if classes and all(x == "rate_limited" for x in classes):
            model_outcome = "rate_limited"
            failure_reason = "Cobertura incompleta devido a limitação operacional de taxa; modelo não avaliado nesta execução."
        elif classes and all(x in {"transient", "unknown"} for x in classes):
            model_outcome = "temporary_instability"
            failure_reason = "Cobertura incompleta devido a falhas técnicas transitórias; modelo não avaliado nesta execução."
        else:
            model_outcome = "incomplete"
            failure_reason = "Cobertura incompleta; modelo fora do ranking para preservar o mesmo denominador entre os modelos."

    summary = {
        "provider": provider,
        "model": model,
        "total_rows": int(total_rows),
        "valid_predictions": int(len(valid)),
        "invalid_predictions": int(invalid_count),
        "observed_failures": int(observed_failures),
        "not_executed_after_stop": int(not_executed_after_stop),
        "errors": int(error_count),
        "error_rate": round(error_rate * 100, 2),
        "coverage": round(coverage * 100, 2),
        "rank_eligible": not excluded,
        "excluded": excluded,
        "model_outcome": model_outcome,
        "failure_reason": failure_reason,
        "accuracy": cls.get("accuracy", 0),
        "kappa": cls.get("kappa", 0),
        "macro_f1": cls.get("macro_f1", 0),
        "macro_precision": cls.get("macro_precision", 0),
        "macro_recall": cls.get("macro_recall", 0),
        "avg_coherence": avg_coherence,
        "avg_explainability": avg_exp,
        "avg_latency": avg_latency,
        "sarh": score,
        "sarh_sensitivity": sensitivity,
        "metrics": cls,
    }

    final_status = "done" if not excluded else (
        "temporary_instability" if model_outcome == "temporary_instability" else "failed"
    )
    _update_run_model(
        run_id, provider, model,
        status=final_status,
        message=failure_reason if excluded else "Concluído",
        processed=int(total_rows - not_executed_after_stop),
        valid_predictions=int(len(valid)),
        errors=int(observed_failures),
        error_rate=round(error_rate * 100, 2),
    )
    return model_rows, summary

def _run_validation_background(
    app,
    run_id,
    prepared_path,
    selected,
    results_folder,
    dataset_type,
    quality,
):
    with app.app_context():
        try:
            df = pd.read_csv(prepared_path)
            # Usa integralmente a referência humana consolidada; não há truncamento
            # silencioso por quantidade máxima de comentários.
            df = df.copy()
            df["_row_id"] = df.index
            # Identificador estável dentro da referência consolidada da execução.
            if "record_id" not in df.columns:
                df["record_id"] = [f"VSSM-{i+1:04d}" for i in range(len(df))]

            total_rows = len(df)
            df_records = df.fillna("").to_dict(orient="records")

            all_rows = []
            summaries = []

            workers = max(1, min(FRAMEWORK_MAX_WORKERS, len(selected)))

            _update_run_status(
                run_id,
                total_rows=total_rows,
                max_workers=workers,
                message="Validação em execução.",
            )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _execute_single_model,
                        chosen,
                        df_records,
                        total_rows,
                        None,
                        FRAMEWORK_MODEL_FAILURE_LIMIT,
                        run_id,
                    )
                    for chosen in selected
                ]

                for future in as_completed(futures):
                    model_rows, summary = future.result()
                    all_rows.extend(model_rows)
                    summaries.append(summary)

            summaries = sorted(
                summaries,
                key=lambda x: (x.get("sarh", 0), x.get("macro_f1", 0), x.get("kappa", 0), x.get("macro_recall", 0), x.get("accuracy", 0), -x.get("avg_latency", 0)),
                reverse=True,
            )

            results_df = pd.DataFrame(all_rows)

            results_path = Path(results_folder) / f"svsm_predictions_{run_id}.csv"
            summary_path = Path(results_folder) / f"svsm_summary_{run_id}.json"

            results_df.to_csv(results_path, index=False)
            summary_path.write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            valid_summaries = [s for s in summaries if not s.get("excluded")]
            rate_limited_summaries = [
                s for s in summaries
                if s.get("excluded") and s.get("model_outcome") == "rate_limited"
            ]
            temporary_summaries = [
                s for s in summaries
                if s.get("excluded") and s.get("model_outcome") == "temporary_instability"
            ]
            failed_summaries = [
                s for s in summaries
                if s.get("excluded") and s.get("model_outcome") not in {"temporary_instability", "rate_limited"}
            ]
            winner = valid_summaries[0] if valid_summaries else None

            # Os artefatos finais já existem neste ponto. Libera a navegação
            # imediatamente; o registro histórico é secundário e não pode bloquear
            # a exibição dos resultados.
            _update_run_status(
                run_id,
                status="done",
                message="Validação concluída.",
                results_url=f"/results?run_id={run_id}",
            )

            try:
                append_winner_history(
                    results_folder,
                    {
                        "run_id": run_id,
                        "dataset_type": dataset_type,
                        "total_comments": int(quality.get("original_rows", total_rows)),
                        "valid_comments": int(quality.get("valid_rows", total_rows)),
                        "tie_comments": int(quality.get("tie_rows", 0)),
                        "dataset_rows": int(total_rows),
                        "models_selected": int(len(selected)),
                        "models_tested": int(len(summaries)),
                        "models_valid": int(len(valid_summaries)),
                        "models_rate_limited": int(len(rate_limited_summaries)),
                        "models_temporary": int(len(temporary_summaries)),
                        "models_failed": int(len(failed_summaries)),
                        "valid_models": int(len(valid_summaries)),
                        "rate_limited_models": int(len(rate_limited_summaries)),
                        "temporary_models": int(len(temporary_summaries)),
                        "failed_models": int(len(failed_summaries)),
                        "winner_provider": winner.get("provider") if winner else "",
                        "winner_model": winner.get("model") if winner else "",
                        "sarh": winner.get("sarh", 0) if winner else 0,
                        "macro_f1": winner.get("macro_f1", 0) if winner else 0,
                        "accuracy": winner.get("accuracy", 0) if winner else 0,
                        "kappa": winner.get("kappa", 0) if winner else 0,
                        "reference_mode": quality.get("reference_mode", ""),
                        "human_agreement": quality.get("avg_agreement", 0),
                        "human_ambiguity": quality.get("avg_ambiguity", 0),
                        "max_workers": workers,
                        "failure_limit": FRAMEWORK_MODEL_FAILURE_LIMIT,
                    },
                )
            except Exception:
                pass

        except Exception as e:
            # Falha global não deve apontar para /results: os artefatos finais
            # podem não existir. O frontend encerra o polling e permanece na
            # Fase 4 exibindo o diagnóstico.
            _update_run_status(
                run_id,
                status="failed",
                message=str(e),
                results_url=None,
            )


@bp.route("/")
def index():
    history = load_winner_history(current_app.config["RESULTS_FOLDER"])

    return render_template(
        "index.html",
        status=provider_status(),
        latest_history=history[0] if history else None,
        last_winner=history[0] if history else None,
        history_count=len(history),
    )


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            flash("Selecione uma base CSV ou XLSX.", "error")
            return redirect(url_for("main.upload"))

        try:
            path = save_upload(file, current_app.config["UPLOAD_FOLDER"])
            df = load_dataframe(path)

            session["uploaded_path"] = path
            session["profile"] = profile_dataframe(df)

            flash("Base carregada com sucesso. Agora selecione as colunas.", "success")
            return redirect(url_for("main.configure"))

        except Exception as e:
            flash(str(e), "error")

    return render_template("upload.html")


@bp.route("/configure", methods=["GET", "POST"])
def configure():
    path = session.get("uploaded_path")

    if not path:
        return redirect(url_for("main.upload"))

    df = load_dataframe(path)
    columns = list(df.columns)
    preview = df.head(8).fillna("").to_dict(orient="records")

    if request.method == "POST":
        text_col = request.form.get("text_col")
        reference_mode = request.form.get("reference_mode", "final")
        reference_col = request.form.get("reference_col")
        label_cols = request.form.getlist("label_cols")
        dataset_type = request.form.get("dataset_type") or "Não informado"

        if not text_col:
            flash("Escolha a coluna textual.", "error")
            return redirect(url_for("main.configure"))

        try:
            if reference_mode == "multiple":
                label_cols = [col for col in label_cols if col != text_col]

                if len(label_cols) < 2:
                    flash("Selecione pelo menos duas colunas de avaliadores humanos.", "error")
                    return redirect(url_for("main.configure"))

                prepared, quality = prepare_labeled_data(
                    df,
                    text_col=text_col,
                    label_cols=label_cols,
                    optional_cols=[],
                    reference_col=None,
                )

            else:
                if not reference_col:
                    flash("Escolha a coluna de rótulo humano final.", "error")
                    return redirect(url_for("main.configure"))

                if reference_col == text_col:
                    flash("A coluna textual não pode ser usada como rótulo humano final.", "error")
                    return redirect(url_for("main.configure"))

                prepared, quality = _prepare_data_with_reference(
                    df,
                    text_col=text_col,
                    reference_col=reference_col,
                )

            # A auditoria detalhada é persistida fora da sessão para manter a
            # rastreabilidade inclusive dos registros excluídos do ground truth.
            audit_records = quality.pop("audit_records", [])
            if audit_records:
                audit_path = Path(current_app.config["RESULTS_FOLDER"]) / f"human_audit_{uuid.uuid4().hex}.csv"
                pd.DataFrame(audit_records).to_csv(audit_path, index=False)
                session["human_audit_path"] = str(audit_path)
            else:
                session.pop("human_audit_path", None)

            prepared = prepared.rename(columns={text_col: "text"})

            if "text" not in prepared.columns:
                raise ValueError("Não foi possível mapear a coluna textual.")

            out = Path(current_app.config["RESULTS_FOLDER"]) / f"prepared_{uuid.uuid4().hex}.csv"
            prepared.to_csv(out, index=False)

            session["prepared_path"] = str(out)
            session["quality"] = quality
            session["text_col"] = text_col
            session["reference_mode"] = reference_mode
            session["reference_col"] = reference_col if reference_mode == "final" else None
            session["label_cols"] = [reference_col] if reference_mode == "final" else label_cols
            session["optional_cols"] = []
            session["dataset_type"] = dataset_type

            if reference_mode == "multiple":
                flash("Base validada com consolidação automática por maioria.", "success")
            else:
                flash("Base validada com rótulo humano final.", "success")

            return redirect(url_for("main.data_quality"))

        except Exception as e:
            flash(str(e), "error")

    return render_template(
        "configure.html",
        columns=columns,
        preview=preview,
        profile=session.get("profile", {}),
    )


@bp.route("/data-quality")
def data_quality():
    if not session.get("prepared_path"):
        return redirect(url_for("main.upload"))

    audit_rows = []
    audit_path = session.get("human_audit_path")
    if audit_path and Path(audit_path).exists():
        audit_df = pd.read_csv(audit_path, keep_default_na=False)
        status_filter = request.args.get("status", "todos")
        if status_filter != "todos" and "human_status" in audit_df.columns:
            audit_df = audit_df[audit_df["human_status"] == status_filter]
        audit_rows = audit_df.to_dict(orient="records")
    else:
        status_filter = "todos"

    return render_template(
        "data_quality.html",
        quality=session.get("quality", {}),
        profile=session.get("profile", {}),
        audit_rows=audit_rows,
        audit_status=status_filter,
    )


@bp.route("/models", methods=["GET", "POST"])
def models():
    discovered = discover_models()

    if request.method == "POST":
        # Revalida no servidor cada seleção contra a descoberta atual.
        # A interface não é fonte de autoridade para elegibilidade.
        allowed = {
            (str(p.get("provider", "")), str(m.get("id", "")))
            for p in discovered
            for m in p.get("eligible_models", [])
            if m.get("eligible") is True
        }

        selected = []
        rejected = []
        seen = set()

        for key in request.form.keys():
            if key.startswith("model__"):
                provider = key.split("__", 1)[1]
                for model in request.form.getlist(key):
                    pair = (provider, model)
                    if pair in allowed and pair not in seen:
                        selected.append({"provider": provider, "model": model})
                        seen.add(pair)
                    elif pair not in allowed:
                        rejected.append(f"{provider}/{model}")

        if rejected:
            flash("Uma ou mais seleções não correspondem a modelos atualmente descobertos e elegíveis pela triagem. Atualize a lista e selecione novamente.", "error")
            return redirect(url_for("main.models"))

        if not selected:
            flash("Selecione pelo menos um modelo elegível pela triagem.", "error")
            return redirect(url_for("main.models"))

        # Elegibilidade é definida pela capacidade para a tarefa, não por quota/crédito momentâneos.
        # Não há probe prévio: a disponibilidade operacional é observada na execução real.
        session.pop("selected_probe_failures", None)
        session.pop("selected_probe_results", None)
        session["selected_models"] = selected
        flash(
            "Modelos elegíveis selecionados. A disponibilidade operacional será registrada durante a execução.",
            "success",
        )
        return redirect(url_for("main.run_validation"))

    return render_template("models.html", discovered=discovered)


@bp.route("/run", methods=["GET"])
def run_validation():
    prepared_path = session.get("prepared_path")
    selected = session.get("selected_models", [])

    if not prepared_path:
        return redirect(url_for("main.upload"))

    if not selected:
        return redirect(url_for("main.models"))

    df = pd.read_csv(prepared_path)

    return render_template(
        "run.html",
        selected=selected,
        rows=len(df),
        max_workers=FRAMEWORK_MAX_WORKERS,
        failure_limit=FRAMEWORK_MODEL_FAILURE_LIMIT,
    )


@bp.route("/run-start", methods=["POST"])
def run_start():
    prepared_path = session.get("prepared_path")
    selected = session.get("selected_models", [])
    quality = session.get("quality", {})

    if not prepared_path or not selected:
        return jsonify({"error": "Sessão inválida. Carregue a base e selecione os modelos."}), 400

    run_id = uuid.uuid4().hex

    with RUN_STATUS_LOCK:
        RUN_STATUS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "message": "Validação iniciada.",
            "models": {
                _model_key(s["provider"], s["model"]): {
                    "provider": s["provider"],
                    "model": s["model"],
                    "status": "waiting",
                    "message": "Aguardando",
                    "processed": 0,
                    "valid_predictions": 0,
                    "errors": 0,
                    "error_rate": 0,
                }
                for s in selected
            },
            "results_url": None,
        }

    app = current_app._get_current_object()
    results_folder = current_app.config["RESULTS_FOLDER"]
    dataset_type = session.get("dataset_type", "Não informado")

    thread = threading.Thread(
        target=_run_validation_background,
        args=(
            app,
            run_id,
            prepared_path,
            selected,
            results_folder,
            dataset_type,
            quality,
        ),
        daemon=True,
    )
    thread.start()

    return jsonify({"run_id": run_id})


@bp.route("/run-status/<run_id>")
def run_status(run_id):
    status = _get_run_status(run_id)

    # Fonte de verdade adicional: se os artefatos finais já foram persistidos,
    # a execução está concluída mesmo que o estado em memória tenha sido perdido
    # (reload/processo diferente) ou ainda não tenha sido atualizado.
    results_folder = Path(current_app.config["RESULTS_FOLDER"])
    summary_path = results_folder / f"svsm_summary_{run_id}.json"
    predictions_path = results_folder / f"svsm_predictions_{run_id}.csv"
    if summary_path.exists() and predictions_path.exists():
        status = dict(status or {})
        status.update({
            "run_id": run_id,
            "status": "done",
            "message": "Validação concluída.",
            "results_url": url_for("main.results", run_id=run_id),
        })

    return jsonify(status)


@bp.route("/results")
def results():
    run_id = request.args.get("run_id")

    if run_id:
        summary_path = Path(current_app.config["RESULTS_FOLDER"]) / f"svsm_summary_{run_id}.json"
        results_path = Path(current_app.config["RESULTS_FOLDER"]) / f"svsm_predictions_{run_id}.csv"
    else:
        summary_path = session.get("summary_path")
        results_path = session.get("results_path")

    if not summary_path or not Path(summary_path).exists():
        return redirect(url_for("main.run_validation"))

    summaries = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    valid_summaries = [s for s in summaries if not s.get("excluded")]
    rate_limited_summaries = [
        s for s in summaries
        if s.get("excluded") and s.get("model_outcome") == "rate_limited"
    ]
    temporary_summaries = [
        s for s in summaries
        if s.get("excluded") and s.get("model_outcome") == "temporary_instability"
    ]
    failed_summaries = [
        s for s in summaries
        if s.get("excluded") and s.get("model_outcome") not in {"temporary_instability", "rate_limited"}
    ]

    valid_summaries = sorted(
        valid_summaries,
        key=lambda x: (x.get("sarh", 0), x.get("macro_f1", 0), x.get("kappa", 0), x.get("macro_recall", 0), x.get("accuracy", 0), -x.get("avg_latency", 0)),
        reverse=True,
    )

    failed_summaries = sorted(
        failed_summaries,
        key=lambda x: x.get("error_rate", 0),
        reverse=True,
    )

    winner = valid_summaries[0] if valid_summaries else None

    sensitivity_winners = {}
    for scenario in ["principal", "A", "B", "C"]:
        candidates = [s for s in valid_summaries if s.get("sarh_sensitivity")]
        if candidates:
            ranked = sorted(candidates, key=lambda s: s.get("sarh_sensitivity", {}).get(scenario, 0), reverse=True)
            sensitivity_winners[scenario] = {
                "provider": ranked[0].get("provider"),
                "model": ranked[0].get("model"),
                "score": ranked[0].get("sarh_sensitivity", {}).get(scenario, 0),
            }
    sensitivity_stable = bool(sensitivity_winners) and len({(v["provider"], v["model"]) for v in sensitivity_winners.values()}) == 1

    executive = {
        "comments": int(max([s.get("total_rows", 0) for s in summaries], default=0)),
        "models_tested": len(summaries),
        "valid_models": len(valid_summaries),
        "rate_limited_models": len(rate_limited_summaries),
        "temporary_models": len(temporary_summaries),
        "failed_models": len(failed_summaries),
    }

    sample = []

    if results_path and Path(results_path).exists():
        df = pd.read_csv(results_path).fillna("")

        if winner:
            sample_df = df[(df["provider"] == winner["provider"]) & (df["model"] == winner["model"])]
            sample = sample_df.head(20).to_dict(orient="records")
        else:
            sample = df.head(20).to_dict(orient="records")

    recommendation = ""

    if winner:
        recommendation = (
            f"O modelo {winner['provider']} / {winner['model']} apresentou o maior "
            f"SARH entre os modelos operacionalmente elegíveis. O SARH combina Macro-F1, Kappa, "
            f"Macro-Recall e Accuracy e representa a aderência à referência humana consolidada."
        )

    return render_template(
        "results.html",
        summaries=valid_summaries,
        rate_limited_summaries=rate_limited_summaries,
        temporary_summaries=temporary_summaries,
        failed_summaries=failed_summaries,
        winner=winner,
        sample=sample,
        executive=executive,
        recommendation=recommendation,
        sensitivity_winners=sensitivity_winners,
        sensitivity_stable=sensitivity_stable,
    )


@bp.route("/history")
def history():
    history_items = load_winner_history(current_app.config["RESULTS_FOLDER"])
    return render_template("history.html", history=history_items)


@bp.route("/history/<run_id>")
def history_detail(run_id):
    results_folder = Path(current_app.config["RESULTS_FOLDER"])
    summary_path = results_folder / f"svsm_summary_{run_id}.json"
    predictions_path = results_folder / f"svsm_predictions_{run_id}.csv"

    history_items = load_winner_history(current_app.config["RESULTS_FOLDER"])
    record = next((item for item in history_items if item.get("run_id") == run_id), None)

    if not record:
        flash("Registro histórico não encontrado.", "error")
        return redirect(url_for("main.history"))

    summaries = []
    valid_summaries = []
    rate_limited_summaries = []
    temporary_summaries = []
    failed_summaries = []
    failure_counts = {}

    if summary_path.exists():
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        valid_summaries = [s for s in summaries if not s.get("excluded")]
        rate_limited_summaries = [
            s for s in summaries
            if s.get("excluded") and s.get("model_outcome") == "rate_limited"
        ]
        temporary_summaries = [
            s for s in summaries
            if s.get("excluded") and s.get("model_outcome") == "temporary_instability"
        ]
        failed_summaries = [
            s for s in summaries
            if s.get("excluded") and s.get("model_outcome") not in {"temporary_instability", "rate_limited"}
        ]

        reasons = [
            s.get("failure_reason") or "Falha técnica ou ausência de previsões válidas."
            for s in failed_summaries
        ]
        failure_counts = dict(Counter(reasons))

    sample = []
    if predictions_path.exists():
        df = pd.read_csv(predictions_path).fillna("")
        sample = df.head(15).to_dict(orient="records")

    return render_template(
        "history_detail.html",
        record=record,
        summaries=valid_summaries,
        rate_limited_summaries=rate_limited_summaries,
        temporary_summaries=temporary_summaries,
        failed_summaries=failed_summaries,
        failure_counts=failure_counts,
        sample=sample,
        run_id=run_id,
    )


@bp.route("/documentation")
def documentation():
    return render_template("documentation.html")


@bp.route("/download/<kind>")
def download(kind):
    run_id = request.args.get("run_id")

    if run_id:
        if kind == "results":
            path = Path(current_app.config["RESULTS_FOLDER"]) / f"svsm_predictions_{run_id}.csv"
        elif kind == "summary":
            path = Path(current_app.config["RESULTS_FOLDER"]) / f"svsm_summary_{run_id}.json"
        else:
            path = None
    else:
        mapping = {
            "results": "results_path",
            "summary": "summary_path",
            "prepared": "prepared_path",
        }
        path = session.get(mapping.get(kind, ""))

    if not path or not Path(path).exists():
        flash("Arquivo não encontrado.", "error")
        return redirect(url_for("main.index"))

    return send_file(path, as_attachment=True)


@bp.route("/reset")
def reset():
    session.clear()
    flash("Sessão reiniciada.", "success")
    return redirect(url_for("main.index"))
