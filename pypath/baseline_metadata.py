import json
import os
import re
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _relpath(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, REPO_ROOT)
    except ValueError:
        return abs_path


def _extract_timestamp_key(path: str) -> str:
    match = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    if match:
        return match.group(1)
    return os.path.basename(path)


def _parse_markdown_scalar(text: str, key: str) -> Optional[float]:
    pattern = rf"-\s+{re.escape(key)}:\s+([0-9eE+\-.]+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def _parse_markdown_int(text: str, key: str) -> Optional[int]:
    pattern = rf"-\s+{re.escape(key)}:\s+(\d+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def ensure_existing_file(path: Optional[str], label: str) -> str:
    if not path:
        raise FileNotFoundError(f"Missing required {label}: path is empty")
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Missing required {label}: {abs_path}")
    return abs_path


def ensure_existing_dir(path: Optional[str], label: str) -> str:
    if not path:
        raise FileNotFoundError(f"Missing required {label}: path is empty")
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"Missing required {label}: {abs_path}")
    return abs_path


def next_versioned_baseline_name(history_path: str, family: str) -> str:
    version = 1
    if os.path.isfile(history_path):
        payload = _read_json(history_path)
        for entry in payload.get("baselines", []):
            name = str(entry.get("baseline_name", ""))
            match = re.fullmatch(rf"{re.escape(family)}_v(\d+)", name)
            if match:
                version = max(version, int(match.group(1)) + 1)
    return f"{family}_v{version}"


def extract_baseline_version_tag(baseline_name: str) -> Optional[str]:
    match = re.search(r"_v(\d+)$", baseline_name.strip())
    if not match:
        return None
    return f"v{int(match.group(1))}"


def build_baseline_artifact_label(baseline_name: str, timestamp: str) -> str:
    baseline_slug = re.sub(r"[^a-z0-9]+", "_", baseline_name.strip().lower()).strip("_")
    version_tag = extract_baseline_version_tag(baseline_name)
    if version_tag:
        family_slug = re.sub(r"_v\d+$", "", baseline_slug).strip("_")
        if family_slug:
            return f"{version_tag}-baseline_{family_slug}_{timestamp}"
        return f"{version_tag}-baseline_{timestamp}"
    return f"baseline_{baseline_slug}_{timestamp}"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_baseline_summary(kind: str, entry: Dict[str, Any]) -> str:
    training = entry.get("training_metrics", {})
    selected_metrics = training.get("selected_metrics", {})
    conflicts = training.get("conflicts", [])
    eval_metrics = entry.get("evaluation_metrics", {})
    lines = [
        "# Baseline Summary",
        "",
        "## Current Version Main Features",
        "",
        f"- baseline_id: `{entry.get('baseline_id')}`",
        f"- baseline_name: `{entry.get('baseline_name')}`",
        f"- artifact_dir: `{entry.get('artifact_dir')}`",
        f"- source_git_commit: `{entry.get('source_git_commit')}`",
        f"- model_snapshot: `{entry.get('artifact_model_dir')}`",
        f"- training_log: `{entry.get('artifact_log_path')}`",
        f"- loss_plot: `{entry.get('artifact_loss_plot_path')}`",
    ]
    if kind == "proposer":
        lines.append(f"- conflict_alignment_dir: `{entry.get('artifact_conflict_alignment_dir')}`")
    else:
        lines.append(f"- warmup_evaluation_dir: `{entry.get('artifact_warmup_eval_dir')}`")

    lines.extend(
        [
            "",
            "## Progress",
            "",
            f"- training_metric_name: `{selected_metrics.get('metric_name')}`",
            f"- training_best_value: `{_fmt_metric(selected_metrics.get('best_value'))}`",
            f"- training_best_epoch: `{_fmt_metric(selected_metrics.get('best_epoch'))}`",
            f"- training_metric_source: `{selected_metrics.get('source_path')}`",
        ]
    )
    if kind == "proposer":
        conflict_summary = eval_metrics.get("conflict_alignment_summary", {})
        conflict_node = conflict_summary.get("conflict_node_level_mean_proposal", {})
        lines.extend(
            [
                f"- conflict_sign_agreement_rate: `{_fmt_metric(conflict_node.get('sign_agreement_rate'))}`",
                f"- conflict_mean_relative_error: `{_fmt_metric(conflict_node.get('mean_relative_error'))}`",
                f"- conflict_alignment_summary: `{conflict_summary.get('summary_json_path')}`",
            ]
        )
    else:
        warmup_summary = eval_metrics.get("warmup_summary", {})
        lines.extend(
            [
                f"- warmup_available_rate: `{_fmt_metric(warmup_summary.get('warmup_available_rate'))}`",
                f"- warmup_converged_rate: `{_fmt_metric(warmup_summary.get('warmup_converged_rate'))}`",
                f"- saved_newton_iters_rate: `{_fmt_metric(warmup_summary.get('saved_newton_iters_rate'))}`",
                f"- warmup_results_path: `{warmup_summary.get('results_path')}`",
            ]
        )

    lines.extend(["", "## Limitations", ""])
    if conflicts:
        for item in conflicts:
            lines.append(f"- conflict: `{json.dumps(item, ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("- no registry-level conflicts detected")
    if not entry.get("artifact_log_path"):
        lines.append("- raw training log was unavailable at publish time; packaged training record relies on summary/diagnostics artifacts.")
    lines.append("")
    return "\n".join(lines)


def summarize_aggregator_warmup(warmup_results_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not warmup_results_path or not os.path.isfile(warmup_results_path):
        return None
    payload = _read_json(warmup_results_path)
    summary = dict(payload.get("summary", {}))
    summary["results_path"] = _relpath(warmup_results_path)
    if "saved_newton_iters_rate" not in summary:
        total = summary.get("num_segments") or summary.get("num_trajectories") or 0
        improved = summary.get("improved_count")
        if improved is None:
            improved = summary.get("saved_newton_iters_count")
        summary["saved_newton_iters_rate"] = float(improved / total) if total else None
    return summary


def summarize_proposer_conflict_alignment(summary_json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not summary_json_path or not os.path.isfile(summary_json_path):
        return None
    payload = _read_json(summary_json_path)
    conflict = payload.get("conflict", {})
    non_conflict = payload.get("non_conflict", {})
    return {
        "summary_json_path": _relpath(summary_json_path),
        "model_path": _relpath(payload.get("model_path")),
        "num_node_records": payload.get("num_node_records"),
        "conflict_node_level_mean_proposal": conflict.get("node_level_mean_proposal"),
        "non_conflict_node_level_mean_proposal": non_conflict.get("node_level_mean_proposal"),
    }


def resolve_aggregator_training_metadata(
    *,
    model_dir: str,
    diagnostics_path: str,
    summary_path: Optional[str] = None,
    explicit_log_path: Optional[str] = None,
) -> Dict[str, Any]:
    diagnostics_payload = _read_json(diagnostics_path)
    val_losses = [float(value) for value in diagnostics_payload.get("val_losses", [])]
    diagnostics_metrics = {
        "metric_name": "best_val_loss",
        "best_value": _safe_float(diagnostics_payload.get("best_val_loss")),
        "best_epoch": diagnostics_payload.get("best_epoch"),
        "first_val_loss": val_losses[0] if val_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "epochs_completed": diagnostics_payload.get("epochs_completed"),
        "source_kind": "diagnostics_json",
        "source_path": _relpath(diagnostics_path),
    }

    summary_metrics = None
    conflicts: List[Dict[str, Any]] = []
    if summary_path and os.path.isfile(summary_path):
        summary_text = _read_text(summary_path)
        summary_metrics = {
            "metric_name": "best_val_loss",
            "best_value": _parse_markdown_scalar(summary_text, "best_val_loss"),
            "best_epoch": _parse_markdown_int(summary_text, "best_epoch"),
            "first_val_loss": _parse_markdown_scalar(summary_text, "first_val_loss"),
            "last_val_loss": _parse_markdown_scalar(summary_text, "last_val_loss"),
            "epochs_completed": _parse_markdown_int(summary_text, "epochs_completed"),
            "source_kind": "training_summary_md",
            "source_path": _relpath(summary_path),
        }
        if (
            summary_metrics["best_value"] is not None
            and diagnostics_metrics["best_value"] is not None
            and abs(summary_metrics["best_value"] - diagnostics_metrics["best_value"]) > 1e-9
        ):
            conflicts.append(
                {
                    "type": "best_val_loss_mismatch",
                    "diagnostics_best_val_loss": diagnostics_metrics["best_value"],
                    "summary_best_val_loss": summary_metrics["best_value"],
                }
            )

    selected_log_path = explicit_log_path if explicit_log_path and os.path.isfile(explicit_log_path) else None
    if selected_log_path is None:
        train_log = os.path.join(model_dir, "train.log")
        if os.path.isfile(train_log):
            selected_log_path = train_log

    return {
        "selected_metrics": diagnostics_metrics,
        "diagnostics_metrics": diagnostics_metrics,
        "summary_metrics": summary_metrics,
        "selected_log_path": _relpath(selected_log_path),
        "conflicts": conflicts,
    }


def _parse_proposer_training_log(log_path: str) -> Dict[str, Any]:
    text = _read_text(log_path)
    best_matches = [
        float(match.rstrip("."))
        for match in re.findall(r"New best val metric:\s*([0-9eE+\-.]+)", text)
    ]
    return {
        "log_path": _relpath(log_path),
        "is_eval_only": "Evaluation-only mode enabled" in text,
        "has_training": "Starting log-space training" in text,
        "best_metric_name": "best_val_metric",
        "best_metric_value": min(best_matches) if best_matches else None,
        "num_improvements": len(best_matches),
        "timestamp_key": _extract_timestamp_key(log_path),
    }


def resolve_proposer_training_metadata(
    *,
    model_dir: str,
    model_path: str,
    summary_json_path: str,
) -> Dict[str, Any]:
    summary_payload = _read_json(summary_json_path)
    docs_dir = os.path.dirname(os.path.abspath(summary_json_path))
    model_name = os.path.basename(model_dir)
    candidate_logs = []
    for name in os.listdir(docs_dir):
        if not name.startswith(f"{model_name}.train_") or not name.endswith(".log"):
            continue
        candidate_logs.append(os.path.join(docs_dir, name))
    parsed_logs = [
        _parse_proposer_training_log(path)
        for path in sorted(candidate_logs)
    ]

    selected_log = None
    training_logs = [
        item for item in parsed_logs
        if item["best_metric_value"] is not None and not item["is_eval_only"]
    ]
    if training_logs:
        selected_log = sorted(training_logs, key=lambda item: item["timestamp_key"])[-1]
    elif parsed_logs:
        selected_log = sorted(parsed_logs, key=lambda item: item["timestamp_key"])[-1]

    selected_metrics = {
        "metric_name": "best_val_metric",
        "best_value": selected_log["best_metric_value"] if selected_log else None,
        "best_epoch": None,
        "epochs_completed": summary_payload.get("epochs"),
        "source_kind": "training_log" if selected_log and selected_log["best_metric_value"] is not None else "training_summary_json",
        "source_path": selected_log["log_path"] if selected_log else _relpath(summary_json_path),
    }

    conflicts = []
    if summary_payload.get("eval_only") and selected_log and not selected_log["is_eval_only"]:
        conflicts.append(
            {
                "type": "summary_points_to_eval_only_run",
                "summary_json_path": _relpath(summary_json_path),
                "selected_training_log": selected_log["log_path"],
            }
        )

    return {
        "selected_metrics": selected_metrics,
        "summary_payload": {
            "summary_json_path": _relpath(summary_json_path),
            "training_log_path": _relpath(summary_payload.get("training_log_path")),
            "loss_plot_path": _relpath(summary_payload.get("loss_plot_path")),
            "eval_only": bool(summary_payload.get("eval_only")),
            "epochs": summary_payload.get("epochs"),
            "batch_size": summary_payload.get("batch_size"),
            "look_back": summary_payload.get("look_back"),
            "learn_rate": summary_payload.get("learn_rate"),
            "best_model_path": _relpath(summary_payload.get("best_model_path") or model_path),
        },
        "parsed_log_candidates": parsed_logs,
        "selected_log": selected_log,
        "conflicts": conflicts,
    }
