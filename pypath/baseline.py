import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELEASE_PALS_ROOT = os.path.join(REPO_ROOT, "release", "pals_data")
DEFAULT_EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "default")
DEFAULT_PROPOSER_EXPERIMENT_ROOT = os.path.join(DEFAULT_EXPERIMENT_ROOT, "proposer")
DEFAULT_PROPOSER_ROOT = os.path.join(DEFAULT_PROPOSER_EXPERIMENT_ROOT, "legacy")
DEFAULT_PROPOSER_DOCS_ROOT = os.path.join(DEFAULT_PROPOSER_EXPERIMENT_ROOT, "docs")
DEFAULT_PROPOSER_EVAL_ROOT = os.path.join(DEFAULT_PROPOSER_EXPERIMENT_ROOT, "evaluations", "conflict_alignment")
DEFAULT_PROPOSER_DATASETS_ROOT = os.path.join(DEFAULT_PROPOSER_EXPERIMENT_ROOT, "datasets")
DEFAULT_AGGREGATOR_ROOT = os.path.join(DEFAULT_EXPERIMENT_ROOT, "aggregator")
DEFAULT_AGGREGATOR_MODEL_ROOT = os.path.join(DEFAULT_AGGREGATOR_ROOT, "model")
DEFAULT_AGGREGATOR_TRAINING_DATA_ROOT = os.path.join(DEFAULT_AGGREGATOR_ROOT, "training_data")
DEFAULT_PROPOSER_DATASET_TAG = "warmup_proposer_independent_midscale_500circuits_v1_onehop_segment_entry"
DEFAULT_PROPOSER_RUN_TAG = f"{DEFAULT_PROPOSER_DATASET_TAG}_lg_e80"


@dataclass(frozen=True)
class ProposerBaselineConfig:
    num_circuits_per_uut: int = 500
    look_back: int = 3
    preprocess_batch_size: int = 10
    test_split_ratio: float = 0.2
    split_seed: int = 42
    train_epochs: int = 100
    train_batch_size: int = 20
    model_subdir: str = "full_iterno_embedding_input_global_model"
    model_path: str = ""


@dataclass(frozen=True)
class AggregatorBaselineConfig:
    num_circuits_to_generate: int = 500
    min_instances: int = 5
    max_instances: int = 25
    max_fanout: int = 3
    coupler_data_filename: str = "coupler_data.teacher_dual_head.json"
    coupler_audit_filename: str = "coupler_data.teacher_dual_head.audit.json"
    stage: int = 3
    train_epochs: int = 80
    entry_batch_size: int = 4
    split_seed: int = 42
    train_seed: int = 42
    sampling_seed: int = 42
    stage3_variant: str = "port_regime_hierarchical_prop_attention"
    stage3_regime_boundaries: Tuple[float, ...] = (-12.0, -9.0)
    model_subdir: str = "stage3_full_graph_teacher_e80"
    warmup_num_trajectories: int = 50
    warmup_seed: int = 42
    warmup_k: int = 1
    warmup_max_newton_iters: int = 20
    warmup_residual_tol: float = 1e-6
    warmup_enable_fallback: bool = False


PROPOSER_BASELINE = ProposerBaselineConfig()
AGGREGATOR_BASELINE = AggregatorBaselineConfig()

PROPOSER_BASELINE_REGISTRY_PATH = os.path.join(
    REPO_ROOT, "pypath", "proposer", "baselines", "proposer_current_baseline.json"
)
AGGREGATOR_BASELINE_REGISTRY_PATH = os.path.join(
    REPO_ROOT, "pypath", "aggregator", "baselines", "aggregator_current_baseline.json"
)


def _sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("tag must not be empty")
    return cleaned


def _ensure_warmup_tag(tag: str) -> str:
    cleaned = _sanitize_tag(tag)
    if cleaned.startswith("warmup_proposer_"):
        return cleaned
    return f"warmup_proposer_{cleaned}"


def _read_current_baseline_entry(registry_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(registry_path):
        return None
    with open(registry_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = payload.get("entry")
    return entry if isinstance(entry, dict) else None


def current_published_proposer_baseline_model_path() -> str:
    entry = _read_current_baseline_entry(PROPOSER_BASELINE_REGISTRY_PATH)
    rel_path = entry.get("artifact_model_path") if entry else None
    if rel_path:
        candidate = os.path.join(REPO_ROOT, rel_path)
        if os.path.isfile(candidate):
            return candidate
    if PROPOSER_BASELINE.model_path:
        return os.path.abspath(PROPOSER_BASELINE.model_path)
    return ""


def current_published_aggregator_baseline_model_path() -> str:
    entry = _read_current_baseline_entry(AGGREGATOR_BASELINE_REGISTRY_PATH)
    rel_path = entry.get("artifact_model_path") if entry else None
    if rel_path:
        candidate = os.path.join(REPO_ROOT, rel_path)
        if os.path.isfile(candidate):
            return candidate
    return ""


def default_proposer_experiment_model_path(
    dataset_tag: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> str:
    normalized_dataset_tag = _ensure_warmup_tag(dataset_tag or DEFAULT_PROPOSER_DATASET_TAG)
    normalized_run_tag = _ensure_warmup_tag(run_tag or DEFAULT_PROPOSER_RUN_TAG)
    return os.path.join(
        DEFAULT_PROPOSER_DATASETS_ROOT,
        normalized_dataset_tag,
        "models",
        normalized_run_tag,
        "best_lg_model.pth",
    )


def proposer_model_dir(uut_name: str, local_data_root: str = DEFAULT_PROPOSER_ROOT) -> str:
    return os.path.join(local_data_root, uut_name, PROPOSER_BASELINE.model_subdir)


def proposer_model_path(
    uut_name: str,
    local_data_root: str = DEFAULT_PROPOSER_ROOT,
    filename: str = "best_lg_model.pth",
    prefer_published_baseline: bool = False,
) -> str:
    if prefer_published_baseline:
        published = current_published_proposer_baseline_model_path()
        if published:
            return published
    experiment_default = default_proposer_experiment_model_path()
    if os.path.isfile(experiment_default):
        return experiment_default
    return os.path.join(proposer_model_dir(uut_name, local_data_root=local_data_root), filename)


def proposer_model_name(model_dir: str = None, model_path: str = None) -> str:
    if model_dir:
        return os.path.basename(os.path.abspath(model_dir))
    if model_path:
        return os.path.basename(os.path.dirname(os.path.abspath(model_path)))
    experiment_default = default_proposer_experiment_model_path()
    if os.path.isfile(experiment_default):
        return os.path.basename(os.path.dirname(os.path.abspath(experiment_default)))
    published = current_published_proposer_baseline_model_path()
    if published:
        return os.path.basename(os.path.dirname(os.path.abspath(published)))
    return PROPOSER_BASELINE.model_subdir


def proposer_training_summary_json_path(
    model_dir: str = None,
    docs_root: str = DEFAULT_PROPOSER_DOCS_ROOT,
) -> str:
    model_name = proposer_model_name(model_dir=model_dir)
    return os.path.join(docs_root, f"{model_name}.training_summary.json")


def proposer_training_log_path(
    model_dir: str = None,
    timestamp: str = None,
    docs_root: str = DEFAULT_PROPOSER_DOCS_ROOT,
) -> str:
    model_name = proposer_model_name(model_dir=model_dir)
    suffix = f".train_{timestamp}.log" if timestamp else ".train.log"
    return os.path.join(docs_root, f"{model_name}{suffix}")


def proposer_conflict_alignment_root(
    model_dir: str = None,
    model_path: str = None,
    eval_root: str = DEFAULT_PROPOSER_EVAL_ROOT,
) -> str:
    return os.path.join(eval_root, proposer_model_name(model_dir=model_dir, model_path=model_path))


def aggregator_model_dir(stage: int = None, model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT) -> str:
    if stage is None:
        stage = AGGREGATOR_BASELINE.stage
    if int(stage) == int(AGGREGATOR_BASELINE.stage):
        return os.path.join(model_root, AGGREGATOR_BASELINE.model_subdir)
    return os.path.join(model_root, f"simple_model_stage_{int(stage)}")


def aggregator_model_path(
    stage: int = None,
    model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT,
) -> str:
    if stage is None:
        stage = AGGREGATOR_BASELINE.stage
    return os.path.join(aggregator_model_dir(stage=stage, model_root=model_root), f"simple_model_stage_{int(stage)}.pth")


def aggregator_diagnostics_path(
    stage: int = None,
    model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT,
) -> str:
    if stage is None:
        stage = AGGREGATOR_BASELINE.stage
    return os.path.join(
        aggregator_model_dir(stage=stage, model_root=model_root),
        f"training_diagnostics_stage_{int(stage)}.json",
    )


def aggregator_training_summary_path(
    stage: int = None,
    model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT,
) -> str:
    if stage is None:
        stage = AGGREGATOR_BASELINE.stage
    return os.path.join(
        aggregator_model_dir(stage=stage, model_root=model_root),
        f"training_summary_stage_{int(stage)}.md",
    )


def aggregator_training_log_path(
    stage: int = None,
    model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT,
) -> str:
    if stage is None:
        stage = AGGREGATOR_BASELINE.stage
    return os.path.join(aggregator_model_dir(stage=stage, model_root=model_root), "train.log")


def aggregator_training_data_path(training_data_root: str = DEFAULT_AGGREGATOR_TRAINING_DATA_ROOT) -> str:
    return os.path.join(training_data_root, "training_data.json")


def aggregator_coupler_data_path(training_data_root: str = DEFAULT_AGGREGATOR_TRAINING_DATA_ROOT) -> str:
    return os.path.join(training_data_root, AGGREGATOR_BASELINE.coupler_data_filename)


def aggregator_coupler_audit_path(training_data_root: str = DEFAULT_AGGREGATOR_TRAINING_DATA_ROOT) -> str:
    return os.path.join(training_data_root, AGGREGATOR_BASELINE.coupler_audit_filename)


def aggregator_warmup_output_dir(model_root: str = DEFAULT_AGGREGATOR_MODEL_ROOT) -> str:
    return os.path.join(aggregator_model_dir(model_root=model_root), "warmup_evaluation")
