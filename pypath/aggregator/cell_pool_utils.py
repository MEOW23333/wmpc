"""Helpers for resolving the cell pool used by aggregator-side tooling."""

import os
from typing import Any, Dict, List

import proposer.config as config

try:
    from proposer.stdcell_pool import NEIGHBOR_CELL_POOL as FULL_STDCELL_POOL
except ImportError:
    FULL_STDCELL_POOL = {}


def get_aggregator_cell_pool() -> Dict[str, Dict[str, Any]]:
    """Return the broadest cell pool the aggregator side can safely recognize.

    The repo-level config currently keeps ``config.NEIGHBOR_CELL_POOL`` very small
    for focused training. Large-scale validation, however, needs to recognize a
    richer mixed-cell library. We therefore merge the auto-generated full pool
    with the current runtime config, while letting runtime config entries take
    precedence if both define the same cell type.
    """

    pool: Dict[str, Dict[str, Any]] = {}
    if FULL_STDCELL_POOL:
        pool.update(FULL_STDCELL_POOL)
    pool.update(config.NEIGHBOR_CELL_POOL)
    return pool


def list_trained_proposer_uuts(proposer_root: str = None) -> List[str]:
    """Return UUT names that already have a trained proposer checkpoint."""

    proposer_root = proposer_root or config.LOCAL_DATA_LOCATION
    if not os.path.isdir(proposer_root):
        return []

    covered: List[str] = []
    for uut_name in sorted(os.listdir(proposer_root)):
        model_path = os.path.join(
            proposer_root,
            uut_name,
            "full_iterno_embedding_input_global_model",
            "best_lg_model.pth",
        )
        if os.path.exists(model_path):
            covered.append(uut_name)
    return covered


def filter_cell_pool_by_trained_proposers(
    cell_pool: Dict[str, Dict[str, Any]],
    proposer_root: str = None,
) -> Dict[str, Dict[str, Any]]:
    """Keep only cell types whose proposer checkpoint already exists."""

    covered = set(list_trained_proposer_uuts(proposer_root=proposer_root))
    return {
        cell_type: cell_info
        for cell_type, cell_info in cell_pool.items()
        if cell_type in covered
    }
