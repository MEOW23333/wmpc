import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pypath.preconditioner.block_schwarz import BlockPlanConfig, build_block_schwarz_plan
from pypath.preconditioner.schur_interface import (
    ExplicitSchurInterfacePreconditioner,
    LOCAL_SCHUR_EDGE_FEATURE_DIM,
    LocalSchurEdgeGateNet,
    aggregate_local_schur_edge_features,
    build_local_schur_edge_features,
)
from pypath.utils.external_gmres_prototype import (
    _load_trajectory_linear_system_steps,
    _make_system_matrix,
    _parse_circuit_ids,
)


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _probe_vectors(rhs: np.ndarray, residual: np.ndarray, count: int, seed: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    rhs = np.asarray(rhs, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if rhs.size:
        out.append(rhs)
    if residual.size and float(np.linalg.norm(residual)) > 0.0:
        out.append(residual)
    rng = np.random.default_rng(int(seed))
    dim = int(rhs.shape[0]) if rhs.size else int(residual.shape[0])
    for _ in range(int(count)):
        out.append(rng.standard_normal(dim).astype(np.float64))
    return out


def _load_case_samples(args_dict: Dict[str, Any], circuit_id: int) -> List[Dict[str, Any]]:
    netlist_dir = args_dict['netlist_dir']
    trajectory_dir = args_dict['trajectory_dir']
    netlist_path = os.path.join(netlist_dir, f'{int(circuit_id)}.sp')
    corpus = _load_trajectory_linear_system_steps(
        trajectory_dir=trajectory_dir,
        circuit_id=int(circuit_id),
        netlist_path=netlist_path,
    )
    steps = corpus.get('steps', [])[int(args_dict['step_offset']):]
    if int(args_dict['max_steps_per_circuit']) > 0:
        steps = steps[: int(args_dict['max_steps_per_circuit'])]
    samples: List[Dict[str, Any]] = []
    cfg = dict(
        max_block_size=int(args_dict['max_block_size']),
        min_block_size=int(args_dict['min_block_size']),
        max_blocks=int(args_dict['max_blocks']),
        max_total_block_nnz=int(args_dict['max_total_block_nnz']),
        uncovered_row_policy=str(args_dict['uncovered_row_policy']),
    )
    for step_id, step in enumerate(steps):
        matrix = _make_system_matrix(step, apply_gmin_diagonal=not bool(args_dict['disable_gmin_diagonal']))
        common = dict(matrix=matrix, node_map=step.get('node_map', {}), netlist_path=netlist_path)
        core_plan = build_block_schwarz_plan(**common, config=BlockPlanConfig(block_mode='cell_core', **cfg))
        boundary_plan = build_block_schwarz_plan(**common, config=BlockPlanConfig(block_mode='cell_core_plus_onehop_boundary', **cfg))
        exact = ExplicitSchurInterfacePreconditioner(
            matrix=matrix,
            core_plan=core_plan,
            boundary_plan=boundary_plan,
            uncovered_row_policy=str(args_dict['uncovered_row_policy']),
            factorize_schur=True,
        )
        local_features, local_edges, local_values, local_block_ids, abb, diag_delta = build_local_schur_edge_features(
            exact,
            candidate_edge_limit=0,
            eps=float(args_dict['eps']),
        )
        features, edges, values, source_counts = aggregate_local_schur_edge_features(
            local_features,
            local_edges,
            local_values,
            local_block_ids,
            interface_count=int(exact.interface_rows.shape[0]),
            candidate_edge_limit=int(args_dict['candidate_edge_limit']),
            eps=float(args_dict['eps']),
        )
        block_ids = np.zeros(int(edges.shape[0]), dtype=np.int64)
        if features.shape[0] == 0 or exact.interface_rows.shape[0] == 0:
            continue
        rhs = np.asarray(step.get('rhs', []), dtype=np.float64)
        residual = np.asarray(step.get('raw_residual', []), dtype=np.float64)
        probes = []
        for probe_id, vec in enumerate(_probe_vectors(rhs, residual, int(args_dict['gaussian_probes']), int(args_dict['seed']) + 1009 * int(circuit_id) + step_id)):
            if vec.shape[0] != matrix.shape[0]:
                continue
            core_solution = exact._apply_core_only(vec)
            interface_rhs = vec[exact.interface_rows] - exact.matrix[np.ix_(exact.interface_rows, exact.core_rows)].dot(core_solution[exact.core_rows])
            target = exact.schur_factor.dot(interface_rhs)
            if target.shape[0] == 0:
                continue
            probes.append({'interface_rhs': interface_rhs, 'target': target, 'probe_id': int(probe_id)})
        if not probes:
            continue
        edge_utility = _edge_teacher_utility(edges, values, probes, float(args_dict['eps']))
        samples.append({
            'circuit_id': int(circuit_id),
            'step_id': int(step_id),
            'features': features,
            'edges': edges,
            'values': values,
            'block_ids': block_ids,
            'source_counts': source_counts,
            'local_candidate_edges': int(local_edges.shape[0]),
            'edge_utility': edge_utility,
            'abb': abb,
            'diag_delta': diag_delta,
            'interface_count': int(exact.interface_rows.shape[0]),
            'candidate_edges': int(edges.shape[0]),
            'matrix_size': int(matrix.shape[0]),
            'probes': probes,
        })
    return samples



def _edge_teacher_utility(edges: np.ndarray, values: np.ndarray, probes: List[Dict[str, Any]], eps: float) -> np.ndarray:
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    target_sq = None
    rhs_sq = None
    for probe in probes:
        target = np.asarray(probe['target'], dtype=np.float64)
        rhs = np.asarray(probe['interface_rhs'], dtype=np.float64)
        if target_sq is None:
            target_sq = np.zeros_like(target, dtype=np.float64)
            rhs_sq = np.zeros_like(rhs, dtype=np.float64)
        target_sq += target * target
        rhs_sq += rhs * rhs
    target_rms = np.sqrt(target_sq / max(len(probes), 1)) if target_sq is not None else np.zeros(0, dtype=np.float64)
    rhs_rms = np.sqrt(rhs_sq / max(len(probes), 1)) if rhs_sq is not None else np.zeros(0, dtype=np.float64)
    utility = np.zeros(int(edges.shape[0]), dtype=np.float64)
    for idx in range(int(edges.shape[0])):
        i = int(edges[idx, 0])
        j = int(edges[idx, 1])
        value = abs(float(values[idx]))
        col_activity = float(target_rms[j]) if j < target_rms.shape[0] else 0.0
        row_activity = float(rhs_rms[i]) if i < rhs_rms.shape[0] else 0.0
        utility[idx] = value * (col_activity + 0.25 * row_activity)
    return utility


def _budget_for_sample(sample: Dict[str, Any], budget_multiplier: float, fallback_edge_budget: int) -> int:
    edge_count = int(sample['candidate_edges'])
    interface_count = int(sample['interface_count'])
    if float(budget_multiplier) > 0.0:
        budget = int(round(float(budget_multiplier) * float(max(interface_count, 0))))
    else:
        budget = int(fallback_edge_budget)
    return max(0, min(int(budget), edge_count))


def _label_scores_for_sample(sample: Dict[str, Any], label_mode: str) -> np.ndarray:
    mode = str(label_mode)
    if mode == 'utility':
        return np.asarray(sample['edge_utility'], dtype=np.float64)
    if mode == 'topk_abs':
        return np.abs(np.asarray(sample['values'], dtype=np.float64))
    if mode == 'topk_abs_source':
        values = np.abs(np.asarray(sample['values'], dtype=np.float64))
        source_counts = np.asarray(sample.get('source_counts', np.ones_like(values)), dtype=np.float64)
        return values * np.log1p(np.maximum(source_counts, 0.0))
    raise ValueError(f'unsupported ranking label mode: {label_mode}')


def load_samples(args) -> List[Dict[str, Any]]:
    circuit_ids = _parse_circuit_ids(args.circuit_ids)
    args_dict = vars(args)
    if int(args.workers) <= 1:
        chunks = [_load_case_samples(args_dict, cid) for cid in circuit_ids]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(_load_case_samples, args_dict, cid) for cid in circuit_ids]
            chunks = [future.result() for future in concurrent.futures.as_completed(futures)]
    samples = [sample for chunk in chunks for sample in chunk]
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    return samples



def _hard_budget_straight_through_gates(
    sample: Dict[str, Any],
    gates: torch.Tensor,
    budget_multiplier: float,
    fallback_edge_budget: int,
) -> torch.Tensor:
    edge_count = int(sample['candidate_edges'])
    if edge_count <= 0:
        return gates
    budget = _budget_for_sample(sample, budget_multiplier, fallback_edge_budget)
    hard = torch.zeros_like(gates)
    if budget > 0:
        values = torch.as_tensor(sample['values'], dtype=torch.float64)
        scores = gates.detach() * torch.abs(values)
        selected = torch.topk(scores, k=budget, largest=True).indices
        hard[selected] = 1.0
    return hard + gates - gates.detach()


def _assemble_soft_schur(sample: Dict[str, Any], gates: torch.Tensor, diagonal_shift: float, eps: float) -> torch.Tensor:
    sparse = torch.as_tensor(sample['abb'] + sample['diag_delta'], dtype=torch.float64).clone()
    edges = torch.as_tensor(sample['edges'], dtype=torch.long)
    values = torch.as_tensor(sample['values'], dtype=torch.float64)
    if edges.numel() > 0:
        contrib = gates * values
        sparse.index_put_((edges[:, 0], edges[:, 1]), contrib, accumulate=True)
    if float(diagonal_shift) > 0.0 and sparse.shape[0] > 0:
        diag = torch.diagonal(sparse).clone()
        offdiag_abs = torch.sum(torch.abs(sparse), dim=1) - torch.abs(diag)
        signs = torch.where(diag >= 0.0, torch.ones_like(diag), -torch.ones_like(diag))
        shifted = diag + signs * float(diagonal_shift) * torch.clamp(offdiag_abs, min=float(eps))
        sparse = sparse.clone()
        idx = torch.arange(sparse.shape[0], dtype=torch.long)
        sparse[idx, idx] = shifted
    return sparse


def main() -> None:
    parser = argparse.ArgumentParser(description='Train per-instance local Schur proposer-aggregator with explicit Schur action teacher.')
    parser.add_argument('--trajectory-dir', required=True)
    parser.add_argument('--netlist-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--circuit-ids', default='0-9,11-16')
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--max-steps-per-circuit', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=96)
    parser.add_argument('--logit-clip', type=float, default=8.0)
    parser.add_argument('--candidate-edge-limit', type=int, default=2048)
    parser.add_argument('--budget-multiplier', type=float, default=2.0)
    parser.add_argument('--fallback-edge-budget', type=int, default=0)
    parser.add_argument('--budget-penalty', type=float, default=0.05)
    parser.add_argument('--teacher-mode', choices=['ranking', 'action'], default='ranking')
    parser.add_argument('--ranking-label-mode', choices=['utility', 'topk_abs', 'topk_abs_source'], default='utility')
    parser.add_argument('--action-loss-weight', type=float, default=0.0)
    parser.add_argument('--soft-budget-training', action='store_true', help='Use dense soft gates instead of hard-budget straight-through gates for action loss.')
    parser.add_argument('--diagonal-shift', type=float, default=1e-8)
    parser.add_argument('--gaussian-probes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-block-size', type=int, default=32)
    parser.add_argument('--min-block-size', type=int, default=2)
    parser.add_argument('--max-blocks', type=int, default=0)
    parser.add_argument('--max-total-block-nnz', type=int, default=0)
    parser.add_argument('--uncovered-row-policy', default='row_sum')
    parser.add_argument('--disable-gmin-diagonal', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-30)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)
    samples = load_samples(args)
    if not samples:
        raise RuntimeError('no local Schur training samples')
    all_features = np.concatenate([sample['features'] for sample in samples], axis=0)
    feature_mean = all_features.mean(axis=0)
    feature_std = np.maximum(all_features.std(axis=0), 1e-12)

    model = LocalSchurEdgeGateNet(LOCAL_SCHUR_EDGE_FEATURE_DIM, int(args.hidden_dim), float(args.logit_clip))
    model.set_feature_stats(feature_mean, feature_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    log = []

    for epoch in range(int(args.epochs)):
        total_loss = 0.0
        total_action = 0.0
        total_budget = 0.0
        count = 0
        for sample in samples:
            features = torch.as_tensor(sample['features'], dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            gates = model(features)
            logits = model.logits(features)
            budget = _budget_for_sample(sample, float(args.budget_multiplier), int(args.fallback_edge_budget))
            label_scores = torch.as_tensor(_label_scores_for_sample(sample, str(args.ranking_label_mode)), dtype=torch.float64)
            labels = torch.zeros_like(gates)
            if budget > 0 and label_scores.numel() > 0:
                selected = torch.topk(label_scores, k=min(int(budget), int(label_scores.numel())), largest=True).indices
                labels[selected] = 1.0
            positive = torch.sum(labels).clamp_min(1.0)
            negative = torch.sum(1.0 - labels).clamp_min(1.0)
            pos_weight = (negative / positive).detach()
            ranking_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            action_loss = torch.zeros((), dtype=torch.float64)
            if str(args.teacher_mode) == 'action' or float(args.action_loss_weight) > 0.0:
                action_gates = gates if bool(args.soft_budget_training) else _hard_budget_straight_through_gates(
                    sample,
                    gates,
                    float(args.budget_multiplier),
                    int(args.fallback_edge_budget),
                )
                sparse = _assemble_soft_schur(sample, action_gates, float(args.diagonal_shift), float(args.eps))
                probe_count = 0
                pinv = torch.linalg.pinv(sparse)
                for probe in sample['probes']:
                    rhs = torch.as_tensor(probe['interface_rhs'], dtype=torch.float64)
                    target = torch.as_tensor(probe['target'], dtype=torch.float64)
                    pred = pinv.matmul(rhs)
                    denom = torch.sum(target * target).clamp_min(float(args.eps))
                    action_loss = action_loss + torch.sum((pred - target) ** 2) / denom
                    probe_count += 1
                action_loss = action_loss / max(probe_count, 1)
            edge_count = int(sample['candidate_edges'])
            target_fraction = min(float(max(budget, 0)) / float(max(edge_count, 1)), 1.0)
            budget_loss = torch.relu(torch.mean(gates) - torch.as_tensor(target_fraction, dtype=torch.float64)) ** 2
            if str(args.teacher_mode) == 'action':
                loss = action_loss + float(args.budget_penalty) * budget_loss
            else:
                loss = ranking_loss + float(args.action_loss_weight) * action_loss + float(args.budget_penalty) * budget_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_action += float(action_loss.detach())
            total_budget += float(budget_loss.detach())
            count += 1
        row = {
            'epoch': int(epoch + 1),
            'mean_loss': total_loss / max(count, 1),
            'mean_action_loss': total_action / max(count, 1),
            'mean_budget_loss': total_budget / max(count, 1),
        }
        log.append(row)
        print(json.dumps(row), flush=True)

    checkpoint_path = os.path.join(args.output_dir, 'learned_local_sparse_schur.pt')
    torch.save(
        {
            'model_kind': 'learned_local_sparse_schur',
            'model_config': {
                'feature_dim': LOCAL_SCHUR_EDGE_FEATURE_DIM,
                'hidden_dim': int(args.hidden_dim),
                'logit_clip': float(args.logit_clip),
            },
            'model_state_dict': model.state_dict(),
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'training_args': vars(args),
            'training_log': log,
            'sample_summaries': [
                {
                    'circuit_id': sample['circuit_id'],
                    'step_id': sample['step_id'],
                    'matrix_size': sample['matrix_size'],
                    'interface_count': sample['interface_count'],
                    'candidate_edges': sample['candidate_edges'],
                    'probe_count': len(sample['probes']),
                }
                for sample in samples
            ],
        },
        checkpoint_path,
    )
    summary = {
        'checkpoint': checkpoint_path,
        'sample_count': len(samples),
        'feature_count': int(all_features.shape[0]),
        'final_mean_loss': log[-1]['mean_loss'],
        'final_mean_action_loss': log[-1]['mean_action_loss'],
        'final_mean_budget_loss': log[-1]['mean_budget_loss'],
        'training_log': log,
    }
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, default=json_default)
    print('checkpoint=' + checkpoint_path)


if __name__ == '__main__':
    main()
