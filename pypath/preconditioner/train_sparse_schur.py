import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pypath.preconditioner.block_schwarz import BlockPlanConfig, build_block_schwarz_plan
from pypath.preconditioner.schur_interface import (
    ExplicitSchurInterfacePreconditioner,
    SCHUR_EDGE_FEATURE_DIM,
    SchurEdgeGateNet,
    build_schur_edge_features,
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


def build_labels(base, edges, edge_budget: int, eps: float):
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    schur = np.asarray(base.schur_matrix, dtype=np.float64)
    diag = np.abs(np.diag(schur))
    strengths = []
    for i, j in edges.tolist():
        raw = max(abs(float(schur[i, j])), abs(float(schur[j, i])))
        denom = np.sqrt(max(float(diag[i] * diag[j]), eps))
        strengths.append(raw / max(denom, eps))
    strengths = np.asarray(strengths, dtype=np.float64)
    budget = int(edge_budget) if int(edge_budget) > 0 else int(edges.shape[0])
    budget = min(budget, int(edges.shape[0]))
    labels = np.zeros(int(edges.shape[0]), dtype=np.float64)
    if budget > 0:
        labels[np.argsort(-strengths)[:budget]] = 1.0
    return labels


def load_samples(args):
    samples = []
    for circuit_id in _parse_circuit_ids(args.circuit_ids):
        netlist_path = os.path.join(args.netlist_dir, f'{int(circuit_id)}.sp')
        corpus = _load_trajectory_linear_system_steps(
            trajectory_dir=args.trajectory_dir,
            circuit_id=int(circuit_id),
            netlist_path=netlist_path,
        )
        steps = corpus.get('steps', [])[args.step_offset:]
        if args.max_steps_per_circuit > 0:
            steps = steps[: args.max_steps_per_circuit]
        for step in steps:
            matrix = _make_system_matrix(step, apply_gmin_diagonal=not args.disable_gmin_diagonal)
            common = dict(matrix=matrix, node_map=step.get('node_map', {}), netlist_path=netlist_path)
            cfg = dict(
                max_block_size=args.max_block_size,
                min_block_size=args.min_block_size,
                max_blocks=args.max_blocks,
                max_total_block_nnz=args.max_total_block_nnz,
                uncovered_row_policy=args.uncovered_row_policy,
            )
            core_plan = build_block_schwarz_plan(**common, config=BlockPlanConfig(block_mode='cell_core', **cfg))
            boundary_plan = build_block_schwarz_plan(**common, config=BlockPlanConfig(block_mode='cell_core_plus_onehop_boundary', **cfg))
            base = ExplicitSchurInterfacePreconditioner(
                matrix=matrix,
                core_plan=core_plan,
                boundary_plan=boundary_plan,
                uncovered_row_policy=args.uncovered_row_policy,
                factorize_schur=False,
            )
            features, edges, _ = build_schur_edge_features(
                base,
                candidate_edge_limit=args.candidate_edge_limit,
                eps=args.eps,
            )
            if features.shape[0] == 0:
                continue
            labels = build_labels(base, edges, args.edge_budget, args.eps)
            samples.append(
                dict(
                    circuit_id=int(circuit_id),
                    features=features,
                    labels=labels,
                    interface_count=int(base.interface_rows.shape[0]),
                    candidate_edges=int(edges.shape[0]),
                    positive_edges=int(np.count_nonzero(labels > 0.5)),
                )
            )
    return samples


def main():
    parser = argparse.ArgumentParser(description='Train learned sparse Schur edge aggregator.')
    parser.add_argument('--trajectory-dir', required=True)
    parser.add_argument('--netlist-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--circuit-ids', default='0-15')
    parser.add_argument('--step-offset', type=int, default=0)
    parser.add_argument('--max-steps-per-circuit', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--logit-clip', type=float, default=8.0)
    parser.add_argument('--edge-budget', type=int, default=64)
    parser.add_argument('--candidate-edge-limit', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
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
        raise RuntimeError('no sparse Schur training samples')
    all_features = np.concatenate([sample['features'] for sample in samples], axis=0)
    feature_mean = all_features.mean(axis=0)
    feature_std = np.maximum(all_features.std(axis=0), 1e-12)

    model = SchurEdgeGateNet(SCHUR_EDGE_FEATURE_DIM, args.hidden_dim, args.logit_clip)
    model.set_feature_stats(feature_mean, feature_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    log = []

    for epoch in range(int(args.epochs)):
        total_loss = 0.0
        total_acc = 0.0
        count = 0
        for sample in samples:
            features = torch.as_tensor(sample['features'], dtype=torch.float64)
            labels = torch.as_tensor(sample['labels'], dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(features)
            pos = float(torch.count_nonzero(labels > 0.5).item())
            neg = float(labels.numel() - pos)
            pos_weight = torch.as_tensor(max(neg / max(pos, 1.0), 1.0), dtype=torch.float64)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).to(torch.float64)
                acc = torch.mean((pred == labels).to(torch.float64))
            total_loss += float(loss.detach())
            total_acc += float(acc.detach())
            count += 1
        row = {'epoch': epoch + 1, 'mean_bce': total_loss / max(count, 1), 'mean_edge_accuracy': total_acc / max(count, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)

    checkpoint_path = os.path.join(args.output_dir, 'learned_sparse_schur.pt')
    torch.save(
        {
            'model_kind': 'learned_sparse_schur',
            'model_config': {'feature_dim': SCHUR_EDGE_FEATURE_DIM, 'hidden_dim': args.hidden_dim, 'logit_clip': args.logit_clip},
            'model_state_dict': model.state_dict(),
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'training_args': vars(args),
            'training_log': log,
            'sample_summaries': [
                {k: v for k, v in sample.items() if k not in {'features', 'labels'}}
                for sample in samples
            ],
        },
        checkpoint_path,
    )
    summary = {
        'checkpoint': checkpoint_path,
        'sample_count': len(samples),
        'feature_count': int(all_features.shape[0]),
        'final_mean_bce': log[-1]['mean_bce'],
        'final_mean_edge_accuracy': log[-1]['mean_edge_accuracy'],
        'training_log': log,
    }
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=json_default)
    print('checkpoint=' + checkpoint_path)


if __name__ == '__main__':
    main()
