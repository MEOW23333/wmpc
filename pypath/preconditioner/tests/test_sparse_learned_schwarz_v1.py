import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from pypath.preconditioner.block_schwarz import BlockSchwarzPlan
from pypath.preconditioner.linear_system_contract import (
    CHECKPOINT_SCHEMA_VERSION,
    EFFECTIVE_MATRIX_CONTRACT,
    FEATURE_CONTRACT,
    INITIAL_RESIDUAL_FORMULA,
    compute_initial_residual,
    resolve_initial_guess,
)
from pypath.preconditioner.learned_schwarz import (
    DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
    LEARNED_SCHWARZ_PARAMETER_MODES,
    LearnedSchwarzPreconditioner,
    build_learned_schwarz_sample,
)
from pypath.preconditioner.sparse_learned_schwarz import (
    SparseLearnedSchwarzV1Preconditioner,
    build_sparse_learned_schwarz_sample,
    load_learned_schwarz_v1_model,
)
from pypath.utils.export_native_learned_schwarz_sidecar import (
    build_native_learned_schwarz_sidecar_payload,
    compute_float64_vector_sha256,
    compute_layout_sha256,
    compute_matrix_fingerprint,
    export_sidecar_from_paths,
    validate_native_learned_schwarz_sidecar,
)
import pypath.utils.export_native_learned_schwarz_sidecar as native_sidecar
import pypath.preconditioner.sparse_learned_schwarz as sparse_learned_schwarz
import pypath.preconditioner.diagnose_boundary_correction as boundary_diagnostics
import pypath.preconditioner.train_learned_schwarz as learned_schwarz_train
import pypath.utils.external_gmres_prototype as external_gmres
import pypath.utils.sparse_gmres_prototype as sparse_gmres


class SparseLearnedSchwarzV1Test(unittest.TestCase):
    @staticmethod
    def _fixture():
        matrix = np.asarray([
            [4.0, -0.5, 0.1, 0.0, 0.0, 0.0],
            [-0.2, 3.5, -0.4, 0.0, 0.0, 0.0],
            [0.1, -0.3, 4.2, -0.6, 0.0, 0.0],
            [0.0, 0.0, -0.5, 3.8, -0.7, 0.0],
            [0.0, 0.0, 0.0, -0.4, 3.2, -0.2],
            [0.0, 0.0, 0.0, 0.0, -0.1, 2.5],
        ], dtype=np.float64)
        blocks = [
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([2, 3, 4], dtype=np.int64),
        ]
        candidates = [
            {"row_role_by_index": {"1": "internal_node", "2": "internal_node", "3": "external_pin"}},
            {"row_role_by_index": {"3": "external_pin", "4": "internal_node", "5": "unknown"}},
        ]
        rhs = np.linspace(0.5, 1.0, matrix.shape[0])
        residual = np.linspace(-0.2, 0.3, matrix.shape[0])
        covered = np.zeros(matrix.shape[0], dtype=bool)
        for rows in blocks:
            covered[rows] = True
        plan = BlockSchwarzPlan(
            block_mode="cell_core_plus_onehop_boundary",
            blocks=blocks,
            block_candidates=candidates,
            covered_mask=covered,
            uncovered_scales=1.0 / np.maximum(np.abs(matrix).sum(axis=1), 1e-30),
            factor_solvers=[],
            factor_modes=[],
            total_block_nnz=0,
            max_block_size=3,
            candidate_block_count=2,
            skipped_block_count=0,
            coverage_ratio=float(np.mean(covered)),
        )
        return matrix, blocks, candidates, rhs, residual, plan

    @staticmethod
    def _model():
        torch.manual_seed(7)
        model = LearnedSchwarzPreconditioner(hidden_dim=8).to(dtype=torch.float64)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.04, 0.04)
        return model.eval()

    def test_sparse_output_matches_dense_v1(self):
        matrix, blocks, candidates, rhs, residual, plan = self._fixture()
        model = self._model()
        dense_sample = build_learned_schwarz_sample(
            matrix=matrix,
            plan=plan,
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
            dtype=torch.float64,
        )
        sparse_sample, _ = build_sparse_learned_schwarz_sample(
            matrix=sp.csr_matrix(matrix),
            blocks=blocks,
            block_candidates=candidates,
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
        )
        torch.testing.assert_close(sparse_sample.block_features, dense_sample.block_features)
        torch.testing.assert_close(
            sparse_sample.neighbor_block_features,
            dense_sample.neighbor_block_features,
        )
        torch.testing.assert_close(
            sparse_sample.lambda_floors,
            dense_sample.lambda_floors,
        )
        for actual, expected in zip(sparse_sample.row_features, dense_sample.row_features):
            torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            dense_parameters = model.predict_parameters(dense_sample)
            sparse_parameters = sparse_learned_schwarz._predict_parameters(
                model,
                sparse_sample,
            )
        for name in ("lambda_pred", "lambda_floor", "lambdas"):
            torch.testing.assert_close(
                sparse_parameters[name],
                dense_parameters[name],
                rtol=0.0,
                atol=0.0,
            )

        deployed = SparseLearnedSchwarzV1Preconditioner(
            matrix=sp.csr_matrix(matrix),
            blocks=blocks,
            block_candidates=candidates,
            model=model,
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
        )
        vector = np.linspace(-1.0, 0.7, matrix.shape[0])
        with torch.no_grad():
            expected = model.apply(
                dense_sample,
                torch.as_tensor(vector, dtype=torch.float64),
            ).cpu().numpy()
        np.testing.assert_allclose(
            deployed.apply(vector),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            deployed.lambda_pred,
            dense_parameters["lambda_pred"].detach().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            deployed.lambda_floors,
            dense_parameters["lambda_floor"].detach().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            deployed.lambdas,
            dense_parameters["lambdas"].detach().cpu().numpy(),
        )
        metadata = deployed.metadata()
        self.assertTrue(metadata["all_candidate_blocks_factorized"])
        self.assertEqual(metadata["skipped_block_count"], 0)
        self.assertEqual(
            metadata["block_lambda_effective"],
            deployed.lambdas.tolist(),
        )

    def test_parameter_modes_match_dense_sparse_and_are_auditable(self):
        matrix, blocks, candidates, rhs, residual, plan = self._fixture()
        model = self._model()
        model.initialize_lambda_prediction(1e-3)
        dense_sample = build_learned_schwarz_sample(
            matrix=matrix,
            plan=plan,
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
            dtype=torch.float64,
        )
        sparse_sample, _ = build_sparse_learned_schwarz_sample(
            matrix=sp.csr_matrix(matrix),
            blocks=blocks,
            block_candidates=candidates,
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
        )
        vector = np.linspace(-1.0, 0.7, matrix.shape[0])
        expected_sources = {
            "fixed_same_blocks": ("lambda_floor", "uniform_row_coverage"),
            "learned_shift_only": (
                "learned_effective",
                "uniform_row_coverage",
            ),
            "learned_overlap_weights_only": (
                "lambda_floor",
                "learned_overlap",
            ),
            "learned_full": ("learned_effective", "learned_overlap"),
        }
        expected_uniform_weights = (
            np.asarray([1.0, 1.0, 0.5]),
            np.asarray([0.5, 1.0, 1.0]),
        )

        for parameter_mode in sorted(LEARNED_SCHWARZ_PARAMETER_MODES):
            with torch.no_grad():
                dense_parameters = model.predict_parameters(
                    dense_sample,
                    parameter_mode=parameter_mode,
                )
                sparse_parameters = sparse_learned_schwarz._predict_parameters(
                    model,
                    sparse_sample,
                    parameter_mode=parameter_mode,
                )
                expected = model.apply(
                    dense_sample,
                    torch.as_tensor(vector, dtype=torch.float64),
                    parameter_mode=parameter_mode,
                ).cpu().numpy()
            for name in (
                "lambda_pred",
                "lambda_floor",
                "learned_lambdas",
                "lambdas",
            ):
                torch.testing.assert_close(
                    sparse_parameters[name],
                    dense_parameters[name],
                    rtol=0.0,
                    atol=0.0,
                )
            for weight_name in ("learned_weights", "weights"):
                for sparse_weights, dense_weights in zip(
                    sparse_parameters[weight_name],
                    dense_parameters[weight_name],
                ):
                    torch.testing.assert_close(
                        sparse_weights,
                        dense_weights,
                        rtol=0.0,
                        atol=0.0,
                    )

            deployed = SparseLearnedSchwarzV1Preconditioner(
                matrix=sp.csr_matrix(matrix),
                blocks=blocks,
                block_candidates=candidates,
                model=model,
                linear_rhs=rhs,
                initial_residual=residual,
                gmin=1e-12,
                parameter_mode=parameter_mode,
            )
            np.testing.assert_allclose(
                deployed.apply(vector),
                expected,
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_array_equal(
                deployed.lambda_pred,
                dense_parameters["lambda_pred"].detach().cpu().numpy(),
            )
            np.testing.assert_array_equal(
                deployed.lambda_floors,
                dense_parameters["lambda_floor"].detach().cpu().numpy(),
            )
            np.testing.assert_array_equal(
                deployed.learned_lambdas,
                dense_parameters["learned_lambdas"].detach().cpu().numpy(),
            )
            np.testing.assert_array_equal(
                deployed.lambdas,
                dense_parameters["lambdas"].detach().cpu().numpy(),
            )
            for actual, expected_weights in zip(
                deployed.learned_weights,
                dense_parameters["learned_weights"],
            ):
                np.testing.assert_array_equal(
                    actual,
                    expected_weights.detach().cpu().numpy(),
                )
            for actual, expected_weights in zip(
                deployed.weights,
                dense_parameters["weights"],
            ):
                np.testing.assert_array_equal(
                    actual,
                    expected_weights.detach().cpu().numpy(),
                )

            metadata = deployed.metadata()
            self.assertEqual(metadata["parameter_mode"], parameter_mode)
            self.assertEqual(
                (metadata["shift_source"], metadata["weight_source"]),
                expected_sources[parameter_mode],
            )
            exported = deployed.export_parameters()
            self.assertEqual(exported["parameter_mode"], parameter_mode)
            self.assertEqual(
                (exported["shift_source"], exported["weight_source"]),
                expected_sources[parameter_mode],
            )
            if parameter_mode in {
                "fixed_same_blocks",
                "learned_overlap_weights_only",
            }:
                np.testing.assert_array_equal(
                    deployed.lambdas,
                    deployed.lambda_floors,
                )
            else:
                np.testing.assert_array_equal(
                    deployed.lambdas,
                    deployed.learned_lambdas,
                )
            if parameter_mode in {
                "fixed_same_blocks",
                "learned_shift_only",
            }:
                for actual, expected_weights in zip(
                    deployed.weights,
                    expected_uniform_weights,
                ):
                    np.testing.assert_array_equal(actual, expected_weights)
            else:
                for actual, expected_weights in zip(
                    deployed.weights,
                    deployed.learned_weights,
                ):
                    np.testing.assert_array_equal(actual, expected_weights)

        with torch.no_grad():
            default_parameters = model.predict_parameters(dense_sample)
        self.assertEqual(
            default_parameters["parameter_mode"],
            DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
        )
        self.assertEqual(
            DEFAULT_LEARNED_SCHWARZ_PARAMETER_MODE,
            "learned_full",
        )

    def test_factorization_failure_does_not_drop_candidate_block(self):
        matrix = sp.csr_matrix(
            np.asarray(
                [
                    [4.0, 0.0],
                    [0.0, -1.0],
                ],
                dtype=np.float64,
            )
        )
        blocks = [
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
        ]
        call_sizes = []

        def predict_with_unfactorizable_block(
            _model,
            sample,
            *,
            parameter_mode="learned_full",
        ):
            del parameter_mode
            call_sizes.append(len(sample.blocks))
            lambda_pred = torch.as_tensor([0.0, 1.0], dtype=torch.float64)
            lambda_floor = sample.lambda_floors.clone()
            return {
                "lambda_pred": lambda_pred,
                "lambda_floor": lambda_floor,
                "lambdas": torch.maximum(lambda_pred, lambda_floor),
                "weights": [
                    torch.ones(rows.numel(), dtype=torch.float64)
                    for rows in sample.blocks
                ],
            }

        with mock.patch.object(
            sparse_learned_schwarz,
            "_predict_parameters",
            side_effect=predict_with_unfactorizable_block,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "requires every candidate block to factorize",
            ):
                SparseLearnedSchwarzV1Preconditioner(
                    matrix=matrix,
                    blocks=blocks,
                    block_candidates=[{}, {}],
                    model=self._model(),
                    linear_rhs=np.zeros(2, dtype=np.float64),
                    initial_residual=np.zeros(2, dtype=np.float64),
                )

        self.assertEqual(call_sizes, [2])
    def test_generic_block_roles_match_training_contract(self):
        cache = {
            "cell_a": {
                "uut_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "uut_local_node_names": (
                    "internal.node",
                    "pin_shared",
                    "pin_private",
                ),
                "external_pin_to_global_node": {
                    "pin_shared": "shared_net",
                    "pin_private": "private_net",
                },
                "cell_type": "A",
            },
            "cell_b": {
                "uut_indices": np.asarray([3, 4], dtype=np.int64),
                "uut_local_node_names": (
                    "internal_b.node",
                    "pin_shared_b",
                ),
                "external_pin_to_global_node": {
                    "pin_shared_b": "shared_net",
                },
                "cell_type": "B",
            },
        }
        with mock.patch(
            "pypath.aggregator.preprocess_coupler.parse_netlist_for_instances",
            return_value={},
        ), mock.patch(
            "pypath.aggregator.preprocess_coupler.build_instance_feature_cache",
            return_value=cache,
        ):
            _, candidates, _ = sparse_learned_schwarz._extract_generic_blocks(
                node_map={},
                netlist_path="fixture.sp",
                min_block_size=2,
                max_block_size=8,
                max_blocks=0,
            )

        roles = candidates[0]["row_role_by_index"]
        self.assertEqual(roles["1"], "internal_node")
        self.assertEqual(roles["2"], "external_pin")
        self.assertEqual(roles["3"], "unknown")

    def test_global_matrix_is_never_dense(self):

        matrix = sp.eye(200, format="csr", dtype=np.float64) * 3.0
        model = self._model()
        original_toarray = sp.csr_matrix.toarray
        shapes = []

        def guarded_toarray(sparse_matrix, *args, **kwargs):
            shapes.append(tuple(sparse_matrix.shape))
            if sparse_matrix.shape == matrix.shape:
                raise AssertionError("global sparse matrix was materialized")
            return original_toarray(sparse_matrix, *args, **kwargs)

        with mock.patch.object(sp.csr_matrix, "toarray", new=guarded_toarray):
            deployed = SparseLearnedSchwarzV1Preconditioner(
                matrix=matrix,
                blocks=[np.asarray([0, 1, 2]), np.asarray([2, 3, 4])],
                block_candidates=[{}, {}],
                model=model,
                linear_rhs=np.ones(200),
                initial_residual=np.ones(200),
            )
            output = deployed.apply(np.ones(200))

        self.assertEqual(output.shape, (200,))
        self.assertTrue(shapes)
        self.assertNotIn(matrix.shape, shapes)
        self.assertTrue(deployed.metadata()["no_global_dense_materialization"])

    def test_memory_saving_exceeds_half(self):
        size = 4096
        matrix = sp.diags(
            [-np.ones(size - 1), 4.0 * np.ones(size), -np.ones(size - 1)],
            [-1, 0, 1],
            format="csr",
        )
        blocks = [
            np.arange(start, min(start + 8, size), dtype=np.int64)
            for start in range(0, size, 8)
        ]
        deployed = SparseLearnedSchwarzV1Preconditioner(
            matrix=matrix,
            blocks=blocks,
            block_candidates=[{} for _ in blocks],
            model=self._model(),
            linear_rhs=np.ones(size),
            initial_residual=np.ones(size),
        )
        metadata = deployed.metadata()
        self.assertGreater(metadata["estimated_peak_memory_saving_ratio"], 0.50)
        self.assertTrue(metadata["memory_saving_target_over_50pct"])

    def test_float64_vector_sha256_contract(self):
        vector = np.asarray([0.0, -0.0, 1.5, -2.25], dtype=np.float64)
        self.assertEqual(
            compute_float64_vector_sha256(vector),
            "479680c217061340d9a7ceee2f0e4710e3fdc35da5bd504159a5db479e811538",
        )
        self.assertEqual(
            compute_float64_vector_sha256(vector.astype(">f8")),
            compute_float64_vector_sha256(vector),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compute_float64_vector_sha256([1.0, float("inf")])

    def test_native_sidecar_v5_flattening_and_validation(self):
        matrix, blocks, candidates, rhs, residual, _ = self._fixture()
        deployed = SparseLearnedSchwarzV1Preconditioner(
            matrix=sp.csr_matrix(matrix),
            blocks=blocks,
            block_candidates=candidates,
            model=self._model(),
            linear_rhs=rhs,
            initial_residual=residual,
            gmin=1e-12,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            checkpoint.write(b"fixture-checkpoint")
            checkpoint.flush()
            initial_guess = np.zeros(matrix.shape[0], dtype=np.float64)
            initial_residual = rhs.copy()
            payload = build_native_learned_schwarz_sidecar_payload(
                preconditioner=deployed,
                effective_matrix=sp.csr_matrix(matrix),
                node_map={f"n{index}": index for index in range(1, 7)},
                checkpoint_path=checkpoint.name,
                time_value=0.0,
                gmin=1e-12,
                newton_iter=3,
                linear_rhs=rhs,
                initial_guess=initial_guess,
                initial_residual=initial_residual,
                initial_guess_mode="zero",
            )

        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["local_shift_contract"], "block_inf_norm_relative_floor_v1")
        self.assertEqual(payload["local_shift_floor_relative"], 1e-6)
        self.assertEqual(
            payload["feature_contract"],
            "learned_schwarz_v1_abs_rhs_abs_initial_residual",
        )
        self.assertEqual(payload["initial_guess_mode"], "zero")
        self.assertEqual(payload["linear_rhs_sha256"], compute_float64_vector_sha256(rhs))
        self.assertEqual(payload["initial_guess_sha256"], compute_float64_vector_sha256(initial_guess))
        self.assertEqual(
            payload["initial_residual_contract"],
            native_sidecar.INITIAL_RESIDUAL_CONTRACT,
        )
        self.assertAlmostEqual(
            payload["initial_residual_norm_l2"],
            float(np.linalg.norm(initial_residual)),
            places=15,
        )
        self.assertEqual(
            payload["initial_residual_norm_atol"],
            native_sidecar.INITIAL_RESIDUAL_NORM_ATOL,
        )
        self.assertEqual(
            payload["initial_residual_norm_rtol"],
            native_sidecar.INITIAL_RESIDUAL_NORM_RTOL,
        )
        self.assertNotIn("initial_residual_sha256", payload)
        self.assertEqual(payload["row_index_base"], 1)
        self.assertEqual(payload["block_offsets"], [0, 3, 6])
        self.assertEqual(payload["block_rows"], [1, 2, 3, 3, 4, 5])
        self.assertEqual(payload["block_count"], 2)
        self.assertEqual(payload["total_block_rows"], 6)
        self.assertEqual(
            payload["layout_sha256"],
            compute_layout_sha256(
                matrix_size=6,
                block_count=2,
                total_block_rows=6,
                block_offsets=[0, 3, 6],
                block_rows=[1, 2, 3, 3, 4, 5],
            ),
        )
        overlap_weights = [
            weight
            for row, weight in zip(
                payload["block_rows"],
                payload["block_row_weights"],
            )
            if row == 3
        ]
        self.assertAlmostEqual(sum(overlap_weights), 1.0, places=12)
        validate_native_learned_schwarz_sidecar(payload)

        missing_digest = copy.deepcopy(payload)
        missing_digest.pop("linear_rhs_sha256")
        with self.assertRaisesRegex(ValueError, "linear_rhs_sha256"):
            validate_native_learned_schwarz_sidecar(missing_digest)

        missing_residual_contract = copy.deepcopy(payload)
        missing_residual_contract.pop("initial_residual_contract")
        with self.assertRaisesRegex(ValueError, "initial_residual_contract"):
            validate_native_learned_schwarz_sidecar(missing_residual_contract)

        malformed_residual_norm = copy.deepcopy(payload)
        malformed_residual_norm["initial_residual_norm_l2"] = -1.0
        with self.assertRaisesRegex(ValueError, "initial_residual_norm_l2"):
            validate_native_learned_schwarz_sidecar(malformed_residual_norm)

        bad_residual_norm_atol = copy.deepcopy(payload)
        bad_residual_norm_atol["initial_residual_norm_atol"] = 1e-9
        with self.assertRaisesRegex(ValueError, "initial_residual_norm_atol"):
            validate_native_learned_schwarz_sidecar(bad_residual_norm_atol)

        legacy_schema_v4 = copy.deepcopy(payload)
        legacy_schema_v4["schema_version"] = 4
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_native_learned_schwarz_sidecar(legacy_schema_v4)

        wrong_zero_digest = copy.deepcopy(payload)
        wrong_zero_digest["initial_guess_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match zero initial guess"):
            validate_native_learned_schwarz_sidecar(wrong_zero_digest)

        missing_feature_contract = copy.deepcopy(payload)
        missing_feature_contract.pop("feature_contract")
        with self.assertRaisesRegex(ValueError, "feature_contract"):
            validate_native_learned_schwarz_sidecar(missing_feature_contract)

        bad_iteration = copy.deepcopy(payload)
        bad_iteration["newton_iter"] = 0
        with self.assertRaisesRegex(ValueError, "at least 1"):
            validate_native_learned_schwarz_sidecar(bad_iteration)

        fingerprint_fixture = sp.coo_matrix(
            (
                [1.25, -1.25, 0.0, 2.5, -3.0],
                ([0, 0, 0, 1, 2], [0, 0, 2, 1, 0]),
            ),
            shape=(3, 3),
            dtype=np.float64,
        )
        canonical_fixture = sp.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.5, 0.0],
                    [-3.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )
        expected_fingerprint = (
            "7e79e46d775a2d5641ae98ab0306d96f"
            "2888fbb75b58bc2f03e62d9d16304eca"
        )
        self.assertEqual(
            compute_matrix_fingerprint(fingerprint_fixture),
            expected_fingerprint,
        )
        self.assertEqual(
            compute_matrix_fingerprint(canonical_fixture),
            expected_fingerprint,
        )

        bad_weights = copy.deepcopy(payload)
        first_overlap = bad_weights["block_rows"].index(3)
        bad_weights["block_row_weights"][first_overlap] += 0.05
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_native_learned_schwarz_sidecar(bad_weights)

        bad_row = copy.deepcopy(payload)
        bad_row["block_rows"][0] = 0
        bad_row["layout_sha256"] = compute_layout_sha256(
            matrix_size=bad_row["matrix_size"],
            block_count=bad_row["block_count"],
            total_block_rows=bad_row["total_block_rows"],
            block_offsets=bad_row["block_offsets"],
            block_rows=bad_row["block_rows"],
        )
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            validate_native_learned_schwarz_sidecar(bad_row)

        bad_lambda = copy.deepcopy(payload)
        bad_lambda["block_lambdas"][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_native_learned_schwarz_sidecar(bad_lambda)

    def test_filename_iteration_maps_to_native_post_increment_contract(self):
        time_value, gmin, newton_iter = native_sidecar._resolve_step_metadata(
            step_path=(
                "circuit_0_time_1.500000e+00_gmin_2.500000e-01_"
                "iter_000.txt"
            ),
            time_value=None,
            gmin=None,
            newton_iter=None,
        )
        self.assertEqual(time_value, 1.5)
        self.assertEqual(gmin, 0.25)
        self.assertEqual(newton_iter, 1)

        with self.assertRaisesRegex(
            ValueError,
            "continuation-local step index",
        ):
            native_sidecar._resolve_step_metadata(
                step_path=(
                    "continuation_circuit_0_time_1.500000e+00_"
                    "gmin_2.500000e-01_iter_000.txt"
                ),
                time_value=None,
                gmin=None,
                newton_iter=None,
            )

        _, _, explicit_iter = native_sidecar._resolve_step_metadata(
            step_path=(
                "continuation_circuit_0_time_1.500000e+00_"
                "gmin_2.500000e-01_iter_000.txt"
            ),
            time_value=None,
            gmin=None,
            newton_iter=7,
        )
        self.assertEqual(explicit_iter, 7)

    def test_sidecar_path_export_recomputes_initial_residual_for_both_x0_modes(self):
        linear_rhs = np.asarray([1.0, 2.0], dtype=np.float64)
        rhsold = np.asarray([0.2, -0.3], dtype=np.float64)
        legacy_residual = np.asarray([0.1, -0.2], dtype=np.float64)
        raw_matrix = sp.csr_matrix(
            np.asarray(
                [
                    [4.0, -1.0],
                    [-1.0, 3.0],
                ],
                dtype=np.float64,
            )
        )
        gmin = 0.25
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jacobian_path = root / "fixture_jac.txt"
            step_path = (
                root
                / "circuit_0_time_1.500000e+00_gmin_2.500000e-01_iter_4.txt"
            )
            netlist_path = root / "0.sp"
            checkpoint_path = root / "model.pt"
            output_path = root / "sidecar.json"
            zero_output_path = root / "sidecar_zero.json"
            jacobian_path.write_text("fixture\n", encoding="utf-8")
            step_path.write_text("fixture\n", encoding="utf-8")
            netlist_path.write_text(".end\n", encoding="utf-8")
            checkpoint_path.write_bytes(b"fixture-checkpoint")

            captured_preconditioner_inputs = []

            def build_preconditioner(**kwargs):
                captured_preconditioner_inputs.append(kwargs)
                return SparseLearnedSchwarzV1Preconditioner(**kwargs)

            with mock.patch.object(
                native_sidecar,
                "read_J_sparse",
                return_value=raw_matrix,
            ), mock.patch.object(
                native_sidecar,
                "read_continuation_step",
                return_value={
                    "rhsold": rhsold.tolist(),
                    "rhsnew": linear_rhs.tolist(),
                    "residual": legacy_residual.tolist(),
                    "node_map": {"n1": 1, "n2": 2},
                },
            ), mock.patch.object(
                native_sidecar,
                "extract_learned_schwarz_blocks",
                return_value=(
                    [np.asarray([0, 1], dtype=np.int64)],
                    [{}],
                    {},
                ),
            ), mock.patch.object(
                native_sidecar,
                "load_learned_schwarz_v1_model",
                return_value=self._model(),
            ), mock.patch.object(
                native_sidecar,
                "SparseLearnedSchwarzV1Preconditioner",
                side_effect=build_preconditioner,
            ):
                payload = export_sidecar_from_paths(
                    jacobian_path=str(jacobian_path),
                    continuation_step=str(step_path),
                    netlist_path=str(netlist_path),
                    checkpoint_path=str(checkpoint_path),
                    output_path=str(output_path),
                )
                zero_payload = export_sidecar_from_paths(
                    jacobian_path=str(jacobian_path),
                    continuation_step=str(step_path),
                    netlist_path=str(netlist_path),
                    checkpoint_path=str(checkpoint_path),
                    output_path=str(zero_output_path),
                    initial_guess_mode="zero",
                )


            effective_matrix = raw_matrix + sp.eye(2, format="csr") * gmin
            expected_rhsold_r0 = linear_rhs - effective_matrix @ rhsold
            self.assertFalse(np.array_equal(legacy_residual, expected_rhsold_r0))
            self.assertEqual(len(captured_preconditioner_inputs), 2)
            np.testing.assert_array_equal(
                captured_preconditioner_inputs[0]["linear_rhs"],
                linear_rhs,
            )
            np.testing.assert_array_equal(
                captured_preconditioner_inputs[0]["initial_residual"],
                expected_rhsold_r0,
            )
            np.testing.assert_array_equal(
                captured_preconditioner_inputs[1]["initial_residual"],
                linear_rhs,
            )
            self.assertEqual(payload["initial_guess_mode"], "rhsold")
            self.assertEqual(
                payload["linear_rhs_sha256"],
                compute_float64_vector_sha256(linear_rhs),
            )
            self.assertEqual(
                payload["initial_guess_sha256"],
                compute_float64_vector_sha256(rhsold),
            )
            self.assertEqual(
                payload["initial_residual_contract"],
                native_sidecar.INITIAL_RESIDUAL_CONTRACT,
            )
            self.assertAlmostEqual(
                payload["initial_residual_norm_l2"],
                float(np.linalg.norm(expected_rhsold_r0)),
                places=15,
            )
            self.assertNotAlmostEqual(
                payload["initial_residual_norm_l2"],
                float(np.linalg.norm(legacy_residual)),
                places=12,
            )
            self.assertEqual(zero_payload["initial_guess_mode"], "zero")
            self.assertEqual(
                zero_payload["initial_guess_sha256"],
                compute_float64_vector_sha256(np.zeros(2, dtype=np.float64)),
            )
            self.assertEqual(
                zero_payload["initial_residual_contract"],
                native_sidecar.INITIAL_RESIDUAL_CONTRACT,
            )
            self.assertAlmostEqual(
                zero_payload["initial_residual_norm_l2"],
                float(np.linalg.norm(linear_rhs)),
                places=15,
            )
            self.assertEqual(payload["gmin"], gmin)
            self.assertEqual(payload["time"], 1.5)
            self.assertEqual(payload["newton_iter"], 5)
            self.assertEqual(
                payload["matrix_fingerprint"],
                compute_matrix_fingerprint(effective_matrix),
            )
            with output_path.open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), payload)

    def test_initial_residual_contract_uses_effective_matrix_and_modes(self):
        jacobian = np.asarray(
            [[4.0, -1.0], [-1.0, 3.0]],
            dtype=np.float64,
        )
        gmin = 0.25
        effective_matrix = jacobian + gmin * np.eye(2, dtype=np.float64)
        linear_rhs = np.asarray([1.0, 2.0], dtype=np.float64)
        rhsold = np.asarray([0.2, -0.3], dtype=np.float64)

        rhsold_guess = resolve_initial_guess(
            rhsold=rhsold,
            matrix_size=2,
            initial_guess_mode="rhsold",
        )
        initial_residual = compute_initial_residual(
            effective_matrix=effective_matrix,
            linear_rhs=linear_rhs,
            initial_guess=rhsold_guess,
        )
        np.testing.assert_allclose(
            initial_residual,
            linear_rhs - effective_matrix @ rhsold,
        )
        self.assertFalse(
            np.allclose(initial_residual, jacobian @ rhsold - linear_rhs)
        )

        zero_guess = resolve_initial_guess(
            rhsold=rhsold,
            matrix_size=2,
            initial_guess_mode="zero",
        )
        np.testing.assert_array_equal(zero_guess, np.zeros(2, dtype=np.float64))
        np.testing.assert_array_equal(
            compute_initial_residual(
                effective_matrix=effective_matrix,
                linear_rhs=linear_rhs,
                initial_guess=zero_guess,
            ),
            linear_rhs,
        )


    def test_checkpoint_contract_rejects_wrong_dimensions(self):
        with self.assertRaisesRegex(ValueError, "learned-schwarz-checkpoint"):
            load_learned_schwarz_v1_model("")
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "feature_contract": FEATURE_CONTRACT,
                    "effective_matrix_contract": EFFECTIVE_MATRIX_CONTRACT,
                    "initial_residual_formula": INITIAL_RESIDUAL_FORMULA,
                    "initial_guess_mode": "rhsold",
                    "local_shift_contract": "block_inf_norm_relative_floor_v1",
                    "local_shift_floor_relative": 1e-6,
                    "model_kind": "learned_schwarz_v1",
                    "model_args": {"block_feature_dim": 8, "row_feature_dim": 14},
                    "model_state_dict": {},
                },
                handle.name,
            )
            with self.assertRaisesRegex(ValueError, "feature dimensions"):
                load_learned_schwarz_v1_model(handle.name)

        with tempfile.NamedTemporaryFile(suffix=".pt") as legacy:
            torch.save(
                {
                    "schema_version": 3,
                    "model_kind": "learned_schwarz_v1",
                    "model_args": {},
                    "model_state_dict": {},
                },
                legacy.name,
            )
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_learned_schwarz_v1_model(legacy.name)

        with tempfile.NamedTemporaryFile(suffix=".pt") as mode_mismatch:
            torch.save(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "feature_contract": FEATURE_CONTRACT,
                    "effective_matrix_contract": EFFECTIVE_MATRIX_CONTRACT,
                    "initial_residual_formula": INITIAL_RESIDUAL_FORMULA,
                    "initial_guess_mode": "rhsold",
                    "local_shift_contract": "block_inf_norm_relative_floor_v1",
                    "local_shift_floor_relative": 1e-6,
                    "model_kind": "learned_schwarz_v1",
                    "model_args": {
                        "block_feature_dim": 9,
                        "row_feature_dim": 14,
                    },
                    "model_state_dict": {},
                },
                mode_mismatch.name,
            )
            with self.assertRaisesRegex(ValueError, "initial_guess_mode mismatch"):
                load_learned_schwarz_v1_model(
                    mode_mismatch.name,
                    initial_guess_mode="zero",
                )


    def test_external_loader_accepts_continuation_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_path = root / (
                "continuation_circuit_7_time_1.500000e+00_"
                "gmin_2.500000e-01_iter_003.txt"
            )
            jacobian_path = root / (
                "continuation_circuit_7_time_1.500000e+00_"
                "gmin_2.500000e-01_iter_003_jac.txt"
            )
            step_path.write_text("fixture\n", encoding="utf-8")
            jacobian_path.write_text("fixture\n", encoding="utf-8")
            payload = {
                "rhsnew": [1.0, 2.0],
                "rhsold": [0.25, -0.5],
                "state0_in": [0.0, 0.0],
                "residual": [0.5, -0.25],
                "node_map": {"n1": 1, "n2": 2},
            }
            with mock.patch.object(
                external_gmres,
                "read_continuation_step",
                return_value=payload,
            ), mock.patch.object(
                external_gmres,
                "read_J",
                return_value=np.eye(2, dtype=np.float64),
            ):
                result = external_gmres._load_trajectory_linear_system_steps(
                    trajectory_dir=str(root),
                    circuit_id=7,
                    netlist_path="fixture.sp",
                )
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], None)
        self.assertEqual(len(result["steps"]), 1)
        step = result["steps"][0]
        self.assertEqual(step["iteration"], 3)
        self.assertEqual(step["gmin_val"], 0.25)
        np.testing.assert_allclose(step["rhs"], [1.0, 2.0])


    def test_external_loader_uses_exact_workpoint_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for iteration in (0, 1):
                step_path = root / (
                    "circuit_7_time_1.500000e+00_"
                    f"gmin_2_2.500000e-01_iter_{iteration:03d}.txt"
                )
                jacobian_path = root / (
                    "circuit_7_time_1.500000e+00_"
                    f"gmin_2_2.500000e-01_iter_{iteration:03d}_jac.txt"
                )
                step_path.write_text("fixture\n", encoding="utf-8")
                jacobian_path.write_text("fixture\n", encoding="utf-8")
            manifest_path = root / "workpoints.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = external_gmres._load_workpoint_manifest(str(manifest_path))
            payload = {
                "rhsnew": [1.0, 2.0],
                "rhsold": [0.25, -0.5],
                "state0_in": [0.0, 0.0],
                "residual": [0.5, -0.25],
                "node_map": {"n1": 1, "n2": 2},
            }
            with mock.patch.object(
                external_gmres,
                "read_continuation_step",
                return_value=payload,
            ) as read_step, mock.patch.object(
                external_gmres,
                "read_J",
                return_value=np.eye(2, dtype=np.float64),
            ) as read_jacobian:
                result = external_gmres._load_trajectory_linear_system_steps(
                    trajectory_dir=str(root),
                    circuit_id=7,
                    netlist_path="fixture.sp",
                    requested_workpoints=manifest[7],
                )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["iteration"], 1)
        self.assertEqual(result["steps"][0]["_workpoint_manifest_index"], 0)
        self.assertEqual(read_step.call_count, 1)
        self.assertEqual(read_jacobian.call_count, 1)


    def test_training_loader_binds_exact_manifest_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory_dir = root / "trajectory"
            netlist_dir = root / "netlists"
            trajectory_dir.mkdir()
            netlist_dir.mkdir()
            netlist_path = netlist_dir / "7.sp"
            netlist_path.write_text("fixture netlist\n", encoding="utf-8")

            def sha256(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            workpoints = []
            for iteration in (1, 0):
                step_path = trajectory_dir / (
                    "circuit_7_time_1.500000e+00_"
                    f"gmin_2_2.500000e-01_iter_{iteration:03d}.txt"
                )
                jacobian_path = trajectory_dir / (
                    "circuit_7_time_1.500000e+00_"
                    f"gmin_2_2.500000e-01_iter_{iteration:03d}_jac.txt"
                )
                step_path.write_text(f"step-{iteration}\n", encoding="utf-8")
                jacobian_path.write_text(f"jacobian-{iteration}\n", encoding="utf-8")
                workpoints.append(
                    {
                        "circuit_id": 7,
                        "time": 1.5,
                        "gmin_val": 0.25,
                        "iteration": iteration,
                        "step_sha256": sha256(step_path),
                        "jacobian_sha256": sha256(jacobian_path),
                        "netlist_sha256": sha256(netlist_path),
                    }
                )
            manifest_path = root / "training_workpoints.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "workpoints": workpoints}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                workpoint_manifest=str(manifest_path),
                netlist_dir=str(netlist_dir),
                trajectory_dir=str(trajectory_dir),
                circuit_ids="0",
                positive_gmin_only=True,
                step_offset=99,
                max_steps_per_circuit=1,
                max_samples=0,
                disable_gmin_diagonal=False,
                initial_guess_mode="rhsold",
                block_mode="cell_core_plus_onehop_boundary",
                core_block_mode="cell_core",
                boundary_block_mode="cell_core_plus_onehop_boundary",
                max_block_size=32,
                min_block_size=2,
                max_blocks=0,
                max_total_block_nnz=0,
                uncovered_row_policy="row_sum",
                model_kind="learned_schwarz_v1",
            )
            payload = {
                "rhsnew": [1.0, 2.0],
                "rhsold": [0.25, -0.5],
                "state0_in": [0.0, 0.0],
                "residual": [0.5, -0.25],
                "node_map": {"n1": 1, "n2": 2},
            }
            plan = self._fixture()[-1]
            with mock.patch.object(
                external_gmres,
                "read_continuation_step",
                return_value=payload,
            ), mock.patch.object(
                external_gmres,
                "read_J",
                return_value=np.eye(2, dtype=np.float64),
            ), mock.patch.object(
                learned_schwarz_train,
                "build_block_schwarz_plan",
                return_value=plan,
            ):
                samples = learned_schwarz_train._load_samples(args)
                self.assertEqual([item["iteration"] for item in samples], [1, 0])
                self.assertEqual(
                    [item["workpoint_manifest_index"] for item in samples], [0, 1]
                )
                self.assertEqual([item["time"] for item in samples], [1.5, 1.5])
                self.assertEqual(
                    args._workpoint_manifest_provenance["workpoint_count"], 2
                )
                self.assertTrue(
                    args._workpoint_manifest_provenance["source_hashes_verified"]
                )
                self.assertTrue(all(item["source_hashes_verified"] for item in samples))

                args.max_samples = 2
                accepted_count_samples = learned_schwarz_train._load_samples(args)
                self.assertEqual(len(accepted_count_samples), 2)
                args.max_samples = 0

                workpoints[0]["step_sha256"] = "0" * 64
                manifest_path.write_text(
                    json.dumps({"schema_version": 1, "workpoints": workpoints}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "workpoint_manifest_step_sha256_mismatch"
                ):
                    learned_schwarz_train._load_samples(args)

            args.max_samples = 1
            with self.assertRaisesRegex(
                ValueError, "workpoint_manifest_requires_max_samples_0_or_count"
            ):
                learned_schwarz_train._load_samples(args)


    def test_sparse_gmres_workpoint_manifest_enforces_exact_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write_step(
                circuit_id,
                iteration,
                *,
                prefix="",
                with_jacobian=True,
            ):
                step_path = root / (
                    f"{prefix}circuit_{circuit_id}_time_1.500000e+00_"
                    f"gmin_2.500000e-01_iter_{iteration:03d}.txt"
                )
                step_path.write_text("fixture\n", encoding="utf-8")
                if with_jacobian:
                    step_path.with_name(
                        step_path.stem + "_jac.txt"
                    ).write_text("fixture\n", encoding="utf-8")
                return step_path

            write_step(7, 0)
            write_step(7, 1)
            manifest_path = root / "exact_workpoints.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 1,
                            },
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = sparse_gmres._load_workpoint_manifest(
                str(manifest_path)
            )
            selected = sparse_gmres._find_steps(
                str(root),
                7,
                max_steps=1,
                requested_workpoints=manifest[7],
            )
            self.assertEqual([step["iteration"] for step in selected], [1, 0])
            self.assertEqual(
                [step["_workpoint_manifest_index"] for step in selected],
                [0, 1],
            )

            write_step(8, 0, prefix="continuation_")
            continuation_manifest_path = root / "continuation_workpoints.json"
            continuation_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 8,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            continuation_manifest = sparse_gmres._load_workpoint_manifest(
                str(continuation_manifest_path)
            )
            continuation_steps = sparse_gmres._find_steps(
                str(root),
                8,
                max_steps=1,
                requested_workpoints=continuation_manifest[8],
            )
            self.assertEqual(len(continuation_steps), 1)
            self.assertTrue(
                Path(continuation_steps[0]["step_path"]).name.startswith(
                    "continuation_circuit_"
                )
            )

    def test_sparse_gmres_workpoint_manifest_rejects_invalid_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_path = root / (
                "circuit_7_time_1.500000e+00_"
                "gmin_2.500000e-01_iter_000.txt"
            )
            step_path.write_text("fixture\n", encoding="utf-8")
            step_path.with_name(step_path.stem + "_jac.txt").write_text(
                "fixture\n",
                encoding="utf-8",
            )

            duplicate_path = root / "duplicate_workpoints.json"
            duplicate_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            },
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "workpoint_manifest_duplicate"):
                sparse_gmres._load_workpoint_manifest(str(duplicate_path))

            missing_path = root / "missing_workpoints.json"
            missing_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            missing_manifest = sparse_gmres._load_workpoint_manifest(
                str(missing_path)
            )
            with self.assertRaisesRegex(ValueError, "workpoint_manifest_missing"):
                sparse_gmres._find_steps(
                    str(root),
                    7,
                    max_steps=1,
                    requested_workpoints=missing_manifest[7],
                )

            no_jacobian_path = root / (
                "circuit_8_time_1.500000e+00_"
                "gmin_2.500000e-01_iter_000.txt"
            )
            no_jacobian_path.write_text("fixture\n", encoding="utf-8")
            missing_jacobian_manifest_path = root / "missing_jacobian.json"
            missing_jacobian_manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 8,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            missing_jacobian_manifest = sparse_gmres._load_workpoint_manifest(
                str(missing_jacobian_manifest_path)
            )
            with self.assertRaisesRegex(
                ValueError,
                "workpoint_manifest_jacobian_missing",
            ):
                sparse_gmres._find_steps(
                    str(root),
                    8,
                    max_steps=1,
                    requested_workpoints=missing_jacobian_manifest[8],
                )

            invalid_schema_path = root / "invalid_schema.json"
            invalid_schema_path.write_text(
                json.dumps({"schema_version": "1", "workpoints": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "workpoint_manifest_schema_version_must_be_1",
            ):
                sparse_gmres._load_workpoint_manifest(str(invalid_schema_path))

            invalid_key_path = root / "invalid_key.json"
            invalid_key_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 7.5,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "workpoint_manifest_item_has_invalid_key_fields",
            ):
                sparse_gmres._load_workpoint_manifest(str(invalid_key_path))

    def test_sparse_gmres_main_preserves_global_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            for circuit_id in (7, 8):
                step_path = root / (
                    f"circuit_{circuit_id}_time_1.500000e+00_"
                    "gmin_2.500000e-01_iter_000.txt"
                )
                step_path.write_text("fixture\n", encoding="utf-8")
                step_path.with_name(step_path.stem + "_jac.txt").write_text(
                    "fixture\n",
                    encoding="utf-8",
                )
            manifest_path = root / "interleaved_workpoints.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workpoints": [
                            {
                                "circuit_id": 8,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            },
                            {
                                "circuit_id": 7,
                                "time": 1.5,
                                "gmin_val": 0.25,
                                "iteration": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_evaluate(step, mode, args):
                return {
                    **step,
                    "mode": mode,
                    "ok": True,
                    "matrix_size": 1,
                    "matrix_nnz": 1,
                    "iterations": 1,
                    "residual_ratio": 0.0,
                    "reason": None,
                    "run_circuit_id": int(args.circuit_id),
                }

            with mock.patch.object(
                sparse_gmres,
                "evaluate_step",
                side_effect=fake_evaluate,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    "sparse_gmres_prototype.py",
                    "--trajectory-dir",
                    str(root),
                    "--workpoint-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--modes",
                    "row_sum",
                ],
            ):
                sparse_gmres.main()

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            jsonl_rows = [
                json.loads(line)
                for line in (output_dir / "per_step.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        for rows in (summary["rows"], jsonl_rows):
            self.assertEqual([row["circuit_id"] for row in rows], [8, 7])
            self.assertEqual(
                [row["_workpoint_manifest_index"] for row in rows],
                [0, 1],
            )
            self.assertEqual([row["run_circuit_id"] for row in rows], [8, 7])

    def test_external_learned_mode_requires_gmin_diagonal(self):
        config = SimpleNamespace(
            requested_mode="learned_schwarz_v1",
            apply_gmin_diagonal=False,
        )
        with self.assertRaisesRegex(ValueError, "gmin diagonal"):
            external_gmres.evaluate_corpus_step({}, config)


    def test_sparse_gmres_recomputes_initial_residual_with_gmin(self):
        matrix = sp.csr_matrix(np.asarray([[4.0, -1.0], [-1.0, 3.0]], dtype=np.float64))
        rhs = np.asarray([1.0, 2.0], dtype=np.float64)
        rhsold = np.asarray([0.2, -0.3], dtype=np.float64)
        args = SimpleNamespace(
            apply_gmin_diagonal=True, use_rhsold_as_x0=True,
            restart=4, max_iters=8, rtol=1e-8, atol=1e-10,
        )
        with mock.patch.object(sparse_gmres, "read_J_sparse", return_value=matrix), mock.patch.object(sparse_gmres, "read_continuation_step", return_value={"rhsnew": rhs, "rhsold": rhsold}), mock.patch.object(sparse_gmres, "_build_preconditioner", return_value=(None, {})):
            result = sparse_gmres._evaluate_scipy_sparse({"jacobian_path": "unused", "step_path": "unused", "gmin_val": 0.25}, "identity", args)
        expected = rhs - (matrix.toarray() + 0.25 * np.eye(2)) @ rhsold
        self.assertAlmostEqual(result["initial_residual_norm"], float(np.linalg.norm(expected)), places=12)


if __name__ == "__main__":
    unittest.main()
