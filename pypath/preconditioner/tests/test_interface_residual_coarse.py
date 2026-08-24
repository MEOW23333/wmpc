import inspect
import unittest

import numpy as np
import scipy.sparse as sp

from pypath.precondition_construction.sparse import (
    SPARSE_SEMANTIC_MODES,
    SparseLocalSchurPreconditioner,
    SparseSemanticBlockJacobi,
    _parse_interface_residual_mode,
)
from pypath.preconditioner.interface_residual_coarse import (
    InterfaceResidualCoarsePreconditioner,
    ResidualCoarseCorrector,
)


class InterfaceResidualCoarseTest(unittest.TestCase):
    @staticmethod
    def _base_schur():
        matrix = np.asarray(
            [
                [4.0, 1.0, 1.0, 0.2],
                [1.0, 3.0, 0.5, 1.0],
                [1.2, 0.4, 3.0, 0.2],
                [0.1, 1.1, 0.3, 2.5],
            ],
            dtype=np.float64,
        )
        sparse = sp.csr_matrix(matrix)
        core_rows = np.asarray([0, 1], dtype=np.int64)
        core = SparseSemanticBlockJacobi(
            sparse,
            [core_rows],
            uncovered_policy="row_sum",
        )
        preconditioner = SparseLocalSchurPreconditioner(
            matrix=sparse,
            core=core,
            boundary_blocks=[np.arange(4, dtype=np.int64)],
            strategy="topk_abs",
            edge_budget=16,
            budget_multiplier=2.0,
            candidate_edge_limit=64,
            diagonal_shift=0.0,
            interface_solve_mode="spilu",
        )
        return matrix, preconditioner

    def test_rank_one_analytic_interface_correction(self):
        schur = 0.2 * np.eye(2)
        corrector = ResidualCoarseCorrector(
            schur_apply=lambda value: schur.dot(value),
            base_solve=lambda value: 0.5 * value,
            basis=np.ones((2, 1), dtype=np.float64),
            test_space="range_action",
        )
        np.testing.assert_allclose(
            corrector.apply(np.ones(2)),
            np.asarray([5.0, 5.0]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_matches_explicit_formula_for_both_test_spaces(self):
        schur = np.asarray(
            [
                [2.0, 0.4, 0.0],
                [0.1, 1.5, 0.3],
                [0.0, 0.2, 1.0],
            ]
        )
        base_matrix = np.diag([2.5, 2.0, 1.5])
        basis, _ = np.linalg.qr(
            np.asarray(
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            )
        )
        rhs = np.asarray([0.7, -0.2, 1.1])
        for mode in ("range_action", "galerkin"):
            with self.subTest(mode=mode):
                corrector = ResidualCoarseCorrector(
                    schur_apply=lambda value: schur.dot(value),
                    base_solve=lambda value: np.linalg.solve(
                        base_matrix, value
                    ),
                    basis=basis,
                    test_space=mode,
                )
                base = np.linalg.solve(base_matrix, rhs)
                defect = rhs - schur.dot(base)
                expected = base + corrector.basis.dot(
                    np.linalg.solve(
                        corrector.test_basis.T.dot(
                            schur.dot(corrector.basis)
                        ),
                        corrector.test_basis.T.dot(defect),
                    )
                )
                np.testing.assert_allclose(
                    corrector.apply(rhs),
                    expected,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
                self.assertEqual(corrector.test_basis.shape, (3, 2))

    def test_invalid_and_singular_bases_fall_back(self):
        rhs = np.asarray([1.0, -2.0])
        base = lambda value: 0.25 * np.asarray(value)
        cases = [
            ResidualCoarseCorrector(
                schur_apply=lambda value: np.asarray(value),
                base_solve=base,
                basis=np.asarray([[np.nan], [1.0]]),
            ),
            ResidualCoarseCorrector(
                schur_apply=lambda value: np.zeros_like(value),
                base_solve=base,
                basis=np.ones((2, 1)),
            ),
        ]
        for corrector in cases:
            with self.subTest(reason=corrector.fallback_reason):
                self.assertFalse(corrector.enabled)
                np.testing.assert_array_equal(
                    corrector.apply(rhs), base(rhs)
                )

    def test_reduced_condition_guard_falls_back(self):
        schur = np.diag([1.0, 100.0])
        corrector = ResidualCoarseCorrector(
            schur_apply=lambda value: schur.dot(value),
            base_solve=lambda value: np.asarray(value),
            basis=np.eye(2),
            test_space="galerkin",
            max_condition=10.0,
        )
        self.assertFalse(corrector.enabled)
        self.assertIn("reduced_condition_exceeded", corrector.fallback_reason)

    def test_runtime_exception_disables_correction_and_falls_back(self):
        calls = {"count": 0}

        def schur_apply(value):
            calls["count"] += 1
            if calls["count"] > 1:
                raise RuntimeError("runtime_failure")
            return np.asarray(value)

        corrector = ResidualCoarseCorrector(
            schur_apply=schur_apply,
            base_solve=lambda value: 0.5 * np.asarray(value),
            basis=np.ones((2, 1)),
        )
        rhs = np.asarray([2.0, -1.0])
        np.testing.assert_array_equal(corrector.apply(rhs), 0.5 * rhs)
        self.assertFalse(corrector.enabled)
        self.assertEqual(corrector.runtime_fallback_count, 1)

    def test_interface_schur_action_matches_explicit_matrix(self):
        matrix, base = self._base_schur()
        core = np.asarray([0, 1])
        interface = np.asarray([2, 3])
        explicit = (
            matrix[np.ix_(interface, interface)]
            - matrix[np.ix_(interface, core)].dot(
                np.linalg.solve(
                    matrix[np.ix_(core, core)],
                    matrix[np.ix_(core, interface)],
                )
            )
        )
        probe = np.asarray([[0.7, 1.0], [-0.4, 0.2]])
        actual = base.apply_interface_schur(probe)
        relative = np.linalg.norm(actual - explicit.dot(probe)) / max(
            np.linalg.norm(explicit.dot(probe)), 1.0e-30
        )
        self.assertLessEqual(relative, 1.0e-12)
        np.testing.assert_allclose(
            base.apply_interface_schur(probe[:, 0]),
            explicit.dot(probe[:, 0]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_interface_schur_path_has_no_global_toarray(self):
        source = inspect.getsource(
            SparseLocalSchurPreconditioner.apply_interface_schur
        )
        self.assertNotIn("toarray(", source)
        _, base = self._base_schur()
        self.assertEqual(base.apply_interface_schur(np.ones(2)).shape, (2,))

    def test_full_wrapper_disabled_output_matches_base(self):
        _, base = self._base_schur()
        rhs = np.asarray([0.3, -0.7, 1.2, 0.4])
        expected = base._apply_with_factor(
            rhs,
            base.schur_factor,
            base.schur_factor_mode,
            base.schur_matrix,
        )
        wrapper = InterfaceResidualCoarsePreconditioner(
            base=base,
            method="constant",
            requested_rank=1,
            guard_vector=rhs,
            max_condition=1.0,
        )
        self.assertFalse(wrapper.corrector.enabled)
        np.testing.assert_array_equal(wrapper.apply(rhs), expected)

    def test_modes_and_parser_are_registered(self):
        mode = "interface_residual_snapshot_pod_r4_sparse"
        self.assertIn(mode, SPARSE_SEMANTIC_MODES)
        self.assertEqual(
            _parse_interface_residual_mode(mode),
            ("snapshot_pod", 4),
        )
        self.assertIsNone(
            _parse_interface_residual_mode(
                "interface_residual_unknown_r4_sparse"
            )
        )


if __name__ == "__main__":
    unittest.main()
