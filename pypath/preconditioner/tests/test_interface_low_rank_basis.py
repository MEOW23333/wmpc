import unittest

import numpy as np

from pypath.preconditioner.interface_low_rank_basis import (
    InterfaceLowRankSchurPreconditioner,
    WoodburyLowRankCorrector,
    build_interface_basis,
)


class InterfaceLowRankBasisTest(unittest.TestCase):
    def setUp(self):
        self.weights = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.schur = np.asarray(
            [
                [2.0, -0.5, 0.0, 0.0],
                [-0.5, 3.0, -0.2, 0.0],
                [0.0, -0.2, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.25],
            ],
            dtype=np.float64,
        )

    def test_constant_basis_uses_components(self):
        basis, info = build_interface_basis(
            "constant",
            graph_weights=self.weights,
            requested_rank=4,
        )
        self.assertEqual(basis.shape, (4, 2))
        self.assertEqual(info["actual_rank"], 2)
        np.testing.assert_allclose(basis.T.dot(basis), np.eye(2), atol=1e-12)

    def test_graph_laplacian_reports_sorted_low_frequencies(self):
        basis, info = build_interface_basis(
            "graph_laplacian",
            graph_weights=self.weights,
            requested_rank=2,
        )
        self.assertEqual(basis.shape, (4, 2))
        values = np.asarray(info["laplacian_eigenvalues"])
        self.assertTrue(np.all(np.diff(values) >= -1e-12))

    def test_snapshot_pod_reports_energy(self):
        snapshots = np.asarray(
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0],
            ]
        )
        basis, info = build_interface_basis(
            "snapshot_pod",
            snapshots=snapshots,
            requested_rank=2,
        )
        self.assertEqual(basis.shape, (4, 2))
        self.assertGreater(float(info["snapshot_energy_retained"]), 0.99)

    def test_schur_slow_eigenbasis_selects_small_absolute_values(self):
        basis, info = build_interface_basis(
            "schur_slow_eig",
            interface_matrix=np.diag([4.0, -0.1, 2.0, 0.5]),
            requested_rank=2,
        )
        self.assertEqual(basis.shape, (4, 2))
        self.assertEqual(len(info["schur_eigenvalues"]), 2)
        self.assertLessEqual(abs(info["schur_eigenvalues"][0]), 0.11)

    def test_zero_snapshot_falls_back_to_constant(self):
        basis, info = build_interface_basis(
            "snapshot_pod",
            graph_weights=self.weights,
            snapshots=np.zeros((4, 2)),
            requested_rank=1,
        )
        self.assertTrue(bool(info.get("fallback_used")))
        self.assertEqual(basis.shape[0], 4)
        self.assertEqual(info["actual_rank"], 1)

    def test_woodbury_matches_target_inverse(self):
        base = np.asarray([[3.0, 0.2], [0.2, 2.0]])
        target = base + np.asarray([[0.4, 0.0], [0.0, 0.0]])
        basis = np.ones((2, 1), dtype=np.float64)
        basis /= np.linalg.norm(basis)
        corrector = WoodburyLowRankCorrector(
            base_matrix=base,
            target_matrix=target,
            basis=basis,
            base_solve=lambda value: np.linalg.solve(base, value),
        )
        rhs = np.asarray([1.0, -0.5])
        projected_target = base + basis.dot(
            basis.T.dot(target - base).dot(basis)
        ).dot(basis.T)
        np.testing.assert_allclose(
            corrector.apply(rhs),
            np.linalg.solve(projected_target, rhs),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
