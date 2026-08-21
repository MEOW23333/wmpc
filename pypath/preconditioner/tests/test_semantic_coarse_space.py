import unittest

import numpy as np
import scipy.sparse as sp

from pypath.preconditioner.semantic_coarse_space import (
    build_semantic_coarse_operator,
)


class SemanticCoarseSpaceTest(unittest.TestCase):
    def test_two_level_operator_preserves_finite_output(self):
        matrix = sp.csr_matrix(
            np.asarray(
                [
                    [4.0, -1.0, 0.0, 0.0],
                    [-1.0, 4.0, -1.0, 0.0],
                    [0.0, -1.0, 4.0, -1.0],
                    [0.0, 0.0, -1.0, 3.0],
                ],
                dtype=np.float64,
            )
        )
        local = lambda value: np.asarray(value, dtype=np.float64) / matrix.diagonal()
        operator, info = build_semantic_coarse_operator(
            matrix=matrix,
            local_apply=local,
            coarse_blocks=[[0, 1], [1, 2], [2, 3]],
            mode_count=1,
        )
        output = np.asarray(operator.matvec(np.ones(4)), dtype=np.float64)
        self.assertTrue(bool(info.get("coarse_enabled_after_guard")))
        self.assertGreaterEqual(int(info.get("coarse_rank", 0)), 1)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_guard_falls_back_for_invalid_condition_limit(self):
        matrix = sp.eye(3, format="csr", dtype=np.float64)
        local = lambda value: np.asarray(value, dtype=np.float64)
        operator, info = build_semantic_coarse_operator(
            matrix=matrix,
            local_apply=local,
            coarse_blocks=[[0, 1, 2]],
            mode_count=1,
            max_condition=1.0,
        )
        output = np.asarray(operator.matvec(np.ones(3)), dtype=np.float64)
        self.assertFalse(bool(info.get("coarse_enabled_after_guard")))
        self.assertIsNotNone(info.get("fallback_reason"))
        self.assertTrue(np.allclose(output, np.ones(3)))


if __name__ == "__main__":
    unittest.main()
