"""

Tests the metrics implemented in this project

"""

import numpy as np
import unittest
from net_est.utils.metrics import picp, mpiw, calc_cwc, nmpiw


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = np.linspace(0, 1, 10)
        self.lower = np.linspace(-1, 0, 10)
        self.upper = np.linspace(1, 2, 10)

    def test_picp_ops(self):
        self.assertEqual(picp(self.targets, self.lower, self.upper), 1.0)

    def test_mpiw_ops(self):
        mpiw_result = mpiw(self.upper, self.lower)
        self.assertEqual(mpiw_result, 2.0)

    def test_cwc(self):
        picp_result = picp(self.targets, self.lower, self.upper)
        nmpiw_result = nmpiw(mpiw(self.upper, self.lower), R=0.5)
        self.assertEqual(calc_cwc(nmpiw_result, picp_result), 4.0)


if __name__ == '__main__':
    unittest.main()