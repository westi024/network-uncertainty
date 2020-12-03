"""

Tests the metrics implemented in this project

"""

import numpy as np
import unittest
from net_est.utils.metrics import picp


class TestPICP(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = np.linspace(0, 1, 10)
        self.lower = np.linspace(-1, 0, 10)
        self.upper = np.linspace(1, 2, 10)

    def test_picp_ops(self):
        self.assertEqual(picp(self.targets, self.lower, self.upper), 1.0)


if __name__ == '__main__':
    unittest.main()