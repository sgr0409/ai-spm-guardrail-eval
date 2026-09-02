import unittest

import numpy as np

from experiments.run_regime_stream import SCHEDULE, allocate_stream


class RegimeStreamTests(unittest.TestCase):
    def test_allocator_keeps_reference_monitor_calibration_and_test_disjoint(self):
        sizes = {
            "source_synthetic": 600,
            "deepset": 800,
            "xtram": 1200,
            "neuralchemy": 800,
            "safeguard": 800,
        }
        labels = {name: np.array([0] * (n // 2) + [1] * (n // 2)) for name, n in sizes.items()}
        reference, windows = allocate_stream(labels, seed=7)
        used = {name: set() for name in sizes}
        used["source_synthetic"].update(reference.tolist())
        for domain, window in zip(SCHEDULE, windows):
            for split in ("monitor", "calibration", "test"):
                indices = set(window[split].tolist())
                self.assertTrue(used[domain].isdisjoint(indices))
                used[domain].update(indices)


if __name__ == "__main__":
    unittest.main()
