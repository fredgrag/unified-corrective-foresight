from __future__ import annotations

import itertools
import unittest

from evaluate import flow_seed_stream


class EvaluateEntrypointTest(unittest.TestCase):
    def test_flow_seed_stream_is_reproducible_and_separate_from_environment_seed(self) -> None:
        first = tuple(itertools.islice(flow_seed_stream(101, 7), 8))
        repeated = tuple(itertools.islice(flow_seed_stream(101, 7), 8))
        other_environment = tuple(itertools.islice(flow_seed_stream(101, 8), 8))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_environment)
        self.assertTrue(all(type(value) is int and value >= 0 for value in first))


if __name__ == "__main__":
    unittest.main()
