"""End-to-end and unit tests for v3 (Inductor cpp_wrapper probe)."""
import unittest

import torch
import torch._inductor.lowering as _lowering
import torch_dispatch_capture.v3 as tdcv3


class TestForceAllFallback(unittest.TestCase):
    def test_restores_lowerings_on_exit(self):
        before = dict(_lowering.lowerings)
        with tdcv3.force_all_fallback():
            inside = dict(_lowering.lowerings)
        after = dict(_lowering.lowerings)
        self.assertEqual(before.keys(), after.keys())
        # At least one entry must have been rewritten to a fallback handler.
        patched_op_count = sum(
            1
            for k, v in inside.items()
            if isinstance(k, torch._ops.OpOverload)
            and getattr(v, "_is_fallback_handler", False)
        )
        self.assertGreater(patched_op_count, 50)
        # And the original handler must be restored.
        for k, v in before.items():
            self.assertIs(after[k], v)


if __name__ == "__main__":
    unittest.main()
