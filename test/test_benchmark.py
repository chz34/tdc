"""Benchmark eager vs replay to quantify dispatcher savings.

Runs on CPU by default. To benchmark on another accelerator, set the
TDC_DEVICE env var before invoking:

    TDC_DEVICE=cuda  python -m unittest test_benchmark
    TDC_DEVICE=xpu   python -m unittest test_benchmark
    TDC_DEVICE=mps   python -m unittest test_benchmark
    TDC_DEVICE=npu   python -m unittest test_benchmark            # PrivateUse1
    TDC_DEVICE=privateuseone  python -m unittest test_benchmark   # generic

The accelerator path adds a device-synchronize call after every benched
fn() so the wall-clock time accurately captures kernel completion (not
just the dispatch-queue submission). For CPU the sync is a no-op.
"""
import statistics
import time
import unittest

import torch
import torch_dispatch_capture as tdc

from _device import DEVICE, SYNC, print_device_banner


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

def bench(fn, iters=400, warmup=50):
    for _ in range(warmup):
        fn()
        SYNC()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        SYNC()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {
        "median_us": samples[len(samples) // 2] / 1000.0,
        "min_us": samples[0] / 1000.0,
        "mean_us": statistics.mean(samples) / 1000.0,
    }


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print_device_banner()

    def test_elementwise(self):
        n_ops = 64
        a = torch.randn(8, 8, device=DEVICE)
        b = torch.randn(8, 8, device=DEVICE)
        out = torch.empty(8, 8, device=DEVICE)

        def eager():
            for _ in range(n_ops):
                torch.add(a, b, out=out)

        with torch.no_grad():
            eager()  # warm caches
            with tdc.capture() as trace:
                eager()
            self.assertEqual(len(trace), n_ops)

            es = bench(eager)
            rs = bench(trace.replay)

        ratio = rs["median_us"] / es["median_us"]
        print(f"\n[elementwise n_ops={n_ops}] device={DEVICE}")
        print(f"  eager_med  = {es['median_us']:8.2f} us  ({es['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  replay_med = {rs['median_us']:8.2f} us  ({rs['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  ratio      = {ratio:.2f}x  (target: < 1.0)")
        self.assertLess(ratio, 1.5, "replay regressed vs eager")

    def test_linear_chain(self):
        torch.manual_seed(0)
        layers = [torch.nn.Linear(8, 8).eval().to(DEVICE) for _ in range(32)]
        x = torch.randn(4, 8, device=DEVICE)
        out = torch.empty_like(x)

        @torch.no_grad()
        def eager():
            t = x
            for layer in layers:
                t = layer(t)
            out.copy_(t)

        with torch.no_grad():
            eager()
            with tdc.capture() as trace:
                eager()

            es = bench(eager)
            rs = bench(trace.replay)

        n_ops = len(trace)
        ratio = rs["median_us"] / es["median_us"]
        print(f"\n[deep_sequential n_layers=32, captured_ops={n_ops}] device={DEVICE}")
        print(f"  eager_med  = {es['median_us']:8.2f} us  ({es['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  replay_med = {rs['median_us']:8.2f} us  ({rs['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  ratio      = {ratio:.2f}x  (target: < 1.0)")
        self.assertLess(ratio, 1.5, "replay regressed vs eager")

    def test_ffn_dynamic_batch(self):
        """SwiGLU FFN with varying batch sizes — capture once, replay all."""
        torch.manual_seed(0)
        d_model, d_ff = 64, 256

        class SwiGLUFFN(torch.nn.Module):
            def __init__(self, d_model: int, d_ff: int) -> None:
                super().__init__()
                self.w_gate = torch.nn.Linear(d_model, d_ff, bias=False)
                self.w_up = torch.nn.Linear(d_model, d_ff, bias=False)
                self.w_down = torch.nn.Linear(d_ff, d_model, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = self.w_gate(x)
                up = self.w_up(x)
                return self.w_down(torch.nn.functional.silu(gate) * up)

        model = SwiGLUFFN(d_model, d_ff).eval().to(DEVICE)
        for p in model.parameters():
            p.requires_grad_(False)

        x = torch.randn(4, d_model, device=DEVICE)
        obs = torch.empty(4, d_model, device=DEVICE)

        @torch.no_grad()
        def eager():
            return model(x)

        with torch.no_grad():
            eager()
            with tdc.capture() as trace:
                result = eager()
                obs.resize_as_(result)
                obs.copy_(result)
        n_ops = len(trace)
        print(f"\n[FFN dynamic batch] d_model={d_model}, d_ff={d_ff}, "
              f"captured_ops={n_ops}, device={DEVICE}")

        batches = [1, 4, 16, 64, 256]
        replay_total_us = 0.0
        eager_total_us = 0.0
        iters = 200

        for batch in batches:
            x_new = torch.randn(batch, d_model, device=DEVICE)
            x.resize_(batch, d_model)
            x.copy_(x_new)

            with torch.no_grad():
                trace.replay()
                SYNC()
                replay_out = obs.clone()
                eager_out = eager()
                SYNC()
            torch.testing.assert_close(
                replay_out, eager_out,
                msg=lambda m: f"numerics mismatch at batch={batch}: {m}")

            es = bench(eager, iters=iters, warmup=20)
            rs = bench(trace.replay, iters=iters, warmup=20)
            replay_total_us += rs["median_us"] * iters
            eager_total_us += es["median_us"] * iters

            per_op_save = (es["median_us"] - rs["median_us"]) * 1000 / n_ops
            print(f"  batch={batch:4d}  eager={es['median_us']:7.2f}us  "
                  f"replay={rs['median_us']:7.2f}us  "
                  f"ratio={rs['median_us']/es['median_us']:.2f}x  "
                  f"per_op_save={per_op_save:5.0f}ns  ✓numerics")

        ratio = replay_total_us / eager_total_us
        print(f"\n[FFN dynamic batch total across {batches}]")
        print(f"  total_ratio = {ratio:.2f}x   (single capture, {len(batches)} shapes)")
        self.assertLess(ratio, 1.5, "FFN replay regressed vs eager")

    def test_ffn_dynamic_seqlen(self):
        """LLM-style SwiGLU with [B, S, L] input — capture once, replay
        across the full prefill→decode spectrum on the same trace."""
        torch.manual_seed(0)
        B, L, d_ff = 2, 256, 1024
        capture_S = 64

        class SwiGLUFFN(torch.nn.Module):
            def __init__(self, L: int, d_ff: int) -> None:
                super().__init__()
                self.w_gate = torch.nn.Linear(L, d_ff, bias=False)
                self.w_up = torch.nn.Linear(L, d_ff, bias=False)
                self.w_down = torch.nn.Linear(d_ff, L, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = self.w_gate(x)
                up = self.w_up(x)
                return self.w_down(torch.nn.functional.silu(gate) * up)

        model = SwiGLUFFN(L, d_ff).eval().to(DEVICE)
        for p in model.parameters():
            p.requires_grad_(False)

        x = torch.randn(B, capture_S, L, device=DEVICE)
        obs = torch.empty(B, capture_S, L, device=DEVICE)

        @torch.no_grad()
        def eager():
            return model(x)

        with torch.no_grad():
            eager()
            with tdc.capture() as trace:
                result = eager()
                obs.resize_as_(result)
                obs.copy_(result)

        n_ops = len(trace)
        print(f"\n[SwiGLU dynamic seqlen] B={B}, L={L}, d_ff={d_ff}, "
              f"capture_S={capture_S}, captured_ops={n_ops}, device={DEVICE}")

        seqlens = [1, 32, 128, 512, 2048]
        replay_total_us = 0.0
        eager_total_us = 0.0
        iters = 100

        for S in seqlens:
            x_new = torch.randn(B, S, L, device=DEVICE)
            x.resize_(B, S, L)
            x.copy_(x_new)

            with torch.no_grad():
                trace.replay()
                SYNC()
                replay_out = obs.clone()
                eager_out = eager()
                SYNC()
            torch.testing.assert_close(
                replay_out, eager_out,
                msg=lambda m: f"numerics mismatch at S={S}: {m}")

            es = bench(eager, iters=iters, warmup=20)
            rs = bench(trace.replay, iters=iters, warmup=20)
            replay_total_us += rs["median_us"] * iters
            eager_total_us += es["median_us"] * iters

            per_op_save = (es["median_us"] - rs["median_us"]) * 1000 / n_ops
            speedup = es["median_us"] / rs["median_us"]
            print(f"  S={S:5d}  eager={es['median_us']:8.2f}us  "
                  f"replay={rs['median_us']:8.2f}us  "
                  f"ratio={rs['median_us']/es['median_us']:.2f}x  "
                  f"speedup={speedup:.2f}x  "
                  f"per_op_save={per_op_save:5.0f}ns  ✓numerics")

        ratio = replay_total_us / eager_total_us
        print(f"\n[SwiGLU dynamic seqlen total across S={seqlens}]")
        print(f"  total_ratio = {ratio:.2f}x   (single capture, {len(seqlens)} shapes,"
              f" S range {min(seqlens)}-{max(seqlens)})")
        self.assertLess(ratio, 1.5, "SwiGLU dynamic-S replay regressed vs eager")

    def test_dynamic_shape_amortization(self):
        torch.manual_seed(0)
        w = torch.randn(8, 8, device=DEVICE)
        x = torch.randn(4, 8, device=DEVICE)
        out = torch.empty(4, 8, device=DEVICE)

        @torch.no_grad()
        def eager():
            torch.matmul(x, w.t(), out=out)

        with torch.no_grad():
            eager()
            with tdc.capture() as trace:
                eager()

            batches = [1, 4, 16, 64]
            replay_total_us = 0.0
            eager_total_us = 0.0
            iters = 200
            print(f"\n[dynamic shape amortization] device={DEVICE}")
            for batch in batches:
                x.resize_(batch, 8); x.normal_()
                out.resize_(batch, 8)

                rs = bench(trace.replay, iters=iters, warmup=20)
                es = bench(eager, iters=iters, warmup=20)
                replay_total_us += rs["median_us"] * iters
                eager_total_us += es["median_us"] * iters
                print(f"  batch={batch:3d}  eager={es['median_us']:6.2f}us  "
                      f"replay={rs['median_us']:6.2f}us  "
                      f"ratio={rs['median_us']/es['median_us']:.2f}x")

        ratio = replay_total_us / eager_total_us
        print(f"\n[dynamic shape total across {batches}]")
        print(f"  total_ratio = {ratio:.2f}x  (target: < 1.0, no recapture)")
        self.assertLess(ratio, 1.5, "dynamic-shape replay regressed")


if __name__ == "__main__":
    unittest.main()
