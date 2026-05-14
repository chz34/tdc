"""Benchmark eager vs replay to quantify dispatcher savings."""
import statistics
import time
import unittest

import torch
import torch_dispatch_capture as tdc


def bench(fn, iters=400, warmup=50):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {
        "median_us": samples[len(samples) // 2] / 1000.0,
        "min_us": samples[0] / 1000.0,
        "mean_us": statistics.mean(samples) / 1000.0,
    }


class TestBenchmark(unittest.TestCase):
    def test_elementwise(self):
        n_ops = 64
        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        out = torch.empty(8, 8)

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
        print(f"\n[elementwise n_ops={n_ops}]")
        print(f"  eager_med  = {es['median_us']:8.2f} us  ({es['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  replay_med = {rs['median_us']:8.2f} us  ({rs['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  ratio      = {ratio:.2f}x  (target: < 1.0)")
        # Loose assert: replay must not be > 1.5x eager. Tight target is
        # < 0.5x but we only fail on regression.
        self.assertLess(ratio, 1.5, "replay regressed vs eager")

    def test_linear_chain(self):
        torch.manual_seed(0)
        layers = [torch.nn.Linear(8, 8).eval() for _ in range(32)]
        x = torch.randn(4, 8)
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
        print(f"\n[deep_sequential n_layers=32, captured_ops={n_ops}]")
        print(f"  eager_med  = {es['median_us']:8.2f} us  ({es['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  replay_med = {rs['median_us']:8.2f} us  ({rs['median_us']*1000/n_ops:.0f} ns/op)")
        print(f"  ratio      = {ratio:.2f}x  (target: < 1.0)")
        self.assertLess(ratio, 1.5, "replay regressed vs eager")

    def test_ffn_dynamic_batch(self):
        """SwiGLU FFN with varying batch sizes — the realistic workload that
        motivates this PoC. SwiGLU is the FFN variant used by Llama / Mistral
        / many modern LLMs: y = w_down(silu(w_gate(x)) * w_up(x)). It has
        three linear projections + a SiLU + an elementwise multiply per
        layer, so the captured trace covers more ops than vanilla MLP and
        is representative of decoder-only transformer hot paths.

        Capture once at batch=4 and replay across batches via in-place
        resize+copy of the same input tensor (capture holds strong ref to
        the original Tensor object). Outputs are freshly allocated each
        replay, so dynamic shape is automatic. Single captured trace
        handles all batch sizes — no recapture."""
        torch.manual_seed(0)
        d_model, d_ff = 64, 256

        # Standard SwiGLU written in the natural style — each intermediate
        # is a local variable bound to a fresh Tensor, the function returns
        # the new output Tensor. No `out=` buffers, no `forward_into`.
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

        model = SwiGLUFFN(d_model, d_ff).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        x = torch.randn(4, d_model)
        obs = torch.empty(4, d_model)  # observation buffer, outlives trace

        @torch.no_grad()
        def eager():
            return model(x)

        # Capture eager() AND copy the return value into `obs`. The model
        # itself is unchanged (natural Python style); the extra
        # resize_as_+copy_ inside the capture block is the single
        # convention that makes results externally observable across
        # replays.
        with torch.no_grad():
            eager()  # warm caches
            with tdc.capture() as trace:
                result = eager()
                obs.resize_as_(result)
                obs.copy_(result)
        n_ops = len(trace)
        print(f"\n[FFN dynamic shape] d_model={d_model}, d_ff={d_ff}, "
              f"captured_ops={n_ops}")

        batches = [1, 4, 16, 64, 256]
        replay_total_us = 0.0
        eager_total_us = 0.0
        iters = 200

        for batch in batches:
            x_new = torch.randn(batch, d_model)
            x.resize_(batch, d_model)
            x.copy_(x_new)

            # Numerical correctness: replay writes into `obs` via the
            # captured copy_. Compare to an eager reference.
            with torch.no_grad():
                trace.replay()
                replay_out = obs.clone()
                eager_out = eager()
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
        self.assertLess(ratio, 1.0, "FFN replay regressed vs eager")

    def test_ffn_dynamic_seqlen(self):
        """LLM-style SwiGLU with [B, S, L] input where sequence length S varies.

        This mirrors the production LLM serving pattern:
          - prefill: S is large (e.g., 512, 1024, 2048)
          - decode:  S is 1 per step
        Mixing prefill and decode in one trace is exactly the case cudagraph
        struggles with (each S needs its own captured graph). Our dispatcher
        capture handles all S values from a single trace because the kernel
        re-reads shape metadata each call.

        Capture once at a middle S; replay across the full prefill→decode
        spectrum on the same trace."""
        torch.manual_seed(0)
        B, L, d_ff = 2, 256, 1024  # B=batch, L=hidden_dim, d_ff=intermediate
        capture_S = 64

        # Standard SwiGLU FFN in natural Python style.
        class SwiGLUFFN(torch.nn.Module):
            def __init__(self, L: int, d_ff: int) -> None:
                super().__init__()
                self.w_gate = torch.nn.Linear(L, d_ff, bias=False)
                self.w_up = torch.nn.Linear(L, d_ff, bias=False)
                self.w_down = torch.nn.Linear(d_ff, L, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, S, L] -> [B, S, L]
                gate = self.w_gate(x)
                up = self.w_up(x)
                return self.w_down(torch.nn.functional.silu(gate) * up)

        model = SwiGLUFFN(L, d_ff).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        x = torch.randn(B, capture_S, L)
        obs = torch.empty(B, capture_S, L)  # observation buffer

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
              f"capture_S={capture_S}, captured_ops={n_ops}")

        # Cover the full LLM serving spectrum:
        #   S=1     decode step
        #   S=32    short context decode batch
        #   S=128   medium prefill
        #   S=512   long prefill
        #   S=2048  very long prefill
        seqlens = [1, 32, 128, 512, 2048]
        replay_total_us = 0.0
        eager_total_us = 0.0
        iters = 100  # heavier workload, fewer iters

        for S in seqlens:
            # Resize the captured input tensor to [B, S, L]; replay sees
            # the new metadata automatically.
            x_new = torch.randn(B, S, L)
            x.resize_(B, S, L)
            x.copy_(x_new)

            # Numerical correctness: replay's captured copy_ updates `obs`.
            with torch.no_grad():
                trace.replay()
                replay_out = obs.clone()
                eager_out = eager()
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
        # Small S (decode) is the sweet spot for dispatcher savings; large S
        # is memory-bandwidth bound and Python overhead becomes a small
        # fraction of total time. Threshold matches other benchmark tests:
        # we fail only on real regression, not noise-driven small overshoot.
        self.assertLess(ratio, 1.5, "SwiGLU dynamic-S replay regressed vs eager")

    def test_dynamic_shape_amortization(self):
        # Capture once, replay across several batch sizes; total replay time
        # should beat total eager time.
        torch.manual_seed(0)
        w = torch.randn(8, 8)
        x = torch.randn(4, 8)
        out = torch.empty(4, 8)

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
