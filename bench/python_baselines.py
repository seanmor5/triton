"""Python Triton / PyTorch baselines matching bench/kernel_bench.exs.

Run with the same environment as the Elixir benchmarks:

    source scripts/env.sh
    python3 bench/python_baselines.py
"""

import torch
import triton
import triton.language as tl


def bench_ms(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-1.0e30)
    e = tl.exp(x - tl.max(x, axis=0))
    tl.store(out_ptr + row * n_cols + offs, e / tl.sum(e, axis=0), mask=mask)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, BM: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K, BK):
        rk = kk + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def cdiv(a, b):
    return (a + b - 1) // b


def vector_add():
    print("\n## vector_add (Python Triton, f32, block=1024)\n")
    print("| case | time | throughput |\n|---|---|---|")
    for pow in (20, 22, 24, 26):
        n = 2 ** pow
        x = torch.rand(n, device="cuda") - 0.5
        y = torch.rand(n, device="cuda") - 0.5
        out = torch.empty(n, device="cuda")
        grid = (cdiv(n, 1024),)
        ms = bench_ms(lambda: add_kernel[grid](x, y, out, n, BLOCK=1024))
        gbps = 3 * n * 4 / (ms * 1e-3) / 1e9
        print(f"| n=2^{pow} | {ms:.4f} ms | {gbps:.1f} GB/s |")
        torch_ms = bench_ms(lambda: torch.add(x, y, out=out))
        print(f"| n=2^{pow} (torch.add) | {torch_ms:.4f} ms | {3*n*4/(torch_ms*1e-3)/1e9:.1f} GB/s |")


def softmax():
    print("\n## fused softmax (Python Triton, f32, 4096 rows)\n")
    print("| case | time | throughput |\n|---|---|---|")
    rows = 4096
    for cols in (256, 512, 1024, 2048, 4096):
        x = torch.rand(rows, cols, device="cuda")
        out = torch.empty_like(x)
        warps = 8 if cols >= 2048 else 4
        ms = bench_ms(lambda: softmax_kernel[(rows,)](x, out, cols, BLOCK=cols, num_warps=warps))
        gbps = 2 * rows * cols * 4 / (ms * 1e-3) / 1e9
        print(f"| 4096x{cols} | {ms:.4f} ms | {gbps:.1f} GB/s |")
        torch_ms = bench_ms(lambda: torch.softmax(x, dim=1, out=out) if hasattr(torch, "_softmax") else torch.softmax(x, dim=1))
        print(f"| 4096x{cols} (torch.softmax) | {torch_ms:.4f} ms | {2*rows*cols*4/(torch_ms*1e-3)/1e9:.1f} GB/s |")


def matmul():
    print("\n## matmul f32 (Python Triton, tf32, 64x64x64 blocks)\n")
    print("| case | time | throughput |\n|---|---|---|")
    torch.backends.cuda.matmul.allow_tf32 = True
    for size in (512, 1024, 2048):
        m = n = k = size
        a = torch.rand(m, k, device="cuda") - 0.5
        b = torch.rand(k, n, device="cuda") - 0.5
        c = torch.empty(m, n, device="cuda")
        grid = (cdiv(m, 64), cdiv(n, 64))
        ms = bench_ms(lambda: matmul_kernel[grid](a, b, c, m, n, K=k, BM=64, BN=64, BK=64, num_warps=4))
        tflops = 2 * m * n * k / (ms * 1e-3) / 1e12
        print(f"| {m}x{n}x{k} | {ms:.4f} ms | {tflops:.2f} TFLOPS |")
        torch_ms = bench_ms(lambda: torch.matmul(a, b, out=c))
        print(f"| {m}x{n}x{k} (cuBLAS tf32) | {torch_ms:.4f} ms | {2*m*n*k/(torch_ms*1e-3)/1e12:.2f} TFLOPS |")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    vector_add()
    softmax()
    matmul()
