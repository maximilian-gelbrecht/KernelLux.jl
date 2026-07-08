# KernelLux

[![CI](https://github.com/maximilian-gelbrecht/KernelLux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/maximilian-gelbrecht/KernelLux.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://maximilian-gelbrecht.github.io/KernelLux.jl/dev/)

Run [Lux](https://github.com/LuxDL/Lux.jl) models **inside**
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) kernels that are compiled
by [Reactant](https://github.com/EnzymeAD/Reactant.jl).

`Lux.apply(model, x, ps, st)` does not compile inside such a kernel: its `Dense` forward pass
relies on machinery (`NNlib.fast_act`, `match_eltype`, `fused_dense_bias_activation`, …) that
performs heap allocation and string/`Symbol` handling with no device lowering, producing an
`InvalidIRError`. KernelLux provides a drop-in replacement,

```julia
apply_in_kernel(model, x, ps, st)
```

with the same signature and return value as `Lux.apply`, implemented with only scalar and
`StaticArrays` operations that lower cleanly to the GPU (and CPU).

## Scope

The current implementation supports a `Lux.Chain` of `Lux.Dense` layers — with or without bias,
and with any device-safe scalar activation (`tanh`, `relu`, `identity`, …).

> [!WARNING]
> **KernelLux targets small to medium sized networks.** These are the narrow multilayer
> perceptrons that typically appear in hybrid / physics-informed models, evaluated pointwise
> inside a larger simulation kernel. **Large networks (wide hidden layers) are slow** in this
> per-thread setting — each thread holds the entire hidden activation vector, so GPU occupancy
> collapses (see [Performance](#performance)). If your workload can be expressed as a single
> batched matrix multiply, prefer the ordinary `model(xmat, ps, st)` forward pass.

## Installation

```julia
import Pkg
Pkg.add(url = "https://github.com/maximilian-gelbrecht/KernelLux.jl")
```

## Usage

`apply_in_kernel` operates on a **single sample**: `x` is an `SVector` of the input features and
the returned `y` is an `SVector`. Inside a kernel you build that `SVector` from your per-thread
data and call the model:

```julia
using KernelLux, Lux, Random, StaticArrays, Reactant, CUDA
using KernelAbstractions: @kernel, @index, get_backend, @Const
import Lux.MLDataDevices as MLDD

dev = MLDD.reactant_device()

model = Chain(Dense(1 => 8, tanh), Dense(8 => 1))
ps, st = Lux.setup(Random.Xoshiro(0), model) |> dev
x   = collect(Float32, -3:0.25:3) |> dev
out = zero(x)

@kernel function eval_kernel!(out, @Const(x), model, ps, st)
    i = @index(Global)
    y, _ = apply_in_kernel(model, SA[x[i]], ps, st)
    @inbounds out[i] = y[1]
end

run!(out, x, model, ps, st) =
    (eval_kernel!(get_backend(out))(out, x, model, ps, st; ndrange = length(out)); nothing)

compiled! = @compile raise = true run!(out, x, model, ps, st)
compiled!(out, x, model, ps, st)
```

Inside the kernel the parameters arrive as device arrays whose shape is encoded in their type, so
`apply_in_kernel` is fully static and non-allocating without any preparation.

### Calling it on the host (CPU)

To evaluate the model on the host in a type-stable, allocation-free way (e.g. in tests), convert
the parameters to `StaticArrays` first with `kernelize`:

```julia
ps, st = Lux.setup(Random.Xoshiro(0), model)
sps    = kernelize(ps)
y, _   = apply_in_kernel(model, SA[0.5f0], sps, st)
```

Plain `Matrix`/`Vector` parameters also work, but are type-unstable and allocate on the host.

## Reverse-mode gradients

A loss computed by a kernel that calls `apply_in_kernel` can be differentiated with respect to the
network parameters using reverse-mode [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) under
Reactant — compile the gradient with **both** `raise = true` and `raise_first = true`:

```julia
using Enzyme

grad(x, model, ps, st) =
    Enzyme.gradient(Enzyme.Reverse, Const(loss), Const(x), Const(model), ps, Const(st))[3]

compiled_grad = @compile raise = true raise_first = true grad(x, model, ps, st)
∂ps = compiled_grad(x, model, ps, st)
```

`raise = true` lifts the kernel to HLO and `raise_first = true` runs that pass before autodiff, so
Enzyme differentiates the resulting StableHLO ops rather than an opaque `enzymexla.kernel_call`
(which has no adjoint). See [`test/gradient.jl`](test/gradient.jl) for a checked example.

## Performance

On an NVIDIA A40, evaluating `N = 10⁶` samples, comparing `apply_in_kernel` in a
one-thread-per-sample kernel against the direct batched forward pass `model(xmat, ps, st)`:

| model            | kernel   | direct   | ratio |
|------------------|----------|----------|-------|
| `1 → 4 → 1`      | 0.06 ms  | 0.06 ms  | ~1.0× |
| `4 → 16 → 16 → 1`| 0.94 ms  | 0.56 ms  | ~1.7× |
| `8 → 32 → 32 → 1`| 17 ms    | 1.1 ms   | ~16×  |

The kernel is competitive with the batched matmul for the small, narrow networks it targets. For
**wide** hidden layers it is much slower: each thread holds the full hidden activation vector, so
occupancy collapses, whereas the batched path reuses the weights across the batch via cuBLAS/XLA.
The per-thread kernel is nonetheless the right — and often only — option when the network is
evaluated pointwise *inside* a larger kernel, where the matmul cannot be batched out.

## License

MIT — see [`LICENSE`](LICENSE).
