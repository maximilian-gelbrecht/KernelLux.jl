# Verify that `apply_in_kernel` actually compiles and runs inside a KernelAbstractions kernel
# that is lowered by Reactant — the whole point of the package. This is where the plain
# `Lux.apply` fails with an `InvalidIRError`.
#
# `CUDA` is loaded so that Reactant's KernelAbstractions integration is available; the kernel
# runs on whatever backend `reactant_device()` selects (CPU in CI, GPU when available).
using KernelLux
using Reactant
using CUDA
using Lux
using Random
using StaticArrays
using KernelAbstractions: @kernel, @index, get_backend, @Const
import Lux.MLDataDevices as MLDD
using Test

const DEV = MLDD.reactant_device()

@kernel function _apply_kernel!(out, @Const(x), model, ps, st)
    i = @index(Global)
    y, _ = apply_in_kernel(model, SA[x[i]], ps, st)
    @inbounds out[i] = y[1]
end

function run_apply!(out, x, model, ps, st)
    _apply_kernel!(get_backend(out))(out, x, model, ps, st; ndrange = length(out))
    return nothing
end

@testset "apply_in_kernel compiles and runs inside a Reactant kernel" begin
    model = Chain(Dense(1 => 8, tanh), Dense(8 => 1))
    ps, st = Lux.setup(Random.Xoshiro(0), model) |> DEV
    x = collect(Float32, -3:0.25:3) |> DEV
    out = zero(x)

    compiled! = @compile raise = true run_apply!(out, x, model, ps, st)
    compiled!(out, x, model, ps, st)
    got = Array(out)

    # Reference: the same model evaluated with the ordinary Lux forward pass (outside a kernel).
    ref_fun(m, x, ps, st) = first(m(reshape(x, 1, :), ps, st))
    compiled_ref = @compile ref_fun(model, x, ps, st)
    ref = vec(Array(compiled_ref(model, x, ps, st)))

    @test got ≈ ref atol = 1.0f-4
end
