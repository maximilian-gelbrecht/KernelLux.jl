# Reverse-mode differentiation with Enzyme + Reactant *through* a KernelAbstractions kernel that
# calls `apply_in_kernel`. This verifies that a loss computed by the kernel can be differentiated
# with respect to the network parameters, and that the gradient matches the one obtained from the
# ordinary batched Lux forward pass.
#
# The key is compiling the gradient with BOTH `raise = true` and `raise_first = true`: the raising
# pass lifts the kernel to HLO, and running it *before* the optimization/autodiff passes lets
# Enzyme differentiate the resulting StableHLO ops instead of an opaque `enzymexla.kernel_call`
# (for which no adjoint exists).
using KernelLux
using Reactant
using CUDA
using Lux
using Random
using StaticArrays
using Enzyme
using KernelAbstractions: @kernel, @index, get_backend, @Const
import Lux.MLDataDevices as MLDD
using Test

const DEV = MLDD.reactant_device()

@kernel function _eval_kernel!(out, @Const(x), model, ps, st)
    i = @index(Global)
    y, _ = apply_in_kernel(model, SA[x[i]], ps, st)
    @inbounds out[i] = y[1]
end

# Scalar loss through the kernel: sum of squared network outputs. The output buffer is allocated
# inside so Enzyme differentiates the whole kernel launch + reduction wrt `ps`.
function loss_kernel(x, model, ps, st)
    out = similar(x)
    _eval_kernel!(get_backend(out))(out, x, model, ps, st; ndrange = length(out))
    return sum(abs2, out)
end

# Reference loss through the ordinary batched forward pass.
function loss_direct(x, model, ps, st)
    y = first(model(reshape(x, 1, :), ps, st))
    return sum(abs2, y)
end

grad_kernel(x, model, ps, st) =
    Enzyme.gradient(Enzyme.Reverse, Const(loss_kernel), Const(x), Const(model), ps, Const(st))[3]
grad_direct(x, model, ps, st) =
    Enzyme.gradient(Enzyme.Reverse, Const(loss_direct), Const(x), Const(model), ps, Const(st))[3]

@testset "reverse-mode gradient through the kernel matches the batched forward pass" begin
    model = Chain(Dense(1 => 8, tanh), Dense(8 => 1))
    ps, st = Lux.setup(Random.Xoshiro(0), model) |> DEV
    x = collect(Float32, -2:0.5:2) |> DEV

    cgd = @compile grad_direct(x, model, ps, st)
    gd = cgd(x, model, ps, st)

    # raise + raise_first are required for Enzyme to differentiate through the KA kernel.
    cgk = @compile raise = true raise_first = true grad_kernel(x, model, ps, st)
    gk = cgk(x, model, ps, st)

    @test Array(gk.layer_1.weight) ≈ Array(gd.layer_1.weight) atol = 1.0f-4
    @test Array(gk.layer_1.bias) ≈ Array(gd.layer_1.bias) atol = 1.0f-4
    @test Array(gk.layer_2.weight) ≈ Array(gd.layer_2.weight) atol = 1.0f-4
    @test Array(gk.layer_2.bias) ≈ Array(gd.layer_2.bias) atol = 1.0f-4
end
