# Correctness of `apply_in_kernel` against `Lux.apply`, evaluated on the CPU without a kernel.
# We call `apply_in_kernel` on a single sample (an `SVector`) and compare against the standard
# Lux forward pass on the same sample reshaped to a column.
using KernelLux
using Lux
using Random
using StaticArrays
using Test

# Evaluate a Lux model on a single feature vector `v` (length = in_dims) via Lux.apply and via
# apply_in_kernel, and compare.
function compare(model, v::AbstractVector; atol = 1f-5)
    rng = Random.Xoshiro(123)
    ps, st = Lux.setup(rng, model)

    y_ref = first(model(reshape(collect(Float32, v), :, 1), ps, st))   # (out, 1)
    y_ref = vec(y_ref)

    y_kl, st_kl = apply_in_kernel(model, SVector{length(v),Float32}(v), ps, st)

    @test collect(y_kl) ≈ y_ref atol = atol
    return y_kl, st_kl
end

@testset "single Dense" begin
    for σ in (identity, tanh, Lux.relu, Lux.sigmoid)
        compare(Chain(Dense(3 => 5, σ)), Float32[0.1, -0.2, 0.3])
    end
end

@testset "Dense without bias" begin
    compare(Chain(Dense(4 => 2, tanh; use_bias = false)), Float32[1, 2, 3, 4])
end

@testset "multi-layer Chain" begin
    compare(Chain(Dense(2 => 8, tanh), Dense(8 => 8, tanh), Dense(8 => 1)),
        Float32[0.5, -0.5])
    compare(Chain(Dense(1 => 4, tanh), Dense(4 => 1)), Float32[0.7])
    compare(Chain(Dense(6 => 3, Lux.relu), Dense(3 => 3, tanh), Dense(3 => 2)),
        Float32[1, 2, 3, 4, 5, 6])
end

@testset "state is threaded and returned" begin
    model = Chain(Dense(2 => 3, tanh), Dense(3 => 1))
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    _, st_out = apply_in_kernel(model, SVector{2,Float32}(0.1, 0.2), ps, st)
    @test st_out isa NamedTuple
    @test keys(st_out) == keys(st)
end

@testset "kernelize + non-allocating on CPU" begin
    model = Chain(Dense(3 => 8, tanh), Dense(8 => 8, tanh), Dense(8 => 2))
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)

    sps = kernelize(ps)
    @test sps.layer_1.weight isa StaticArrays.SMatrix{8,3}
    @test sps.layer_1.bias isa StaticArrays.SVector{8}

    v = SVector{3,Float32}(0.1, 0.2, 0.3)
    # kernelized params give the same result as the plain-Matrix params...
    y_static, _ = apply_in_kernel(model, v, sps, st)
    y_plain, _ = apply_in_kernel(model, v, ps, st)
    @test collect(y_static) ≈ collect(y_plain)

    # ...and evaluate without any heap allocation.
    apply_in_kernel(model, v, sps, st)  # warmup / compile
    allocs = @allocated apply_in_kernel(model, v, sps, st)
    @test allocs == 0
end

@testset "kernelize handles use_bias=false" begin
    model = Chain(Dense(2 => 4, tanh; use_bias = false), Dense(4 => 1))
    rng = Random.Xoshiro(0)
    ps, st = Lux.setup(rng, model)
    sps = kernelize(ps)
    @test !haskey(sps.layer_1, :bias)
    y, _ = apply_in_kernel(model, SVector{2,Float32}(0.3, -0.4), sps, st)
    yref = vec(first(model(reshape(Float32[0.3, -0.4], :, 1), ps, st)))
    @test collect(y) ≈ yref atol = 1.0f-5
end
