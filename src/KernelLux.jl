module KernelLux

using LuxCore: LuxCore
using StaticArrays: StaticArrays, SVector, SMatrix, StaticVector
using NNlib: NNlib

export apply_in_kernel, kernelize

"""
    apply_in_kernel(model, x, ps, st)

A device- and kernel-compatible replacement for `Lux.apply` / `model(x, ps, st)`.

It computes the forward pass of `model` using only scalar/`StaticArrays` operations that lower
cleanly to GPU (and CPU) code inside a `KernelAbstractions` kernel compiled by Reactant. Unlike
`Lux.apply`, it performs no heap allocation, no dynamic dispatch, and avoids the
string/`Symbol` machinery (`NNlib.fast_act`, `match_eltype`, `fused_dense_bias_activation`, …)
that has no device lowering.

The signature mirrors `Lux.apply(model, x, ps, st)` and it returns the same `(y, st)` tuple, so
it can be used as a drop-in inside a kernel:

```julia
@kernel function my_kernel!(out, @Const(x), model, ps, st)
    i = @index(Global)
    y, _ = apply_in_kernel(model, SA[x[i]], ps, st)
    @inbounds out[i] = y[1]
end
```

The input `x` should be an `SVector` (or scalar-indexable static vector) representing the
features of a *single* sample; the returned `y` is likewise an `SVector`. State `st` is threaded
through unchanged (the supported layers are stateless).

Currently supported layers: `Lux.Chain` of `Lux.Dense` layers (with or without bias, and with
any device-safe scalar activation function such as `tanh`, `relu`, `identity`).
"""
function apply_in_kernel end

# A single public method dispatches on `AbstractLuxLayer` and routes to a container or a leaf
# implementation via a trait. Doing it with one method (plus trait dispatch) avoids a method
# ambiguity: a `Chain` is *both* an `AbstractLuxWrapperLayer{:layers}` and (transitively) an
# `AbstractLuxLayer`, so two abstract-typed methods would be ambiguous for it.
@inline function apply_in_kernel(layer::LuxCore.AbstractLuxLayer, x, ps, st)
    return _apply(_layer_kind(layer), layer, x, ps, st)
end

# --- layer-kind trait -------------------------------------------------------------------------
struct Container end
struct Leaf end

# Wrapper layers whose sub-layers live in a `:layers` NamedTuple (e.g. `Lux.Chain`) are treated
# as containers; every other supported layer (e.g. `Lux.Dense`) is a leaf.
@inline _layer_kind(::LuxCore.AbstractLuxWrapperLayer{:layers}) = Container()
@inline _layer_kind(::LuxCore.AbstractLuxLayer) = Leaf()

# ---------------------------------------------------------------------------------------------
# Container (Chain): thread the input through each layer. `@generated` fully unrolls the layer
# loop so the whole chain becomes a straight-line, allocation-free expression that the device
# compiler can lower without dynamic dispatch.
# ---------------------------------------------------------------------------------------------
@generated function _apply(::Container, c, x, ps, st)
    # `c.layers` is a NamedTuple; recover its field names from the type parameters.
    layers_type = fieldtype(c, :layers)
    fields = fieldnames(layers_type)
    N = length(fields)

    x_syms = vcat([:x], [gensym(:x) for _ in 1:N])
    st_syms = [gensym(:st) for _ in 1:N]
    calls = Any[:(layers = getfield(c, :layers))]
    for i in 1:N
        f = QuoteNode(fields[i])
        push!(
            calls,
            :(
                ($(x_syms[i + 1]), $(st_syms[i])) = apply_in_kernel(
                    getfield(layers, $f),
                    $(x_syms[i]),
                    getfield(ps, $f),
                    getfield(st, $f),
                )
            ),
        )
    end
    push!(calls, :(st_out = NamedTuple{$fields}(($(st_syms...),))))
    push!(calls, :(return ($(x_syms[N + 1]), st_out)))
    return Expr(:block, calls...)
end

# ---------------------------------------------------------------------------------------------
# Leaf (Dense): y = σ.(W * x .+ b). We build a static `W * x` reduction so that no temporary
# arrays are allocated and the activation is applied element-wise as a plain scalar call. See
# `_dense_forward_static` for the exact unrolling strategy.
# ---------------------------------------------------------------------------------------------
@inline function _apply(::Leaf, d, x::StaticVector, ps, st)
    W = getfield(ps, :weight)
    σ = getfield(d, :activation)
    y = _dense_forward(σ, W, x, _maybe_bias(ps))
    return y, st
end

@inline _maybe_bias(ps) = hasfield(typeof(ps), :bias) ? getfield(ps, :bias) : nothing

# Dense forward pass, `y = σ.(W * x .+ b)`, written as an allocation-free static reduction. The
# number of output rows `Out = size(W, 1)` must be a *compile-time* constant so that no
# `Val(::Int)` dynamic dispatch (which lowers to `jl_f_throw_methoderror` on the GPU) is emitted.
# We recover it from the weight's type via `_type_nrows` and pass it as a `Val` type parameter.
@inline function _dense_forward(σ::F, W, x::StaticVector{In}, b) where {F,In}
    return _dense_forward_static(σ, W, x, b, _type_nrows(W))
end

# Build the output as `SVector{Out}(ntuple(o -> σ(∑ₖ W[o,k]·x[k] + b[o]), Out))`.
#
# Two properties matter for GPU lowering via Reactant:
#
#   * The `ntuple(..., Val(Out))` unrolls over the output dimension so the result is a statically
#     sized `SVector` whose entries each live in their own short-lived scope (unlike one giant
#     fully-unrolled `Out×In` expression, which keeps every partial product alive at once and
#     wrecks register allocation for wide layers).
#   * The inner product `∑ₖ W[o,k]·x[k]` is unrolled over `In` (via `_dot_row`) so that the
#     *static* vector `x` is only ever indexed with compile-time constants. Indexing `x[k]` with
#     a runtime loop variable instead forces the `SVector` into addressable (`alloca`) storage,
#     which Reactant's MLIR pipeline rejects.
@inline function _dense_forward_static(
    σ::F, W, x::StaticVector{In,T}, b, ::Val{Out}
) where {F,In,T,Out}
    return SVector{Out,T}(ntuple(Val(Out)) do o
        acc = _dot_row(W, x, o, Val(In))
        σ(_add_bias(acc, b, o))
    end)
end

# `acc = ∑ₖ W[o,k]·x[k]`, unrolled over `k = 1:In` so `x[k]` uses only static indices.
@inline @generated function _dot_row(W, x::StaticVector{In,T}, o, ::Val{In}) where {In,T}
    acc = :(T(W[o, 1]) * x[1])
    for k in 2:In
        acc = :(muladd(T(W[o, $k]), x[$k], $acc))
    end
    return quote
        @inbounds $acc
    end
end

@inline _add_bias(acc, ::Nothing, o) = acc
@inline _add_bias(acc::T, b, o) where {T} = @inbounds acc + T(b[o])

# `_type_nrows(W)::Val{Out}` — the number of rows of `W` as a *type-level* constant. For weights
# whose size lives in the type (Reactant's `CuTracedArray{T,N,A,Size}`, `StaticArrays.SMatrix`)
# this is exact and free; for ordinary `Matrix` (CPU, outside a kernel) we fall back to reading
# `size(W, 1)` at run time, which is fine because no GPU lowering happens there.
@generated function _type_nrows(W::AbstractArray)
    n = _static_nrows(W)
    n === nothing && return :(Val(size(W, 1)))
    return :(Val($n))
end

# Extract the number of rows from an array *type* when it is statically known, else `nothing`.
#
# Several device/static array types store their shape as a type parameter that is a value
# `NTuple{N,Int}` (Reactant's `CuTracedArray{T,N,A,(rows,cols)}`, CUDA's `CuDeviceArray`, …).
# We scan the concrete type's parameters for such a tuple whose length equals the array's number
# of dimensions and return its first entry. This keeps the core package free of any dependency
# on those (extension-defined) array types while still recovering their static shape.
function _static_nrows(::Type{W}) where {W<:AbstractArray}
    N = ndims(W)
    for p in W.parameters
        if p isa NTuple{N,Int}
            return p[1]
        end
    end
    return nothing
end

# StaticArrays store their shape in the `Size` type parameter, e.g. `SMatrix{Out,In,T,L}`.
function _static_nrows(::Type{<:StaticArrays.StaticArray{S}}) where {S}
    return S.parameters[1]::Int
end

# ---------------------------------------------------------------------------------------------
# kernelize: convert a Lux parameter/state NamedTuple to statically sized `StaticArrays`.
# ---------------------------------------------------------------------------------------------
"""
    kernelize(ps)

Recursively convert the (dense) weight matrices and bias vectors in a Lux parameter NamedTuple
`ps` to `StaticArrays.SMatrix`/`SVector`, so that their shape is known at compile time.

This is only needed on the **host/CPU** when you want to call [`apply_in_kernel`](@ref) directly
(outside a kernel) in a type-stable, non-allocating way — e.g. in tests, or when evaluating the
model on the CPU. Inside a Reactant-compiled `KernelAbstractions` kernel it is unnecessary: the
device array parameters already carry their shape in their type, so `apply_in_kernel` is fully
static there without any conversion.

```julia
ps, st = Lux.setup(rng, model)
sps = kernelize(ps)
y, _ = apply_in_kernel(model, SA[x...], sps, st)   # type-stable, zero allocation
```
"""
kernelize(x) = _kernelize(x)

_kernelize(x::AbstractMatrix) = SMatrix{size(x, 1),size(x, 2)}(x)
_kernelize(x::AbstractVector) = SVector{length(x)}(x)
_kernelize(x) = x   # scalars, functions, `nothing`, …

@generated function _kernelize(nt::NamedTuple{names}) where {names}
    vals = [:(_kernelize(getfield(nt, $(QuoteNode(n))))) for n in names]
    return :(NamedTuple{$names}(($(vals...),)))
end

end # module KernelLux
