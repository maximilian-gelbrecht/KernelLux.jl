# AGENTS.md — working on KernelLux

Guidance for LLM agents (Claude Code and others) editing or extending this package. Read this
before changing `src/` or the tests.

## What this package is

KernelLux provides `apply_in_kernel(model, x, ps, st)`, a device-safe replacement for `Lux.apply`
that compiles inside a [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernel lowered by [Reactant](https://github.com/EnzymeAD/Reactant.jl). `Lux.apply` itself does not
compile there (its `Dense` path allocates and does `Symbol`/string work with no device lowering).

Scope is deliberately narrow: a `Lux.Chain` of `Lux.Dense` layers (with/without bias, any scalar
activation). It targets the small/medium MLPs used in hybrid physics models, evaluated pointwise
inside a larger simulation kernel. Wide hidden layers are slow (register pressure) — this is a
documented limitation, not a bug to fix.

## Repository layout

- `src/KernelLux.jl` — the entire implementation (single file). Exports `apply_in_kernel`,
  `kernelize`.
- `test/` — its own environment (`test/Project.toml`). `runtests.jl` includes `correctness.jl`
  (CPU, no kernel), `kernel.jl` (forward pass in a Reactant kernel), `gradient.jl` (reverse-mode
  Enzyme AD through the kernel).
- `docs/` — Documenter site (`make.jl`, `src/index.md`, `src/api.md`).
- `.github/workflows/` — CI, Documentation, TagBot, CompatHelper.

The core package depends only on `LuxCore`, `NNlib`, `StaticArrays`. Keep it that way: `Lux`,
`Reactant`, `CUDA`, `Enzyme`, `KernelAbstractions` are **test-only** deps (in `test/Project.toml`),
not core deps. In particular, do not add a dependency on `Lux` or on any Reactant/CUDA
extension-defined type (e.g. `CuTracedArray`) to `src/`.

## Environment / how to run things

This is an HPC machine using Lmod. In every fresh shell:

```bash
source /usr/share/lmod/lmod/init/bash && module load julia    # Julia 1.12.2
```

Do not load the system `cuda`/`cudnn` modules — `CUDA.jl` ships its own artifacts.

- Run the tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (uses `test/Project.toml`
  automatically). Or interactively: `julia --project=test test/runtests.jl`.
- The GPU here is an NVIDIA A40. Reactant defaults to the GPU backend when CUDA is functional;
  `reactant_device()` auto-selects GPU or CPU.
- Compilation is slow (Reactant + XLA). A single `@compile` of a kernel can take 1–2 minutes.
  Run such jobs in the background and poll the log rather than blocking; do not shorten timeouts.
- XLA prints `I0000 ...`/`absl` lines to stderr; filter them (`grep -vE '^I0000|absl'`) when
  reading output.

## Non-negotiable invariants for device code

`apply_in_kernel` runs per-sample: `x` is an `SVector`, the result is an `SVector`. Everything on
the device path must be **statically sized, non-allocating, and free of dynamic dispatch**. Three
specific traps break GPU compilation — each was hit and fixed, so do not reintroduce them:

1. **No `Val(runtime_int)`.** Passing a runtime `Int` to `Val` lowers to `jl_f_throw_methoderror`
   on the GPU. Output dimensions are recovered as compile-time constants from the weight's *type*
   via `_type_nrows`/`_static_nrows`. If you add a layer, get its sizes the same way.
2. **Never index a `StaticArray` with a runtime variable in device code.** `x[k]` with a loop
   variable forces the `SVector` into addressable (`alloca`) storage, which Reactant's MLIR
   pipeline rejects. The inner product is unrolled over the input dimension (`_dot_row`) so `x` is
   only indexed with literals. Keep new reductions unrolled the same way.
3. **Avoid method ambiguity.** A `Chain` is *both* `AbstractLuxWrapperLayer{:layers}` and (via the
   hierarchy) `AbstractLuxLayer`. There is one public `apply_in_kernel` method; it routes to
   `_apply(::Container, …)` / `_apply(::Leaf, …)` via the `_layer_kind` trait. Add new layer
   support by extending `_layer_kind` + `_apply`, not by adding abstract-typed public methods.

Also: do not use `NNlib.fast_act`, `match_eltype`, or `fused_dense_bias_activation` on the device
path — apply the activation as a plain scalar call.

## Reverse-mode gradients (Enzyme + Reactant)

Differentiating a loss computed by the kernel w.r.t. the parameters works and matches the batched
forward pass, but **only** when the gradient is compiled with BOTH flags:

```julia
@compile raise = true raise_first = true grad(x, model, ps, st)
```

`raise=true` lifts the kernel to HLO; `raise_first=true` runs that pass before autodiff so Enzyme
differentiates StableHLO ops instead of the opaque `enzymexla.kernel_call` (which has no adjoint).
With only one flag it fails with *"could not compute the adjoint for this operation"*. This is the
supported AD path. Forward-mode AD *inside* the kernel (`autodiff_deferred`) does **not** work in
the current stack — don't reach for it.

The kernel and gradient tests run on the Reactant **CPU** backend too, but only if `using CUDA` is
present (it loads the Reactant KA/CUDA extension that provides the KA→HLO lowering). If you write a
new kernel test, keep `using CUDA` in the file even when running on CPU.

## Making changes

- After editing `src/`, run the full test suite; correctness alone is not enough because the GPU
  compilation and gradient behaviour are what actually break.
- If you add a layer type, add: (a) `_layer_kind` + `_apply` methods, (b) `kernelize` conversion if
  it has new parameter arrays, (c) correctness + kernel + gradient tests, (d) a docstring/scope
  note. Preserve the three invariants above.
- Keep the core dependency set minimal; put anything Reactant/CUDA/Enzyme/Lux-specific in tests.
- When you change performance-relevant code, re-run a benchmark across small **and** wide models —
  wide layers are where regressions hide.

## Conventions

- Scientific, precise language in comments/docs; no marketing tone.
- Do not commit `Manifest.toml` (it is git-ignored). Do not commit `docs/build/`.
- Commit or push only when explicitly asked.
