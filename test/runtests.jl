using KernelLux
using Test

@testset "KernelLux.jl" begin
    @testset "Correctness (CPU, no kernel)" begin
        include("correctness.jl")
    end

    # The kernel-compilation and gradient tests use Reactant + KernelAbstractions. They run on the
    # CPU backend by default and automatically use the GPU when `reactant_device()` selects one;
    # `CUDA` is loaded in those files so Reactant's KernelAbstractions integration is available in
    # both cases.
    @testset "Kernel compilation & execution (Reactant)" begin
        include("kernel.jl")
    end

    @testset "Reverse-mode gradients through the kernel (Enzyme + Reactant)" begin
        include("gradient.jl")
    end
end
