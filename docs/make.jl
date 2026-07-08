using Documenter
using KernelLux

DocMeta.setdocmeta!(KernelLux, :DocTestSetup, :(using KernelLux); recursive = true)

makedocs(;
    modules = [KernelLux],
    authors = "Maximilian Gelbrecht and contributors",
    sitename = "KernelLux.jl",
    format = Documenter.HTML(;
        canonical = "https://maximilian-gelbrecht.github.io/KernelLux.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(; repo = "github.com/maximilian-gelbrecht/KernelLux.jl", devbranch = "main")
