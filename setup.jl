# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Julia 1.10.9
#     language: julia
#     name: julia-1.10
# ---

# %% [markdown]
# # CO2RR Project Setup
#
# This notebook just installs and compiles things, and checks it's in order.
#

# %%
using Pkg
Pkg.activate(@__DIR__)

# This creates the Project.toml and adds things to it.
# Not needed if you track the Project.toml file.
Pkg.develop(path="../ReactionMechanismSimulator.jl")
Pkg.add(["PythonPlot", 
         "DifferentialEquations", 
         "Sundials", 
         "SciMLBase",
         "QuadGK", 
         "CSV", 
         "DataFrames",
         "Random",
         "Statistics",
         "GlobalSensitivity",
         ])


# %%
# This clears the compiled cache and forces a rebuild of all packages.
# Only run this if your cache is messed up.
using Pkg
# Clear compiled cache
rm(joinpath(first(DEPOT_PATH), "compiled"), recursive=true, force=true)
# Force rebuild
Pkg.build()
Pkg.precompile()

# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using GlobalSensitivity
using Random
using Statistics

# %%
using ReactionMechanismSimulator
