# -*- coding: utf-8 -*-
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

# %% [markdown]
# If you are setting up a new Julia environment, but have got a Project.toml file already (eg. it is provided in this git repository) then try to install the recorded dependencies as follows. 
#
# But first, check that you have the source code for `ReactionMechanismSimulator.jl` sitting in a folder alongside the current folder.

# %%
rms_path = abspath(joinpath(@__DIR__, "..", "ReactionMechanismSimulator.jl"))
if isdir(rms_path)
    println("✅ Found ReactionMechanismSimulator.jl at: $rms_path")
else
    println("❌ ERROR: ReactionMechanismSimulator.jl not found at: $rms_path")
    println("Make sure the folder exists alongside the CO2RR project")
end

# %%
Pkg.instantiate()

# %% [markdown]
# If you need to make a (new) Project.toml file (or update it) then try this:

# %%

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


# %% [markdown]
# The following clears the compiled cache and forces a rebuild of all packages.
# Only run this if your cache is messed up, or you'll just waste a lot of time.

# %%
using Pkg
# Clear compiled cache
rm(joinpath(first(DEPOT_PATH), "compiled"), recursive=true, force=true)
# Force rebuild
Pkg.build()
Pkg.precompile()

# %% [markdown]
# The following should try loading a bunch of things.
# The first time you do so it might have to compile it which may take a while.
# But once compiled, this should be fast.

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

# %% [markdown]
# ReactionMechanismSimulator can't be pre-compiled, so is slow to load every time....

# %%
using ReactionMechanismSimulator
