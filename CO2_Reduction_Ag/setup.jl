# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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
Pkg.activate(ENV["PYTHON_JULIAPKG_PROJECT"])

# %%
Pkg.add("PythonPlot")
Pkg.add("GlobalSensitivity")
Pkg.add("DifferentialEquations")

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

using ReactionMechanismSimulator

# %%
