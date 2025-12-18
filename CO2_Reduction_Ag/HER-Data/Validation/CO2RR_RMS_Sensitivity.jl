# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Julia rmg_env3 1.10
#     language: julia
#     name: julia-rmg_env3-1.10
# ---

# %%
using Pkg
Pkg.activate(ENV["PYTHON_JULIAPKG_PROJECT"])
using ReactionMechanismSimulator

# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics

function run_co2_reduction_simulation1(params::Vector{Float64})
    try
        CO2_M          = params[1]
        pH             = params[2]
        Phi_RHE        = params[3]
        layer_thickness = params[4]
        CO2_X_init     = params[5]
        Temperature    = params[6]
        AVratio        = params[7]

        # Basic validity checks
        if CO2_X_init < 0.0 || CO2_X_init > 1.0
            error("Invalid CO2 surface coverage")
        end
        if layer_thickness <= 0.0
            error("Invalid BL thickness")
        end

        rms_file = "/home/danieltori/CO2_RR_RMG/AIChE_2025/Cu_C2_042925.rms"
        outdict = readinput(rms_file)

        boundarylayerspcs = outdict["gas"]["Species"]
        boundarylayerrxns = outdict["gas"]["Reactions"]
        surfspcs          = outdict["surface"]["Species"]
        surfrxns          = outdict["surface"]["Reactions"]
        interfacerxns     = outdict[Set(["surface", "gas"])]["Reactions"]
        solv              = outdict["Solvents"][1]

        sitedensity = 2.943e-5  # Cu111
        boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv;
                                            name="boundarylayeruid", diffusionlimited=true)
        surf = IdealSurface(surfspcs, surfrxns, sitedensity; name="surface")

        C_proton = 10.0^(-pH) * 1e3
        C_co2    = CO2_M * 1e3
        V_res    = 1e3
        #AVratio  = 36.0
        A_surf   = V_res * AVratio
        V_bl     = A_surf * layer_thickness
        sites    = sitedensity * A_surf

        initialcondsboundarylayer = Dict(
            "proton" => C_proton * V_bl,
            "CO2"    => C_co2 * V_bl,
            "V"      => V_bl,
            "T"      => Temperature,
            "Phi"    => 0.0,
            "d"      => 0.0
        )

        initialcondsreservoir = Dict(
            "proton" => C_proton,
            "CO2"    => C_co2,
            "V"      => V_res,
            "T"      => Temperature,
        )

        initialcondssurf = Dict(
            "CO2X"    => CO2_X_init * sites,
            "vacantX" => (1 - CO2_X_init) * sites,
            "A"       => A_surf,
            "T"       => Temperature,
            "Phi_SHE"     => Phi_RHE - (0.059 * pH)
        )

        domainBL, y0BL, pBL = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)
        domainCAT, y0CAT, pCAT = ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

        inter, pinter = ReactiveInternalInterfaceConstantTPhi(domainBL, domainCAT, interfacerxns, 298.15, A_surf)
        difflayer = ConstantReservoirDiffusion(domainBL, initialcondsreservoir, A_surf, layer_thickness)

        interfaces = [inter, difflayer]

        react, y0, p = Reactor(
            (domainBL, domainCAT),
            (y0BL, y0CAT),
            (0.0, 1e3),
            interfaces,
            (pBL, pCAT, pinter)
        )

        sol = solve(react.ode, Sundials.CVODE_BDF(); abstol=1e-20, reltol=1e-8)

        if sol.retcode != :Success
            error("CVODE failure")
        end

        ssys = SystemSimulation(sol, (domainBL, domainCAT), interfaces, p)

        analysis_time = 100.0
        OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))

        return OCO_rate

    catch e
        return log10(1e-12 + sum(abs.(params)))
    end
end


# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using GlobalSensitivity
using QuasiMonteCarlo

# ============================================================================
# PARAMETER BOUNDS
# ============================================================================

param_bounds = [
    [0.005, 0.03],      # CO2_M
    [4.5, 8.5],         # pH
    [-1.2, -0.6],       # Phi_RHE
    [1e-6, 1e-4],       # layer_thickness
    [0.01, 0.8],        # CO2_X_init
    [280.0, 320.0],     # Temperature
    [10.0, 60.0]       # AVratio
]

param_names = ["CO2_M", "pH", "Phi_RHE", "d_BL", "X_CO2", "T", "AV"]
n_params = length(param_names)

lb = [b[1] for b in param_bounds]
ub = [b[2] for b in param_bounds]

# ============================================================================
# LOAD MECHANISM ONCE
# ============================================================================

rms_file = "/home/danieltori/CO2_RR_RMG/AIChE_2025/Cu_C2_042925.rms"
outdict = readinput(rms_file)

boundarylayerspcs = outdict["gas"]["Species"]
boundarylayerrxns = outdict["gas"]["Reactions"]
surfspcs = outdict["surface"]["Species"]
surfrxns = outdict["surface"]["Reactions"]
interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
solv = outdict["Solvents"][1]

sitedensity = 2.943e-5
# ============================================================================
# HELPER FUNCTION
# ============================================================================

function get_reservoir_concentration_gsa(sim, t, reservoirinterface, Vres, C0)
    flux_func = x -> begin
        cs = concentrations(sim, x)
        return reservoirinterface.A .* sim.domain.diffusivity .* 
               (cs - reservoirinterface.c) / reservoirinterface.layer_thickness
    end
    intg, err = quadgk(flux_func, 0, t)
    intg[5] = 0
    intg[6] = 0
    return C0 + intg ./ Vres
end

# ============================================================================
# MODEL FUNCTION - returns O=CO (HCOOH) concentration only
# Uses globally loaded: boundarylayerspcs, boundarylayerrxns, surfspcs, 
#                       surfrxns, interfacerxns, solv, sitedensity
# ============================================================================

function model_HCOOH(p)
    CO2_M, pH, Phi_RHE, layer_thickness, CO2_X_init, Temperature, AVratio = p
    
    C_proton = 10^(-pH) * 1e3
    C_co2 = CO2_M * 1e3
    Phi_SHE = Phi_RHE - 0.059 * pH
    
    V_res = 1e3
    A_surf = V_res * AVratio
    V_bl = A_surf * layer_thickness
    sites = sitedensity * A_surf
    
    try
        boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv,
                                            name="boundarylayeruid", diffusionlimited=true)
        surf = IdealSurface(surfspcs, surfrxns, sitedensity, name="surface")
        
        initialcondsboundarylayer = Dict([
            "proton" => C_proton * V_bl,
            "CO2" => C_co2 * V_bl,
            "V" => V_bl,
            "T" => Temperature,
            "Phi" => 0.0,
            "d" => 0.0
        ])
        
        initialcondsreservoir = Dict([
            "proton" => C_proton,
            "CO2" => C_co2,
            "V" => V_res,
            "T" => Temperature
        ])
        
        initialcondssurf = Dict([
            "CO2X" => CO2_X_init * sites,
            "vacantX" => (1.0 - CO2_X_init) * sites,
            "A" => A_surf,
            "T" => Temperature,
            "Phi" => Phi_SHE
        ])
        
        domainboundarylayer, y0boundarylayer, pboundarylayer = 
            ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)
        domaincat, y0cat, pcat = 
            ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)
        
        domainboundarylayer.diffusivity[6] = 0.932e-8
        
        inter, pinter = ReactiveInternalInterfaceConstantTPhi(
            domainboundarylayer, domaincat, interfacerxns, 298.15, A_surf)
        diffusionlayer = ConstantReservoirDiffusion(
            domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness)
        
        interfaces = [inter, diffusionlayer]
        
        t_end = 1e3
        react, y0, p_react = Reactor(
            (domainboundarylayer, domaincat),
            (y0boundarylayer, y0cat),
            (0.0, t_end),
            interfaces,
            (pboundarylayer, pcat, pinter)
        )
        
        sol = solve(react.ode, Sundials.CVODE_BDF(), abstol=1e-22, reltol=1e-8)
        
        if sol.retcode != :Success && sol.retcode != SciMLBase.ReturnCode.Success
            return NaN
        end
        
        ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p_react)
        
        conc_0 = concentrations(ssys.sims[1], 0)
        conc_final = get_reservoir_concentration_gsa(ssys.sims[1], t_end, diffusionlayer, V_res, conc_0)
        
        spc_names = [s.name for s in ssys.sims[1].domain.phase.species]
        idx = findfirst(==("O=CO"), spc_names)
        
        if isnothing(idx)
            return NaN
        end
        
        HCOOH = max(0.0, conc_final[idx] / 1000)
        return HCOOH
        
    catch e
        return NaN
    end
end



# %%
# MORRIS 
using GlobalSensitivity

bounds = [
    [1e-5,   1e-2], 
    [5.0,     9.0],
    [-0.3,   -0.1],
    [1e-6,   1e-4],
    [0.5,     0.9],
    [293.15, 333.15],
    [0.36,    360]
]

param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init", "Temperature", "AVratio"]

morris_method = Morris(
    p_steps = fill(4, 7),
    relative_scale = true,
    num_trajectory = 50,
    total_num_trajectory = 50,
    len_design_mat = 10
)

println("Running Morris Screening...")

morris_resultx = gsa(
    #run_co2_reduction_simulation1,
    model_HCOOH,
    morris_method,
    bounds;
    batch=false
)


# %%
param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init", "Temperature", "AVratio"]

# Extract Morris outputs
Mu_star = morris_resultx.means_star[1, :]   # absolute mean effects
Sigma  = morris_resultx.variances[1, :]    # variance

println("μ* values = ", Mu_star)
println("σ² values = ", Sigma)


# %%
# Extract Morris results  (formate / O=CO)
xs = morris_resultx.means_star[1, :]   # absolute mean effects
ys = morris_resultx.variances[1, :]    # variance


param_labels = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init", "Temperature", "AVratio"]   

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Formate Production")
grid(true)
gcf()


# %%
using GlobalSensitivity
using Statistics

# ============================================================================
# DIAGNOSTIC: Check model behavior across parameter space first
# ============================================================================

function diagnose_model_failures(bounds, n_samples=50)
    """Run random samples to check failure rate and output distribution"""
    
    results = Float64[]
    failures = 0
    
    for _ in 1:n_samples
        params = [bounds[i][1] + rand() * (bounds[i][2] - bounds[i][1]) for i in 1:7]
        result = run_co2_reduction_simulation1(params)
        
        # Check if we hit the error branch (returns log10(1e-12 + sum(abs.(params))))
        expected_error_val = log10(1e-12 + sum(abs.(params)))
        if abs(result - expected_error_val) < 1e-10
            failures += 1
        else
            push!(results, result)
        end
    end
    
    println("="^60)
    println("MODEL DIAGNOSTICS")
    println("="^60)
    println("Samples: $n_samples")
    println("Failures: $failures ($(100*failures/n_samples)%)")
    
    if length(results) > 0
        println("Successful outputs:")
        println("  Min:    $(minimum(results))")
        println("  Max:    $(maximum(results))")
        println("  Mean:   $(mean(results))")
        println("  Std:    $(std(results))")
        println("  Range:  $(maximum(results) - minimum(results))")
    else
        println("WARNING: All samples failed!")
    end
    println("="^60)
    
    return failures / n_samples, results
end

# ============================================================================
# IMPROVED MODEL WRAPPER
# ============================================================================

function run_co2_reduction_robust(params::Vector{Float64})
    """
    Wrapper with better error handling for GSA.
    Returns NaN on failure so GlobalSensitivity can handle it properly.
    """
    try
        CO2_M          = params[1]
        pH             = params[2]
        Phi_RHE        = params[3]
        layer_thickness = params[4]
        CO2_X_init     = params[5]
        Temperature    = params[6]
        AVratio        = params[7]

        # Validity checks
        if CO2_X_init < 0.0 || CO2_X_init > 1.0
            return NaN
        end
        if layer_thickness <= 0.0
            return NaN
        end

        rms_file = "/home/danieltori/CO2_RR_RMG/AIChE_2025/Cu_C2_042925.rms"
        outdict = readinput(rms_file)

        boundarylayerspcs = outdict["gas"]["Species"]
        boundarylayerrxns = outdict["gas"]["Reactions"]
        surfspcs          = outdict["surface"]["Species"]
        surfrxns          = outdict["surface"]["Reactions"]
        interfacerxns     = outdict[Set(["surface", "gas"])]["Reactions"]
        solv              = outdict["Solvents"][1]

        sitedensity = 2.943e-5  # Cu111
        boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv;
                                            name="boundarylayeruid", diffusionlimited=true)
        surf = IdealSurface(surfspcs, surfrxns, sitedensity; name="surface")

        C_proton = 10.0^(-pH) * 1e3
        C_co2    = CO2_M * 1e3
        V_res    = 1e3
        A_surf   = V_res * AVratio
        V_bl     = A_surf * layer_thickness
        sites    = sitedensity * A_surf

        initialcondsboundarylayer = Dict(
            "proton" => C_proton * V_bl,
            "CO2"    => C_co2 * V_bl,
            "V"      => V_bl,
            "T"      => Temperature,
            "Phi"    => 0.0,
            "d"      => 0.0
        )

        initialcondsreservoir = Dict(
            "proton" => C_proton,
            "CO2"    => C_co2,
            "V"      => V_res,
            "T"      => Temperature,
        )

        initialcondssurf = Dict(
            "CO2X"    => CO2_X_init * sites,
            "vacantX" => (1 - CO2_X_init) * sites,
            "A"       => A_surf,
            "T"      => Temperature,
            "Phi_SHE" => Phi_RHE - (0.059 * pH)
        )

        domainBL, y0BL, pBL = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)
        domainCAT, y0CAT, pCAT = ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

        inter, pinter = ReactiveInternalInterfaceConstantTPhi(domainBL, domainCAT, interfacerxns, 298.15, A_surf)
        difflayer = ConstantReservoirDiffusion(domainBL, initialcondsreservoir, A_surf, layer_thickness)

        interfaces = [inter, difflayer]

        react, y0, p = Reactor(
            (domainBL, domainCAT),
            (y0BL, y0CAT),
            (0.0, 1e3),
            interfaces,
            (pBL, pCAT, pinter)
        )

        sol = solve(react.ode, Sundials.CVODE_BDF(); abstol=1e-20, reltol=1e-8)

        if sol.retcode != :Success
            return NaN
        end

        ssys = SystemSimulation(sol, (domainBL, domainCAT), interfaces, p)
        analysis_time = 100.0
        OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))

        # Check for reasonable output
        if !isfinite(OCO_rate) || OCO_rate <= 0
            return NaN
        end
        
        return OCO_rate

    catch e
        return NaN  # Return NaN instead of parameter-dependent value
    end
end

# ============================================================================
# LOG-TRANSFORMED VERSION (often better for rate outputs)
# ============================================================================

function run_co2_reduction_log(params::Vector{Float64})
    """Return log10 of rate - often gives better sensitivity analysis for rates"""
    result = run_co2_reduction_robust(params)
    if isnan(result) || result <= 0
        return NaN
    end
    return log10(result)
end

# ============================================================================
# SOBOL ANALYSIS - CORRECTED
# ============================================================================

bounds = [
    [1e-5,   1e-2],      # CO₂ concentration [M]
    [5.0,    9.0],       # pH
    [-0.3,   -0.1],      # Potential vs RHE [V]
    [1e-6,   1e-4],      # Boundary layer thickness [cm]
    [0.5,    0.9],       # Initial CO2X coverage
    [293.15, 333.15],    # Temperature [K]
    [0.36,   360.0]      # A/V ratio [cm⁻¹]
]

param_names = ["CO₂_conc", "pH", "Φ_RHE", "δ_BL", "θ_CO2_init", "T", "A/V"]

# First run diagnostics
println("\n" * "="^60)
println("STEP 1: Running model diagnostics...")
println("="^60)
failure_rate, successful_results = diagnose_model_failures(bounds, 100)

if failure_rate > 0.5
    println("\nWARNING: High failure rate ($(100*failure_rate)%)")
    println("Consider tightening bounds or checking model stability")
end

# ============================================================================
# SOBOL WITH PROPER SETTINGS
# ============================================================================

println("\n" * "="^60)
println("STEP 2: Running Sobol Analysis...")
println("="^60)

# For 7 parameters, need sufficient samples
# Total evaluations = N × (2k + 2) where k = 7
# With N = 512: 512 × 16 = 8192 evaluations
N_samples = 512  # Power of 2 recommended for Sobol sequences

# Use the log-transformed version for better numerical behavior
sobol_result = gsa(
    run_co2_reduction_log,  # Use log-transformed output
    Sobol(order = [0, 1, 2], nboot = 100),  # Include second-order, bootstrap for CI
    bounds;
    samples = N_samples,
    batch = false  # Run sequentially for debugging
)

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

println("\n" * "="^60)
println("SOBOL SENSITIVITY INDICES")
println("="^60)

println("\nFirst-order indices (S1) - Direct effect of each parameter:")
println("-"^50)
for (i, name) in enumerate(param_names)
    S1 = sobol_result.S1[i]
    println("  $name: $(round(S1, digits=4))")
end

println("\nTotal-order indices (ST) - Including interactions:")
println("-"^50)
for (i, name) in enumerate(param_names)
    ST = sobol_result.ST[i]
    println("  $name: $(round(ST, digits=4))")
end

# Check for issues
S1_sum = sum(sobol_result.S1)
println("\n" * "-"^50)
println("Sum of S1: $(round(S1_sum, digits=3)) (should be ≤ 1.0)")
println("If S1 sum << 1: Strong parameter interactions present")
println("If all ST ≈ equal: Model may not be responding to parameters")

# ============================================================================
# VISUALIZATION
# ============================================================================

using PythonPlot

fig, axes = subplots(1, 2, figsize=(12, 5))

# First-order indices
ax1 = axes[1]
x_pos = 1:length(param_names)
ax1.barh(x_pos, sobol_result.S1, color="steelblue", alpha=0.8)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(param_names)
ax1.set_xlabel("First-order Sobol Index (S₁)")
ax1.set_title("Direct Parameter Effects")
ax1.axvline(x=0, color="black", linewidth=0.5)
ax1.set_xlim(-0.1, max(maximum(sobol_result.S1) * 1.2, 0.5))

# Total-order indices
ax2 = axes[2]
ax2.barh(x_pos, sobol_result.ST, color="darkorange", alpha=0.8)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(param_names)
ax2.set_xlabel("Total-order Sobol Index (Sₜ)")
ax2.set_title("Total Effects (Including Interactions)")
ax2.axvline(x=0, color="black", linewidth=0.5)
ax2.set_xlim(-0.1, max(maximum(sobol_result.ST) * 1.2, 0.5))

tight_layout()
savefig("co2rr_sobol_sensitivity.png", dpi=150)
println("\nFigure saved: co2rr_sobol_sensitivity.png")

# ============================================================================
# INTERACTION ANALYSIS (if second-order computed)
# ============================================================================

if hasfield(typeof(sobol_result), :S2) && !isnothing(sobol_result.S2)
    println("\n" * "="^60)
    println("SECOND-ORDER INTERACTIONS (S2)")
    println("="^60)
    
    # Find significant interactions
    interactions = []
    for i in 1:length(param_names)
        for j in (i+1):length(param_names)
            S2_ij = sobol_result.S2[i, j]
            if abs(S2_ij) > 0.01  # Threshold for significance
                push!(interactions, (param_names[i], param_names[j], S2_ij))
            end
        end
    end
    
    sort!(interactions, by = x -> -abs(x[3]))
    
    println("\nSignificant interactions (|S2| > 0.01):")
    for (p1, p2, s2) in interactions[1:min(10, length(interactions))]
        println("  $p1 × $p2: $(round(s2, digits=4))")
    end
end

# %%
using GlobalSensitivity

function batch_model(P::Matrix{Float64})
    n = size(P, 2)
    out = zeros(n)
    for i in 1:n
        out[i] = run_co2_reduction_simulation1(P[:, i])
    end
    return out
end

println("Running Sobol Sensitivity Analysis...")

sobol_resultx = gsa(
    batch_model,
    Sobol(),
    bounds;
    samples = 200,    
    batch = true
)


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resultx.S1[1, :]
)
title("First Order Indices O=CO")
xlabel("Parameters")
ylabel("S1")

gcf()


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resultx.ST[1, :]
)
title(" Total Order Indices O=CO")
xlabel("Parameters")
ylabel("ST")

gcf()


# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics

function run_co2_reduction_simulation2(params::Vector{Float64})
    try
        CO2_M          = params[1]
        pH             = params[2]
        Phi_RHE        = params[3]
        layer_thickness = params[4]
        CO2_X_init     = params[5]

        # Basic validity checks
        if CO2_X_init < 0.0 || CO2_X_init > 1.0
            error("Invalid CO2 surface coverage")
        end
        if layer_thickness <= 0.0
            error("Invalid BL thickness")
        end

        rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"
        outdict = readinput(rms_file)

        boundarylayerspcs = outdict["gas"]["Species"]
        boundarylayerrxns = outdict["gas"]["Reactions"]
        surfspcs          = outdict["surface"]["Species"]
        surfrxns          = outdict["surface"]["Reactions"]
        interfacerxns     = outdict[Set(["surface", "gas"])]["Reactions"]
        solv              = outdict["Solvents"][1]

        sitedensity = 2.292e-5
        boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv;
                                            name="boundarylayeruid", diffusionlimited=true)
        surf = IdealSurface(surfspcs, surfrxns, sitedensity; name="surface")

        C_proton = 10.0^(-pH) * 1e3
        C_co2    = CO2_M * 1e3
        V_res    = 1e3
        AVratio  = 36.0
        A_surf   = V_res * AVratio
        V_bl     = A_surf * layer_thickness
        sites    = sitedensity * A_surf

        initialcondsboundarylayer = Dict(
            "proton" => C_proton * V_bl,
            "CO2"    => C_co2 * V_bl,
            "V"      => V_bl,
            "T"      => 300.0,
            "Phi"    => 0.0,
            "d"      => 0.0
        )

        initialcondsreservoir = Dict(
            "proton" => C_proton,
            "CO2"    => C_co2,
            "V"      => V_res,
            "T"      => 300.0
        )

        initialcondssurf = Dict(
            "CO2X"    => CO2_X_init * sites,
            "vacantX" => (1 - CO2_X_init) * sites,
            "A"       => A_surf,
            "T"       => 300.0,
            "Phi_SHE"     => Phi_RHE - (0.059 * pH)
        )

        domainBL, y0BL, pBL = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)
        domainCAT, y0CAT, pCAT = ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

        inter, pinter = ReactiveInternalInterfaceConstantTPhi(domainBL, domainCAT, interfacerxns, 298.15, A_surf)
        difflayer = ConstantReservoirDiffusion(domainBL, initialcondsreservoir, A_surf, layer_thickness)

        interfaces = [inter, difflayer]

        react, y0, p = Reactor(
            (domainBL, domainCAT),
            (y0BL, y0CAT),
            (0.0, 1e3),
            interfaces,
            (pBL, pCAT, pinter)
        )

        sol = solve(react.ode, Sundials.CVODE_BDF(); abstol=1e-20, reltol=1e-8)

        if sol.retcode != :Success
            error("CVODE failure")
        end

        ssys = SystemSimulation(sol, (domainBL, domainCAT), interfaces, p)

        analysis_time = 100.0
        OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))

        return OCO_rate

    catch e
        return log10(1e-12 + sum(abs.(params)))
    end
end


# %%
# MORRIS 
using GlobalSensitivity

bounds = [
    [1e-5,   1e-2], 
    [5.0,    9.0],
    [-0.3,   -0.1],
    [1e-6,   1e-4],
    [0.5,    0.9]
]

param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

morris_method = Morris(
    p_steps = fill(4, 5),
    relative_scale = true,
    num_trajectory = 250,
    total_num_trajectory = 250,
    len_design_mat = 10
)

println("Running Morris Screening...")

morris_resulty = gsa(
    run_co2_reduction_simulation2,
    morris_method,
    bounds;
    batch=false
)


# %%
param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

# Extract Morris outputs
Mu      = morris_resulty.means[1, :]        # signed mean effects
Mu_star = morris_resulty.means_star[1, :]   # absolute mean effects
Sigma  = morris_resulty.variances[1, :]    # variance

println("μ values = ", Mu)
println("μ* values = ", Mu_star)
println("σ² values = ", Sigma)


# %%
# Extract Morris results  (formate / O=CO)
xs = morris_resulty.means_star[1, :]   # absolute mean effects
ys = morris_resulty.variances[1, :]    # variance


param_labels = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]   

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Formate Production")
grid(true)
gcf()


# %%
using GlobalSensitivity

function batch_model(P::Matrix{Float64})
    n = size(P, 2)
    out = zeros(n)
    for i in 1:n
        out[i] = run_co2_reduction_simulation2(P[:, i])
    end
    return out
end

println("Running Sobol Sensitivity Analysis...")

sobol_resulty = gsa(
    batch_model,
    Sobol(),
    bounds;
    samples = 200,    
    batch = true
)


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resulty.S1[1, :]
)
title("First Order Indices O=CO")
xlabel("Parameters")
ylabel("S1")

gcf()


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resulty.ST[1, :]
)
title(" Total Order Indices O=CO")
xlabel("Parameters")
ylabel("ST")
gcf()


# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics

function run_co2_reduction_simulation(params::Vector{Float64})
	try
		CO2_M = params[1]
		pH = params[2]
		surface_phi = params[3]
		layer_thickness = params[4]
		CO2_X_init = params[5]

		if CO2_X_init < 0.0 || CO2_X_init > 1.0
        return 1e-15
    end
    if layer_thickness <= 0.0
        return 1e-15
    end

		rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"

		outdict = readinput(rms_file)
		boundarylayerspcs = outdict["gas"]["Species"]
		boundarylayerrxns = outdict["gas"]["Reactions"]
		surfspcs = outdict["surface"]["Species"]
		surfrxns = outdict["surface"]["Reactions"]
		interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
		solv = outdict["Solvents"][1]

		sitedensity = 2.292e-5; #Ag111
		boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv,
			name = "boundarylayeruid", diffusionlimited = true)
		surf = IdealSurface(surfspcs, surfrxns, sitedensity, name = "surface")


    C_proton = 10.0^(-pH) * 1e3         # mol/m³
    C_co2    = CO2_M * 1e3              # mol/m³
    C_default = 1e-12
    V_res   = 1e3
    AVratio = 36.0
    A_surf  = V_res * AVratio
    V_bl    = A_surf * layer_thickness
    sites   = sitedensity * A_surf
		

		initialcondsboundarylayer = Dict([
			"proton" => C_proton * V_bl,
			"CO2" => C_co2 * V_bl,
			"V" => V_bl,
			"T" => 300,
			"Phi" => 0.0,
			"d" => 0.0,
		])

		initialcondsreservoir = Dict([
			"proton" => C_proton,
			"CO2" => C_co2,
			"V" => V_res,
			"T" => 300,
		])

		initialcondssurf = Dict([
			"CO2X" => CO2_X_init * sites,
			"vacantX" => (1 - CO2_X_init) * sites,
			"A" => A_surf,
			"T" => 300,
			"Phi" => surface_phi,
		])

		domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(
			phase = boundarylayer, initialconds = initialcondsboundarylayer)
		domaincat, y0cat, pcat = ConstantTAPhiDomain(
			phase = surf, initialconds = initialcondssurf)

		inter, pinter = ReactiveInternalInterfaceConstantTPhi(
			domainboundarylayer, domaincat, interfacerxns, 298.15, A_surf)
		diffusionlayer = ConstantReservoirDiffusion(
			domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness)
		interfaces = [inter, diffusionlayer]

		@time react, y0, p = Reactor((domainboundarylayer, domaincat),
			(y0boundarylayer, y0cat),
			(0.0, 1e3),
			interfaces,
			(pboundarylayer, pcat, pinter))

		@time sol = solve(react.ode, Sundials.CVODE_BDF(), abstol = 1e-20, reltol = 1e-8)

		if sol.retcode != :Success
			return 1e-15
		end

		ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p)

		# EXACT CALCULATION METHOD
		analysis_time = 100
		OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))
		
		return OCO_rate

	catch e
		return 1e-15
	end
end



# %%
using GlobalSensitivity

bounds = [
    [1e-5,   1e-2],     # CO2 concentration
    [5.0,    9.0],      # pH
    [-0.714, -0.514],   # potential
    [1e-6,   1e-4],     # layer thickness
    [0.5,    0.9]       # CO2X surface coverage
]

param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

morris_method = Morris(
    p_steps = fill(4, 5),        
    relative_scale = true, 
    num_trajectory = 50,
    total_num_trajectory = 50,
    len_design_mat = 10          #
)

println("Running Morris Global Sensitivity Analysis...")

morris_result = gsa(
    run_co2_reduction_simulation,
    morris_method,
    bounds;
    batch = false
)


# %%
param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

# Extract Morris outputs
Mu      = morris_result.means[1, :]        # signed mean effects
Mu_star = morris_result.means_star[1, :]   # absolute mean effects
Sigma  = morris_result.variances[1, :]    # variance

println("μ values = ", Mu)
println("μ* values = ", Mu_star)
println("σ² values = ", Sigma)


# %%
# Extract Morris results  (formate / O=CO)
xs = morris_result.means_star[1, :]   # absolute mean effects
ys = morris_result.variances[1, :]    # variance


param_labels = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]   

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Formate Production")
grid(true)
gcf()


# %%
# MU BAR PLOT
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    morris_result.means[1, :]
)
title("Morris Signed Mean Effects")
xlabel("Parameters")
ylabel("Mu")


gcf()


# %%
# MU* BAR PLOT
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    morris_result.means_star[1, :]
)
title("Morris Absolute Mean Effects")
xlabel("Parameters")
ylabel("μ* (Mean Absolute Effect)")

gcf()

# %%
# VARIANCE BAR PLOT
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    morris_result.variances[1, :]
)
title("Morris Variance (Nonlinearity / Interaction)")
xlabel("Parameters")
ylabel("σ² (Variance)")
gcf()

# %%
using GlobalSensitivity
using Random

println("Running Sobol Global Sensitivity Analysis...")

sobol_result = gsa(run_co2_reduction_simulation, Sobol(), [[1e-5, 1e-2], [5.0, 9.0], [-0.714, -0.514], [1e-6, 1e-4], [0.5, 0.9]], samples = 32, batch = false)

# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_result.S1[1, :]
)
title("First Order Indices O=CO")
xlabel("Parameters")
ylabel("S1")

gcf()


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_result.ST[1, :]
)
title("Sobol Total Indices O=CO")
xlabel("Parameters")
ylabel("ST")

gcf()


# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics

function run_co2_reduction_simulation1(params::Vector{Float64})
    try
        CO2_M          = params[1]
        pH             = params[2]
        surface_phi    = params[3]
        layer_thickness = params[4]
        CO2_X_init     = params[5]

        # Basic validity checks
        if CO2_X_init < 0.0 || CO2_X_init > 1.0
            error("Invalid CO2 surface coverage")
        end
        if layer_thickness <= 0.0
            error("Invalid BL thickness")
        end

        rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"
        outdict = readinput(rms_file)

        boundarylayerspcs = outdict["gas"]["Species"]
        boundarylayerrxns = outdict["gas"]["Reactions"]
        surfspcs          = outdict["surface"]["Species"]
        surfrxns          = outdict["surface"]["Reactions"]
        interfacerxns     = outdict[Set(["surface", "gas"])]["Reactions"]
        solv              = outdict["Solvents"][1]

        sitedensity = 2.292e-5
        boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv;
                                            name="boundarylayeruid", diffusionlimited=true)
        surf = IdealSurface(surfspcs, surfrxns, sitedensity; name="surface")

        C_proton = 10.0^(-pH) * 1e3
        C_co2    = CO2_M * 1e3
        V_res    = 1e3
        AVratio  = 36.0
        A_surf   = V_res * AVratio
        V_bl     = A_surf * layer_thickness
        sites    = sitedensity * A_surf

        initialcondsboundarylayer = Dict(
            "proton" => C_proton * V_bl,
            "CO2"    => C_co2 * V_bl,
            "V"      => V_bl,
            "T"      => 300.0,
            "Phi"    => 0.0,
            "d"      => 0.0
        )

        initialcondsreservoir = Dict(
            "proton" => C_proton,
            "CO2"    => C_co2,
            "V"      => V_res,
            "T"      => 300.0
        )

        initialcondssurf = Dict(
            "CO2X"    => CO2_X_init * sites,
            "vacantX" => (1 - CO2_X_init) * sites,
            "A"       => A_surf,
            "T"       => 300.0,
            "Phi"     => surface_phi
        )

        domainBL, y0BL, pBL = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)
        domainCAT, y0CAT, pCAT = ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

        inter, pinter = ReactiveInternalInterfaceConstantTPhi(domainBL, domainCAT, interfacerxns, 298.15, A_surf)
        difflayer = ConstantReservoirDiffusion(domainBL, initialcondsreservoir, A_surf, layer_thickness)

        interfaces = [inter, difflayer]

        react, y0, p = Reactor(
            (domainBL, domainCAT),
            (y0BL, y0CAT),
            (0.0, 1e3),
            interfaces,
            (pBL, pCAT, pinter)
        )

        sol = solve(react.ode, Sundials.CVODE_BDF(); abstol=1e-20, reltol=1e-8)

        if sol.retcode != :Success
            error("CVODE failure")
        end

        ssys = SystemSimulation(sol, (domainBL, domainCAT), interfaces, p)

        analysis_time = 100.0
        OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))

        return OCO_rate

    catch e
        return log10(1e-12 + sum(abs.(params)))
    end
end


# %%
# MORRIS 
using GlobalSensitivity

bounds = [
    [1e-5,   1e-2], 
    [5.0,    9.0],
    [-0.714, -0.514],
    [1e-6,   1e-4],
    [0.5,    0.9]
]

param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

morris_method = Morris(
    p_steps = fill(4, 5),
    relative_scale = true,
    num_trajectory = 40,
    total_num_trajectory = 40,
    len_design_mat = 10
)

println("Running Morris Screening...")

morris_resultx = gsa(
    run_co2_reduction_simulation1,
    morris_method,
    bounds;
    batch=false
)


# %%
param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

# Extract Morris outputs
Mu      = morris_resultx.means[1, :]        # signed mean effects
Mu_star = morris_resultx.means_star[1, :]   # absolute mean effects
Sigma  = morris_resultx.variances[1, :]    # variance

println("μ values = ", Mu)
println("μ* values = ", Mu_star)
println("σ² values = ", Sigma)


# %%
# Extract Morris results  (formate / O=CO)
xs = morris_resultx.means_star[1, :]   # absolute mean effects
ys = morris_resultx.variances[1, :]    # variance


param_labels = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]   

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Formate Production")
grid(true)
gcf()


# %%
using GlobalSensitivity

function batch_model(P::Matrix{Float64})
    n = size(P, 2)
    out = zeros(n)
    for i in 1:n
        out[i] = run_co2_reduction_simulation1(P[:, i])
    end
    return out
end

println("Running Sobol Sensitivity Analysis...")

sobol_resultx = gsa(
    batch_model,
    Sobol(),
    bounds;
    samples = 200,    
    batch = true
)


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resultx.S1[1, :]
)
title("First Order Indices O=CO")
xlabel("Parameters")
ylabel("S1")

gcf()


# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential", "layer_thickness", "CO2_X_init"],
    sobol_resultx.ST[1, :]
)
title("Total order Indices O=CO")
xlabel("Parameters")
ylabel("ST")

gcf()


# %%

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

function run_co2_reduction_simulation(params::Vector{Float64})
	try
		CO2_M = params[1]
		pH = params[2]
		surface_phi = params[3]

		rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"

		outdict = readinput(rms_file)
		boundarylayerspcs = outdict["gas"]["Species"]
		boundarylayerrxns = outdict["gas"]["Reactions"]
		surfspcs = outdict["surface"]["Species"]
		surfrxns = outdict["surface"]["Reactions"]
		interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
		solv = outdict["Solvents"][1]

		sitedensity = 2.292e-5; #Ag111
		boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv,
			name = "boundarylayeruid", diffusionlimited = true)
		surf = IdealSurface(surfspcs, surfrxns, sitedensity, name = "surface")


    C_proton = 10.0^(-pH) * 1e3         # mol/m³
    C_co2    = CO2_M * 1e3              # mol/m³
    C_default = 1e-12
    V_res   = 1e3
		layer_thickness = 1e-6;
    AVratio = 36.0
    A_surf  = V_res * AVratio
    V_bl    = A_surf * layer_thickness
    sites   = sitedensity * A_surf
		

		initialcondsboundarylayer = Dict([
			"proton" => C_proton * V_bl,
			"CO2" => C_co2 * V_bl,
			"V" => V_bl,
			"T" => 300,
			"Phi" => 0.0,
			"d" => 0.0,
		])

		initialcondsreservoir = Dict([
			"proton" => C_proton,
			"CO2" => C_co2,
			"V" => V_res,
			"T" => 300,
		])

		initialcondssurf = Dict([
			"CO2X" => 0.6 * sites,
			"vacantX" => 0.4 * sites,
			"A" => A_surf,
			"T" => 300,
			"Phi" => surface_phi,
		])

		domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(
			phase = boundarylayer, initialconds = initialcondsboundarylayer)
		domaincat, y0cat, pcat = ConstantTAPhiDomain(
			phase = surf, initialconds = initialcondssurf)

		inter, pinter = ReactiveInternalInterfaceConstantTPhi(
			domainboundarylayer, domaincat, interfacerxns, 298.15, A_surf)
		diffusionlayer = ConstantReservoirDiffusion(
			domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness)
		interfaces = [inter, diffusionlayer]

		@time react, y0, p = Reactor((domainboundarylayer, domaincat),
			(y0boundarylayer, y0cat),
			(0.0, 1e3),
			interfaces,
			(pboundarylayer, pcat, pinter))

		@time sol = solve(react.ode, Sundials.CVODE_BDF(), abstol = 1e-20, reltol = 1e-8)

		if sol.retcode != :Success
			return [1e-15, 1e-15, 0.001, 0]
		end

		ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p)

		# EXACT CALCULATION METHOD
		analysis_time = 100
		co2_rate = abs(sum(rops(ssys, "CO2", analysis_time)))

		OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))


		return [OCO_rate]

	catch e
		return [100]
	end
end

function run_gsa()
	bounds = [ [1e-5, 1e-2], [5, 9], [-0.714, -0.514]]
	param_names = ["CO2_conc", "pH", "surface_potential"]

	# Initialize variables
	morris_result = nothing

	# Morris
	println("Running Morris...")
	try
		morris_result = GlobalSensitivity.gsa(run_co2_reduction_simulation, GlobalSensitivity.Morris(), bounds; N = 200)
		println("Morris completed")
	catch e
		println("Morris failed: $e")
	end

	# Results
	if morris_result !== nothing
		num_outputs = size(morris_result.means_star, 1)
		num_params  = size(morris_result.means_star, 2)

		println("\nMorris Results:")
		for i in 1:num_outputs
			for j in 1:num_params
				#println("$(param_names[j]) -> Output $i: μ* = $(round(morris_result.means_star[i,j], digits=4))")
				println("Output $i  ←  $(param_names[j]) : μ* = $(round(morris_result.means_star[i,j], digits=4))")
			end
		end
	end

	return morris_result
end

morris_result = run_gsa()

# %%
morris_result.means

# %%
morris_result.variances

# %%
# Extract Morris results for Output 1 (formate / O=CO)
xs = morris_result.means[1, :]       # μ*
ys = morris_result.variances[1, :]   # σ²

param_labels = ["CO₂ conc", "pH", "Potential"]   # your 3 parameters

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — O=CO (Formate Production)")
grid(true)
gcf()


# %%
model_OCO(x) = run_co2_reduction_simulation(x)[1]

sobol_result = GlobalSensitivity.gsa(model_OCO, GlobalSensitivity.Sobol(), [[1e-5, 1e-2], [5, 9], [-0.714, -0.514]], samples = 32)

# %%
Pkg.add("QuasiMonteCarlo")
using QuasiMonteCarlo

samples = 32
lb = [1e-5, 5, -0.714]
ub = [1e-2, 9, -0.514]
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(samples, lb, ub, sampler)

# %%
model_OCO(x) = run_co2_reduction_simulation(x)[1]

sobol_result1 = GlobalSensitivity.gsa(model_OCO, GlobalSensitivity.Sobol(), A, B)

# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential"],
    sobol_result1.ST[1, :]
)
title("Total Order Indices O=CO")
xlabel("Parameters")
ylabel("ST")

gcf()

# %%
clf()

bar(
    ["CO2_conc", "pH", "surface_potential"],
    sobol_result1.S1[1, :]
)
title("First Order Indices O=CO")
xlabel("Parameters")
ylabel("S1")

gcf()


# %%

# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics


safe_log10(x) = log10(x + 1e-30)  

function run_co2_reduction_simulation1(params::Vector{Float64})
    try
        CO2_M         = params[1]
        pH            = params[2]
        surface_phi   = params[3]
        layer_thickness     = params[4]
        CO2X_init     = params[5]

        if !(0 < CO2X_init <= 1)
            return safe_log10(1e-20)
        end
        if layer_thickness <= 0
            return safe_log10(1e-20)
        end

        rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"
        outdict = readinput(rms_file)

        gas_species   = outdict["gas"]["Species"]
        gas_rxns      = outdict["gas"]["Reactions"]
        surf_species  = outdict["surface"]["Species"]
        surf_rxns     = outdict["surface"]["Reactions"]
        interface_rxns = outdict[Set(["surface","gas"])]["Reactions"]
        solv = outdict["Solvents"][1]

        sitedensity = 2.292e-5
        C_proton = 10.0^(-pH) * 1e3
        C_co2    = CO2_M * 1e3
        V_res = 1e3
        AVratio = 36.0
        A_surf = V_res * AVratio
        V_bl   = A_surf * layer_thk
        sites = sitedensity * A_surf

        boundary = IdealDiluteSolution(gas_species, gas_rxns, solv;
            name="boundary", diffusionlimited=true)

        surface = IdealSurface(surf_species, surf_rxns, sitedensity;
            name="surface")

        init_bl = Dict(
            "proton" => C_proton * V_bl,
            "CO2"    => C_co2 * V_bl,
            "V"      => V_bl,
            "T"      => 300.0,
            "Phi"    => 0.0,
            "d"      => 0.0,
        )

        init_res = Dict(
            "proton" => C_proton,
            "CO2"    => C_co2,
            "V"      => V_res,
            "T"      => 300.0,
        )

        init_surf = Dict(
            "CO2X"    => CO2X_init * sites,
            "vacantX" => (1 - CO2X_init) * sites,
            "A"       => A_surf,
            "T"       => 300.0,
            "Phi"     => surface_phi,
        )

        dom_bl, y0_bl, p_bl = ConstantTVDomain(phase=boundary, initialconds=init_bl)
        dom_s,  y0_s,  p_s  = ConstantTAPhiDomain(phase=surface, initialconds=init_surf)

        inter, p_inter = ReactiveInternalInterfaceConstantTPhi(dom_bl, dom_s, interface_rxns, 298.15, A_surf)
        diff           = ConstantReservoirDiffusion(dom_bl, init_res, A_surf, layer_thk)

        interfaces = [inter, diff]

        # ------------ Reactor Solve ------------------------
        react, y0, p = Reactor((dom_bl, dom_s),
                               (y0_bl, y0_s),
                               (0.0, 1e3),
                               interfaces,
                               (p_bl, p_s, p_inter))

        sol = solve(react.ode, Sundials.CVODE_BDF(); abstol=1e-20, reltol=1e-8, maxiters=1e7)

        if sol.retcode != :Success
            return safe_log10(1e-20)
        end

        ssys = SystemSimulation(sol, (dom_bl, dom_s), interfaces, p)

        tₐ = 100.0
        rate_OCO = try
            abs(sum(rops(ssys, "O=CO", tₐ)))
        catch
            1e-20
        end

        return safe_log10(rate_OCO)

    catch e
        @warn "Model exception: $e"
        return safe_log10(1e-20)
    end
end


# %%
# MORRIS 
using GlobalSensitivity

bounds = [
    [1e-5,   1e-2], 
    [5.0,    9.0],
    [-0.714, -0.514],
    [1e-6,   1e-4],
    [0.5,    0.9]
]

param_names = ["CO₂_conc", "pH", "potential", "layer_thickness", "CO2X_init"]

morris_method = Morris(
    p_steps = fill(4, 5),
    relative_scale = true,
    num_trajectory = 40,
    total_num_trajectory = 40,
    len_design_mat = 10
)

println("Running Morris Screening...")

morris_resultx= gsa(
    run_co2_reduction_simulation,
    morris_method,
    bounds;
    batch=false
)


# %%
plt = PythonPlot

μs = morris_result.means_star
σ² = morris_result.variances

plt.figure(figsize=(8,6))
plt.scatter(μs, σ², s=150, color="blue")

for i in 1:length(param_names)
    plt.annotate(param_names[i],
        (μs[i], σ²[i]),
        textcoords="offset points",
        xytext=(10, 5)
    )
end

plt.xlabel("μ* (importance)")
plt.ylabel("σ² (nonlinearity / interaction)")
plt.title("Morris Sensitivity — Formate Production")
plt.grid(true)

gcf()


# %%

# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK
using DataFrames
using Statistics

function run_co2_reduction_simulation(params::Vector{Float64})
	try
		CO2_M = params[1]
		pH = params[2]
		surface_phi = params[3]
		
		rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms"

		outdict = readinput(rms_file)
		boundarylayerspcs = outdict["gas"]["Species"]
		boundarylayerrxns = outdict["gas"]["Reactions"]
		surfspcs = outdict["surface"]["Species"]
		surfrxns = outdict["surface"]["Reactions"]
		interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
		solv = outdict["Solvents"][1]

		sitedensity = 2.292e-5; #Ag111
		boundarylayer = IdealDiluteSolution(boundarylayerspcs, boundarylayerrxns, solv,
			name = "boundarylayeruid", diffusionlimited = true)
		surf = IdealSurface(surfspcs, surfrxns, sitedensity, name = "surface")


    C_proton = 10.0^(-pH) * 1e3         # mol/m³
    C_co2    = CO2_M * 1e3              # mol/m³
    C_default = 1e-12
    V_res   = 1e3
    AVratio = 36.0
		layer_thickness = 1e-6
    A_surf  = V_res * AVratio
    V_bl    = A_surf * layer_thickness
    sites   = sitedensity * A_surf
		

		initialcondsboundarylayer = Dict([
			"proton" => C_proton * V_bl,
			"CO2" => C_co2 * V_bl,
			"V" => V_bl,
			"T" => 300,
			"Phi" => 0.0,
			"d" => 0.0,
		])

		initialcondsreservoir = Dict([
			"proton" => C_proton,
			"CO2" => C_co2,
			"V" => V_res,
			"T" => 300,
		])

		initialcondssurf = Dict([
			"CO2X" => 0.6 * sites,
			"vacantX" => 0.4 * sites,
			"A" => A_surf,
			"T" => 300,
			"Phi" => surface_phi,
		])

		domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(
			phase = boundarylayer, initialconds = initialcondsboundarylayer)
		domaincat, y0cat, pcat = ConstantTAPhiDomain(
			phase = surf, initialconds = initialcondssurf)

		inter, pinter = ReactiveInternalInterfaceConstantTPhi(
			domainboundarylayer, domaincat, interfacerxns, 298.15, A_surf)
		diffusionlayer = ConstantReservoirDiffusion(
			domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness)
		interfaces = [inter, diffusionlayer]

		@time react, y0, p = Reactor((domainboundarylayer, domaincat),
			(y0boundarylayer, y0cat),
			(0.0, 1e3),
			interfaces,
			(pboundarylayer, pcat, pinter))

		@time sol = solve(react.ode, Sundials.CVODE_BDF(), abstol = 1e-20, reltol = 1e-8)

		if sol.retcode != :Success
			return 1e-15
		end

		ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p)

		# EXACT CALCULATION METHOD
		analysis_time = 100
		OCO_rate = abs(sum(rops(ssys, "O=CO", analysis_time)))
		
		return OCO_rate

	catch e
		return 1e-15
	end
end



# %%
using GlobalSensitivity
bounds = [
    [1e-5,   1e-2],      # CO2 concentration (mol/L)
    [5.0,    9.0],       # pH
    [-0.714, -0.514]     # Potential (V)
]

param_names = ["CO₂_conc", "pH", "potential"]

morris_method = Morris(
    p_steps = fill(4, 3),       # 4 levels, 3 parameters
    relative_scale = true,
    num_trajectory = 50,
    total_num_trajectory = 50,
    len_design_mat = 10
)

println("Running Morris Global Sensitivity Analysis...")

morris_result = gsa(
    run_co2_reduction_simulation,
    morris_method,
    bounds;
    batch = false
)


# %%
miu_star = morris_result.means_star  
var     = morris_result.variances   

println("μ* (importance): ", miu_star)
println("σ² (nonlinearity/interaction): ", var)

# %%
morris_result.means

# %%
morris_result.variances

# %%
scatter(
    morris_result.means[1, :],
    morris_result.variances[1, :],
    color="blue"
)


# %%
# Extract Morris results for Output 1 (formate / O=CO)
xs = morris_result.means[1, :]       # μ*
ys = morris_result.variances[1, :]   # σ²

param_labels = ["CO₂ conc", "pH", "Potential"]   # your 3 parameters

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Output 1 (Formate Production)")
grid(true)
gcf()


# %%
scatter(
    morris_result.means[2, :],
    morris_result.variances[2, :],
    color="red"
)

# %%
# Extract Morris results for Output 1 (formate / O=CO)
xs = morris_result.means[2, :]       # μ*
ys = morris_result.variances[2, :]   # σ²

param_labels = ["CO₂ conc", "pH", "Potential"]   # your 3 parameters

# SCATTER PLOT
clf()
scatter(xs, ys, color="blue", s=120)

# LABEL EACH POINT
for i in 1:length(xs)
    text(xs[i], ys[i], "  $(param_labels[i])",
         fontsize=12, ha="left", va="bottom")
end

xlabel("μ* (Mean Elementary Effect)")
ylabel("σ² (Variance → nonlinearity / interaction)")
title("Morris Screening — Output 1 (Formate Production)")
grid(true)
gcf()


# %%
# Correct plotting syntax for PythonPlot
param_labels = ["a", "b", "c"]

# Create Morris signature plot
scatter(morris_result.means[1, :], morris_result.variances[1, :], s = 100, alpha = 0.7)

# Add labels manually
for (i, label) in enumerate(param_labels)
	annotate(label, (morris_result.means[1, i], morris_result.variances[1, i]),
		xytext = (5, 5), textcoords = "offset points")
end

xlabel("μ (Mean Effect)")
ylabel("σ (Standard Deviation)")
title("Morris Analysis - OCO_rate")
grid(true, alpha = 0.3)
gcf()


# %%
# Correct plotting syntax for PythonPlot
param_labels = ["a", "b", "c"]

# Create Morris signature plot
scatter(morris_result.means[2, :], morris_result.variances[2, :], s = 100, alpha = 0.7)

# Add labels manually
for (i, label) in enumerate(param_labels)
	annotate(label, (morris_result.means[2, i], morris_result.variances[2, i]),
		xytext = (5, 5), textcoords = "offset points")
end

xlabel("μ (Mean Effect)")
ylabel("σ (Standard Deviation)")
title("Morris Analysis - OCO_rate")
grid(true, alpha = 0.3)
gcf()


# %%
samples = 32
lb = [1e-5, 5.0, -0.714]
ub = [1e-2, 9.0, -0.514]
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(samples, lb, ub, sampler)

# %%
# Simple readable output for Morris results
param_names = ["CO2_conc", "pH", "surface_potential"]

println("MORRIS RESULTS FOR OCO_RATE:")
for (i, param) in enumerate(param_names)
	mu_star = abs(morris_result.means[1, i])
	sigma = morris_result.variances[1, i]
	importance = mu_star > 0.1 ? "HIGH" : (mu_star > 0.01 ? "MEDIUM" : "LOW")
	println("$param: μ* = $(round(mu_star, digits=6)) ($importance)")
end

# %%
# SOBOL GLOBAL SENSITIVITY ANALYSIS
function run_sobol_gsa()

    bounds = [
        [1e-5, 1e-2],      # CO₂ concentration (mol/L)
        [5.0, 9.0],        # pH
        [-0.714, -0.514]   # potential (V)
    ]

    param_names  = ["CO₂ conc", "pH", "potential"]
    output_names = ["OCO_rate (formate)", "CO2HX_rate"]

    sobol_results = Vector{Any}(undef, length(output_names))

    Random.seed!(1234)
    nsamples = 32    

    println("\nRunning Sobol GSA...")

    for k in 1:length(output_names)

        println("\n--- Sobol for output $k: $(output_names[k]) ---")

        # Scalar model wrapper
        model_k(x) = run_co2_reduction_simulation(x)[k]

        # Run Sobol (S1 and ST always computed)
        sob = GlobalSensitivity.gsa(
            model_k,
            GlobalSensitivity.Sobol(),
            bounds;
            samples = nsamples
        )

        sobol_results[k] = sob

        S1 = sob.S1
        ST = sob.ST
        S2 = sob.S2   # often nothing

        # PRINT RESULTS 
        println("\nFirst-order S1:")
        for j in 1:length(param_names)
            println("  $(param_names[j]) : S1 = $(round(S1[j], digits=4))")
        end

        println("\nTotal-order ST:")
        for j in 1:length(param_names)
            println("  $(param_names[j]) : ST = $(round(ST[j], digits=4))")
        end

        # PLOT S1
        clf()
        bar(1:length(S1), S1)
        xticks(1:length(param_names), param_names, rotation=45, ha="right")
        ylabel("S1")
        title("Sobol S1 for $(output_names[k])")
        tight_layout()
        gcf()

        # PLOT ST 
        bar(1:length(ST), ST)
        xticks(1:length(param_names), param_names, rotation=45, ha="right")
        ylabel("ST")
        title("Sobol ST for $(output_names[k])")
        tight_layout()
        gcf()

        # OPTIONAL S₂
        if S2 !== nothing
            S2_plot = deepcopy(S2)
            for i in 1:size(S2_plot, 1)
                S2_plot[i,i] = 0.0
            end

            clf()
            imshow(
                S2_plot,
                origin="lower",
                aspect="equal"
            )
            colorbar()
            xticks(1:length(param_names), param_names, rotation=45)
            yticks(1:length(param_names), param_names)
            title("Sobol S2 interactions for $(output_names[k])")
            tight_layout()
            gcf()
        else
            println("\n[S2 unavailable — interactions cannot be computed with current sample size.]")
        end
    end

    return sobol_results
end

sobol_results = run_sobol_gsa()

