# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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
using GlobalSensitivity
using Random

# %%
outdict = readinput("/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/Ag_C2_042925.rms")

# %%
boundarylayerspcs = outdict["gas"]["Species"]
boundarylayerrxns = outdict["gas"]["Reactions"]
surfspcs = outdict["surface"]["Species"]
surfrxns = outdict["surface"]["Reactions"]
interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
solv = outdict["Solvents"][1];

# %%
sitedensity = 2.292e-5; # Ag111 site density is 2.294e-9 mol/cm^2 or 2.294e-5 mol/m^2
boundarylayer = IdealDiluteSolution(boundarylayerspcs,boundarylayerrxns,solv,name="boundarylayeruid",diffusionlimited=true);
surf = IdealSurface(surfspcs,surfrxns,sitedensity,name="surface");

# %%
# PARAMETER BOUNDS
lb = [4.0,   0.01, -1.614, 1e-7]   # pH, CO2, potential, layer thickness
ub = [9.0,   10.0, -0.514, 5e-5]
bounds = hcat(lb, ub)

param_labels = ["pH", "CO₂ (mol/m3)", "Potential (V)", "Layer thickness (m)"]

# MODEL FOR GLOBAL SENSITIVITY — RETURNS FORMATE
function run_formate_model(p)
    pH      = p[1]
    CO2_M   = p[2]
    phi_val = p[3]
    layer_thickness = p[4]

    C_proton = 10.0^(-pH) * 1e3         # mol/m³
    C_co2    = CO2_M * 1e3              # mol/m³
    C_default = 1e-12
    V_res   = 1e3
    AVratio = 36.0
    A_surf  = V_res * AVratio
    V_bl    = A_surf * layer_thickness
    sites   = sitedensity * A_surf


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
        "CO2X"    => 0.1 * sites,
        "vacantX" => 0.9 * sites,
        "A"       => A_surf,
        "T"       => 300.0,
        "Phi"     => phi_val
    )


    domainboundarylayer, y0boundarylayer, pboundarylayer =
        ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)

    domaincat, y0cat, pcat =
        ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

    domainboundarylayer.diffusivity[6] = 0.932e-8

    inter, pinter = ReactiveInternalInterfaceConstantTPhi(
        domainboundarylayer, domaincat, interfacerxns, 298.15, A_surf
    )

    diffusionlayer = ConstantReservoirDiffusion(
        domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness
    )

    interfaces = [inter, diffusionlayer]

    react, y0, p_all = Reactor(
        (domainboundarylayer, domaincat),
        (y0boundarylayer, y0cat),
        (0.0, 1e3),
        interfaces,
        (pboundarylayer, pcat, pinter)
    )

    sol = solve(react.ode, Sundials.CVODE_BDF(), abstol=1e-22, reltol=1e-8)

    if sol.retcode != :Success
        return NaN
    end

    ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p_all)

    # extract formate
    formate_index = findfirst(==("O=CO"), [s.name for s in ssys.species])
    return concentrations(ssys, sol.t[end])[formate_index]
end


model_wrapper(x) = run_formate_model(x)


# %%
# quick test at some baseline:
baseline_p = [7.0, 0.01, -1.414, 1e-6]
println("Baseline test [O=CO] = ", run_formate_model(baseline_p), " mol/L")


# %%
bounds

# %%
method = Sobol()
morris_result = gsa(run_formate_model,
    method,
    bounds;
    samples = 2000,
    batch=false
)
morris_result

# %%
println(morris_result.means_star)
println(morris_result.variances)

# %%
# SOBOL GLOBAL SENSITIVITY ANALYSIS
sobol_samples = 150  

sobol_result = gsa(model_wrapper,
    Sobol(),
    lb,
    ub;
    samples = sobol_samples
)

println("\nSobol S1 (first-order indices):")
println(sobol_result.S1)

println("\nSobol ST (total-order indices):")
println(sobol_result.ST)

# %%
# Create bounds matrix (num_params × 2)
bounds = hcat(lb, ub)

# Correct Morris structure for your version
morris_method = Morris(
    p_steps = [6],         # discretization levels (vector)
    relative_scale = true,
    num_trajectory = 20    # scalar
)

morris_result = gsa(model_wrapper, method::Morris,
    morris_method,
    bounds;
    samples = 50
    batch = false
)

mu_star = morris_result.means_star
sigma2  = morris_result.variances

println("\nMorris μ* (importance): ", mu_star)
println("\nMorris σ² (nonlinearity / interaction): ", sigma2)


# %%
# plot Morris μ*
clf()
bar(param_labels, mu_star)
ylabel("Morris μ*")
title("Formate (O=CO) – Morris Mean Elementary Effect")
tight_layout()
gcf()


# %%
# plot Morris σ²
clf()
bar(param_labels, sigma2)
ylabel("Morris σ²")
title("Formate (O=CO) – Morris Variance of Elementary Effects")
tight_layout()
gcf()

# %%
# SOBOL GLOBAL SENSITIVITY ANALYSIS
println("\n Running Sobol sensitivity analysis")

sobol_samples = 150  

sobol_result = gsa(
    model_wrapper,
    Sobol(),
    lb,
    ub;
    samples = sobol_samples
)

println("\nSobol S1 (first-order indices):")
println(sobol_result.S1)

println("\nSobol ST (total-order indices):")
println(sobol_result.ST)

# %%
# plot Sobol S1
clf()
bar(param_labels, sobol_result.S1)
ylabel("Sobol S1")
title("Formate (O=CO) – Sobol First-order Sensitivity")
tight_layout()
gcf()

# %%
# plot Sobol ST 
clf()
bar(param_labels, sobol_result.ST)
ylabel("Sobol ST")
title("Formate (O=CO) – Sobol Total Sensitivity")
tight_layout()
gcf()

# %%
#Model function that:
#  - takes pH, CO2 molarity (mol/L), applied potential Phi (V)
#  - builds domains and interfaces
#  - runs RMS simulation
#  - returns boundary-layer O=CO concentration at final time (mol/m^3)
function run_model_formate(pH, co2_molarity, phi_applied)
    #### 1. Convert pH and CO2 molarity to concentrations (mol/m^3)
    C_proton_new = 10.0^(-pH) * 1000.0    # mol/m^3
    C_co2_new    = co2_molarity * 1000.0  # mol/m^3

    #### 2. Define initial conditions for boundary layer and reservoir
    initialcondsboundarylayer = Dict(
        "proton" => C_proton_new * V_bl,
        "CO2"    => C_co2_new * V_bl,
        "V"      => V_bl,
        "T"      => 300.0,
        "Phi"    => 0.0,
        "d"      => 0.0
    )

    initialcondsreservoir_local = Dict(
        "proton" => C_proton_new,
        "CO2"    => C_co2_new,
        "V"      => V_res,
        "T"      => 300.0
    )

    #### 3. Surface initial conditions (use same site split as before, variable Phi)
    initialcondssurf = Dict(
        "CO2X"    => 0.1 * sites,
        "vacantX" => 0.9 * sites,
        "A"       => A_surf,
        "T"       => 300.0,
        "Phi"     => phi_applied
    )

    #### 4. Construct domains for this parameter set
    domainboundarylayer_local, y0bl, pbl =
        ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer)

    domaincat_local, y0cat, pcat =
        ConstantTAPhiDomain(phase=surf, initialconds=initialcondssurf)

    # Maintain your proton diffusivity modification
    domainboundarylayer_local.diffusivity[6] = 0.932e-8

    #### 5. Interfaces (reaction + diffusion)
    inter_local, pinter =
        ReactiveInternalInterfaceConstantTPhi(domainboundarylayer_local,
                                              domaincat_local,
                                              interfacerxns,
                                              298.15,
                                              A_surf)

    diffusionlayer_local =
        ConstantReservoirDiffusion(domainboundarylayer_local,
                                   initialcondsreservoir_local,
                                   A_surf,
                                   layer_thickness)

    interfaces_local = [inter_local, diffusionlayer_local]

    #### 6. Build reactor and solve
    reactor_local, y0_local, p_local =
        Reactor((domainboundarylayer_local, domaincat_local),
                (y0bl, y0cat),
                (0.0, 1e3),
                interfaces_local,
                (pbl, pcat, pinter))

    sol_local = solve(reactor_local.ode,
                      Sundials.CVODE_BDF();
                      abstol=1e-22,
                      reltol=1e-8)

    # If solver fails, return NaN (so GSA can still proceed)
    if sol_local.retcode != :Success
        @warn "Simulation failed for pH = $pH, CO2 = $co2_molarity, Φ = $phi_applied; retcode = $(sol_local.retcode)"
        return NaN
    end

    #### 7. Wrap as SystemSimulation
    ssys_local = SystemSimulation(sol_local,
                                  (domainboundarylayer_local, domaincat_local),
                                  interfaces_local,
                                  p_local)

    #### 8. Extract boundary-layer O=CO concentration at final time
    t_final = sol_local.t[end]
    concs_bl = concentrations(ssys_local.sims[1], t_final)  # boundary-layer

    species_names_bl = [s.name for s in ssys_local.sims[1].domain.phase.species]
    idx_formate = findfirst(==("O=CO"), species_names_bl)

    if idx_formate === nothing
        @warn "Species O=CO not found in boundary-layer species list."
        return NaN
    end

    # Return O=CO concentration in boundary layer (mol/m^3)
    return concs_bl[idx_formate]
end

# WRAPPER FOR GLOBALSENSITIVITY
# GlobalSensitivity expects a function f(x::Vector)
model_simulation(x) = run_model_formate(x[1], x[2], x[3])

# PARAMETER BOUNDS
#   x1 = pH               ∈ [4, 8]
#   x2 = CO2 molarity     ∈ [0.005, 0.05] mol/L
#   x3 = applied potential∈ [-1.6, -0.6] V vs RHE (for example)
lb = [4.0,   0.005, -1.6]   # lower bounds
ub = [8.0,   0.050, -0.6]   # upper bounds

# MORRIS METHOD (SCREENING)
# samples here is number of model evaluations (approx);
# Morris is cheaper, but still runs many simulations.
morris_samples = 40

morris_result = gsa(model_wrapper,
                    Morris(),
                    lb,
                    ub;
                    samples = morris_samples)

println("\n MORRIS RESULTS (Formate O=CO)")
println("mu*    (mean absolute elementary effect): ", morris_result.mu_star)
println("sigma  (standard deviation of effects):   ", morris_result.sigma)


# %%
# Plot Morris μ*
clf()
bar(["pH", "CO2", "Voltage"], morris_result.mu_star)
ylabel("μ* (Mean absolute elementary effect)")
title("Morris Screening for Formate (O=CO)")
gcf()


# %%
# Plot Morris σ (interaction / nonlinearity indication)
clf()
bar(["pH", "CO2", "Voltage"], morris_result.sigma)
ylabel("σ (Std dev of elementary effects)")
title("Morris σ for Formate (O=CO) – Nonlinearity / Interaction")
gcf()

# %%
# SOBOL SENSITIVITY ANALYSIS (VARIANCE-BASED)
# samples: number of base quasi-random samples.
# Total runs ≈ samples * (2 * n_params + 2)
sobol_samples = 64  # start small; increase later when it's stable

sobol_result = gsa(model_simulation,
                   Sobol(),
                   lb,
                   ub;
                   samples = sobol_samples)

println("\n SOBOL RESULTS (Formate O=CO)")
println("First-order indices S1: ", sobol_result.S1)
println("Total indices    ST: ", sobol_result.ST)
println("Second-order S2 matrix: ")
println(sobol_result.S2)

# %%
# Plot first-order Sobol indices
clf()
bar(["pH", "CO2", "Voltage"], sobol_result.S1)
ylabel("First-order Sobol index S1 (O=CO at t_final)")
title("Sobol Sensitivity for Formate (O=CO)")
gcf()

# %%
# Plot total Sobol indices
clf()
bar(["pH", "CO2", "Voltage"], sobol_result.ST)
ylabel("Total Sobol index ST (O=CO at t_final)")
title("Total Sobol Sensitivity for Formate (O=CO)")
gcf()

