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
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase

# %%
# CONSTANTS & CONFIGURATION
F_const = 96485.0   # C/mol
R = 8.314           # J/(mol·K)
T = 298.15          # K

pH = 0
E_eq_SHE = -0.059 * pH  # 0 V vs SHE at pH 0

sitedensity = 2.483e-5  # mol/m^2 for Pt(111)
A_surf = 1e-4           # m^2 (1 cm^2)
layer_thickness = 1e-6  # m
V_bl = A_surf * layer_thickness
sites = sitedensity * A_surf

# LOAD MECHANISM
rms_file = "/home/danieltori/CO2_RR_RMG/CO2_Reduction_Ag/HER/Pt_Corr2.rms" 
outdict = readinput(rms_file)

boundarylayerspcs = outdict["gas"]["Species"]
surfspcs = outdict["surface"]["Species"]
surfrxns = outdict["surface"]["Reactions"]
interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
solv = outdict["Solvents"][1]

# Create phases
boundarylayer = IdealDiluteSolution(
    boundarylayerspcs,
    Float64[],   # no homogeneous gas reactions
    solv,
    name = "boundarylayeruid",
    diffusionlimited = true
)

surf = IdealSurface(
    surfspcs,
    surfrxns,
    sitedensity,
    name = "surface"
)

println("=== Reaction flags (surface) ===")
for (i, rxn) in enumerate(surfrxns)
    println("rxn $i: reversible=$(rxn.reversible) forwardable=$(rxn.forwardable) electronchange=$(rxn.electronchange)")
end

println("=== Reaction flags (interface) ===")
for (i, rxn) in enumerate(interfacerxns)
    println("rxn $i: reversible=$(rxn.reversible) forwardable=$(rxn.forwardable) electronchange=$(rxn.electronchange)")
end

"""
Run HER simulation at a given potential (V vs SHE) and return electrochemical current density.
Current is computed from interface electron-transfer rates, not from H2 species ROP.
"""

function run_HER(Phi_SHE; t_end=1e3, pH=0)
    C_proton = 10.0^(-pH) * 1e3   # mol/m^3
    C_H2_initial = 0.78            # mol/m^3
    V_res = 1e3

    # Boundary layer initial conditions
    initialcondsboundarylayer = Dict(
        "proton" => C_proton * V_bl,
        "H2"     => C_H2_initial * V_bl,
        "V"      => V_bl,
        "T"      => T,
        "Phi"    => 0.0,
        "d"      => 0.0
    )

    # Reservoir conditions
    initialcondsreservoir = Dict(
        "proton" => C_proton,
        "H2"     => C_H2_initial,
        "V"      => V_res,
        "T"      => T
    )

    # Surface initial conditions
    initialcondssurf = Dict(
        "HX"      => 0.11 * sites,
        "vacantX" => 0.89 * sites,
        "A"       => A_surf,
        "T"       => T,
        "Phi"     => Phi_SHE
    )

    # Create domains
    domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(
        phase = boundarylayer,
        initialconds = initialcondsboundarylayer
    )

    domaincat, y0cat, pcat = ConstantTAPhiDomain(
        phase = surf,
        initialconds = initialcondssurf
    )

    domainboundarylayer.diffusivity[1] = 0.932e-8

    # Interfaces
    inter, pinter = ReactiveInternalInterfaceConstantTPhi(
        domainboundarylayer, domaincat, interfacerxns, T, A_surf
    )

    diffusionlayer = ConstantReservoirDiffusion(
        domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness
    )

    interfaces = [inter, diffusionlayer]

    # Create and solve reactor
    react, y0, p = Reactor(
        (domainboundarylayer, domaincat),
        (y0boundarylayer, y0cat),
        (0.0, t_end),
        interfaces,
        (pboundarylayer, pcat, pinter)
    )

    sol = solve(react.ode, Sundials.CVODE_BDF(), abstol=1e-22, reltol=1e-8)

    if sol.retcode != :Success
        @warn "Simulation at Phi = $Phi_SHE did not converge: $(sol.retcode)"
        return NaN, nothing, nothing
    end

    ssys = SystemSimulation(sol, (domainboundarylayer, domaincat), interfaces, p)

    t0 = sol.t[1]
    tss = sol.t[end]

    hx0 = concentrations(ssys, "HX", t0)
    vac0 = concentrations(ssys, "vacantX", t0)
    theta0 = hx0 / (hx0 + vac0)

    hxss = concentrations(ssys, "HX", tss)
    vacss = concentrations(ssys, "vacantX", tss)
    thetass = hxss / (hxss + vacss)

    println("θ_H initial = ", theta0)
    println("θ_H final   = ", thetass)

    # Net rates from the system simulation
    r_all = rates(ssys, sol.t[end])

    # In this HER setup, the returned rates should correspond to the interface reactions
    if length(r_all) != length(interfacerxns)
        error("rates(ssys, t) returned $(length(r_all)) rates, but there are $(length(interfacerxns)) interface reactions.")
    end

    # Electrochemical current from interface reactions only
    j = 0.0
    for i in eachindex(interfacerxns)
        j += interfacerxns[i].electronchange * r_all[i]
    end

    j = -F_const * j / A_surf   # A/m^2
    j_mA_cm2 = j / 10.0         # mA/cm^2

    return j_mA_cm2, r_all, ssys
end

# %%
# RUN SINGLE-POINT TEST
phi_test = -0.10
single = run_HER(phi_test; pH=0, t_end=1e3)

# %%
phi_test = 0.0
single = run_HER(phi_test; pH=0, t_end=1e3)

# %%
run_HER(0.0; pH=0, t_end=1e-1)

# %%
run_HER(0.0; pH=0, t_end=1.0)

# %%
run_HER(0.0; pH=0, t_end=1e-3)

# %%
# Single-point diagnostic at equilibrium (0 V vs SHE)
j, θ_H, ssys = run_HER(0.0; t_end=1e3, pH=0)

println("j = $j mA/cm²")
println("θ_H = $θ_H")

# Check individual forward and reverse rate constants
sim_surf = ssys.sims[2]
t_end = sim_surf.sol.t[end]

# Print all rate-of-production for each species
println("\n=== Species ROP at steady state ===")
for sp in ["proton", "H2", "HX", "vacantX"]
    try
        rop = rops(ssys, sp, t_end)
        println("$sp: total ROP = $(sum(rop))")
        for (i, r) in enumerate(rop)
            println("  rxn $i contribution: $r")
        end
    catch e
        println("$sp: couldn't get ROP — $e")
    end
end

# %%
# Diagnostic: check Kc and krev for interface reactions
println("Interface reaction Kc diagnostic \n")

# Get the interface object
inter_obj = ssys.interfaces[1]  # ReactiveInternalInterfaceConstantTPhi

# Check if we can access the internal state
println("Interface type: $(typeof(inter_obj))")

# Manually compute ΔG and Kc for each reaction
F_val = 96485.0
R_val = 8.314
phi = 0.0  # the potential we ran at

for (i, rxn) in enumerate(interfacerxns)
    # Compute ΔG from species Gibbs energies
    G_reactants = sum([getGibbs(sp.thermo, T) for sp in rxn.reactants])
    G_products = sum([getGibbs(sp.thermo, T) for sp in rxn.products])
    dG_chem = G_products - G_reactants
    dG_elec = rxn.electronchange * phi * F_val
    dG_total = dG_chem + dG_elec
    
    # Kc (without C0 correction for now)
    Kc_thermo = exp(-dG_total / (R_val * T))
    
    rnames = join([sp.name for sp in rxn.reactants], " + ")
    pnames = join([sp.name for sp in rxn.products], " + ")
    
    println("Rxn $i: $rnames => $pnames")
    println("  ΔG_chem = $(round(dG_chem, digits=1)) J/mol")
    println("  ΔG_elec = $(round(dG_elec, digits=1)) J/mol")
    println("  ΔG_total = $(round(dG_total, digits=1)) J/mol")
    println("  Kc (no C0 corr) = $(Kc_thermo)")
    println("  electronchange = $(rxn.electronchange)")
    println("")
end

# %%
println("C0 for gas/solution phase: $(1e5/(R*T)) mol/m³")
println("C0 for surface phase: $sitedensity mol/m²")

# %%
# POTENTIAL SWEEP
E_eq_SHE = -0.059 * pH  # 0 V vs SHE at pH 0

Phi_values = collect(E_eq_SHE:-0.025:(E_eq_SHE - 0.500))

overpotential_values = Float64[]
current_density_values = Float64[]
theta_H_values = Float64[]

# Get HX species index for coverage extraction
spc_names = [sp.name for sp in surfspcs]
HX_idx = findfirst(x -> x == "HX", spc_names)

println("HER POTENTIAL SWEEP — Pt(111), pH $pH")

for Phi in Phi_values
    overpotential = Phi - E_eq_SHE

    print("Phi_SHE = $(round(Phi, digits=3)) V, η = $(round(overpotential*1000, digits=1)) mV ... ")

    j_mA_cm2, r_all, ssys = run_HER(Phi; pH=pH, t_end=1e3)

    if !isnan(j_mA_cm2) && abs(j_mA_cm2) > 0 && ssys !== nothing
        # Extract coverage
        t_ss = ssys.sims[2].sol.t[end]
        hx_conc = concentrations(ssys, "HX", t_ss)
        vac_conc = concentrations(ssys, "vacantX", t_ss)
        θ_H = hx_conc / (hx_conc + vac_conc)
       
        push!(overpotential_values, overpotential)
        push!(current_density_values, abs(j_mA_cm2))
        push!(theta_H_values, θ_H)
        println("j = $(round(j_mA_cm2, digits=4)) mA/cm², θ_H = $(round(θ_H, digits=4))")
    else
        println("FAILED or zero current")
    end
end

# %%
# TAFEL SLOPE CALCULATION
tafel_idx = findall(-0.10 .<= overpotential_values .<= -0.0)
# tafel_idx = findall(-0.25 .<= overpotential_values .<= -0.10)

tafel_slope = NaN
slope = NaN
intercept = NaN
η_fit = Float64[]
logj_fit = Float64[]

if length(tafel_idx) >= 3
    η_fit = overpotential_values[tafel_idx]
    logj_fit = log10.(current_density_values[tafel_idx])

    n_pts = length(η_fit)
    x = logj_fit
    y = η_fit
    x_mean = sum(x) / n_pts
    y_mean = sum(y) / n_pts
    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    intercept = y_mean - slope * x_mean

    tafel_slope = abs(slope * 1000.0)
end

theoretical_slope = 2.303 * R * T / (0.5 * F_const) * 1000

println("TAFEL SLOPE RESULTS")
println("Computed:               $(round(tafel_slope, digits=1)) mV/dec")
println("Expected (Tang et al.): ~121 mV/dec (Table 1)")
println("Theoretical (β=0.5):   $(round(theoretical_slope, digits=1)) mV/dec")

# %%
# CELL 4: TAFEL PLOT

clf()
fig, ax = subplots(figsize=(8, 6))

ax.plot(overpotential_values .* 1000, log10.(current_density_values), "o-",
        markersize=8, linewidth=2, color="blue", label="RMS simulation")

if length(tafel_idx) >= 3
    η_line = range(minimum(η_fit), maximum(η_fit), length=50)
    logj_line = (η_line .- intercept) ./ slope
    ax.plot(η_line .* 1000, logj_line, "--", color="red", linewidth=2,
            label="Fit: $(round(tafel_slope, digits=1)) mV/dec")
end

ax.axvline(x=-100, linestyle=":", color="gray", alpha=0.5)
ax.text(-120, maximum(log10.(current_density_values)) - 0.5, "Tafel regime",
        fontsize=10, color="gray")

ax.set_xlabel("Overpotential η (mV)", fontsize=14)
ax.set_ylabel("log₁₀|j| (mA/cm²)", fontsize=14)
ax.set_title("HER Tafel Plot — Pt(111), pH $pH\nTang et al. (2020) parameters", fontsize=13)
ax.legend(fontsize=12)
ax.grid(true, alpha=0.3)
ax.tick_params(labelsize=12)

tight_layout()
savefig("tafel_Pt111_pH$(pH).png", dpi=150)
gcf()

# %%
# Diagnose Kc and krev for each interface reaction
println("Kc / krev diagnostic \n")

# C0 values
C0_solution = 1e5 / (R * T)
C0_surface = sitedensity

println("C0_solution = $C0_solution mol/m³")
println("C0_surface  = $C0_surface mol/m²")
println()

for (i, rxn) in enumerate(interfacerxns)
    rnames = join([sp.name for sp in rxn.reactants], " + ")
    pnames = join([sp.name for sp in rxn.products], " + ")
    
    # Count reactants/products in each phase
    n_react_sol = count(sp -> sp.name in ["proton", "H2"], rxn.reactants)
    n_react_surf = count(sp -> sp.name in ["HX", "vacantX"], rxn.reactants)
    n_prod_sol = count(sp -> sp.name in ["proton", "H2"], rxn.products)
    n_prod_surf = count(sp -> sp.name in ["HX", "vacantX"], rxn.products)
    
    # ΔG
    G_react = sum([getGibbs(sp.thermo, T) for sp in rxn.reactants])
    G_prod = sum([getGibbs(sp.thermo, T) for sp in rxn.products])
    dG = G_prod - G_react
    
    # Kc without C0 correction (thermodynamic)
    Kc_thermo = exp(-dG / (R * T))
    
    # C0 correction factor
    # Δν_sol = n_prod_sol - n_react_sol
    # Δν_surf = n_prod_surf - n_react_surf
    # C0_factor = C0_solution^Δν_sol × C0_surface^Δν_surf
    dv_sol = n_prod_sol - n_react_sol
    dv_surf = n_prod_surf - n_react_surf
    C0_factor = C0_solution^dv_sol * C0_surface^dv_surf
    
    Kc_with_C0 = Kc_thermo * C0_factor
    
    println("Rxn $i: $rnames => $pnames")
    println("  ΔG = $(round(dG, digits=1)) J/mol")
    println("  Kc (thermo only) = $Kc_thermo")
    println("  Δν_solution = $dv_sol, Δν_surface = $dv_surf")
    println("  C0 factor = $C0_factor")
    println("  Kc (with C0) = $Kc_with_C0")
    println("  If kf = 1e-5, then krev = $(1e-5 / Kc_with_C0)")
    println()
end

# %%
# Plot θ_H vs time from the most recent single-point run

# In this notebook, `run_HER` returns (j_mA_cm2, r_all, ssys)
_ssys = if @isdefined(ssys)
    ssys
elseif @isdefined(single) && single isa Tuple && length(single) >= 3
    single[3]
else
    error("`ssys` not defined. Run the single-point cell first (it defines `single`), or define `ssys` explicitly.")
end

# Use the surface simulation time grid
ts = _ssys.sims[2].sol.t
hx_vals = [concentrations(_ssys, "HX", t) for t in ts]
vac_vals = [concentrations(_ssys, "vacantX", t) for t in ts]

theta_h = hx_vals ./ (hx_vals .+ vac_vals)

clf()
plot(ts, theta_h, marker="o")
xscale("log")
xlabel("Time (s)")
ylabel("θ_H")
title("Hydrogen surface coverage vs time")
grid(true)
gcf()

# %%
run_HER(0.0; pH=0, t_end=1e-5)

# %%
