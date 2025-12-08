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
    morris_result.variance[1, :]
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

