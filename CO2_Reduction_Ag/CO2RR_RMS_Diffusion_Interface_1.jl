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
#     display_name: Julia 1.10.10
#     language: julia
#     name: julia-1.10
# ---

# %%
using Pkg
Pkg.activate(ENV["PYTHON_JULIAPKG_PROJECT"])

# %%
using ReactionMechanismSimulator

# %%
using PythonPlot
using DifferentialEquations
using Sundials
using SciMLBase
using QuadGK

# %%
outdict = readinput("Ag_C2_042925.rms")

# %%
boundarylayerspcs = outdict["gas"]["Species"]
boundarylayerrxns = outdict["gas"]["Reactions"]
surfspcs = outdict["surface"]["Species"]
surfrxns = outdict["surface"]["Reactions"]
interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
solv = outdict["Solvents"][1];

# %%
sitedensity = 2.294e-5; # Ag111 site density is 2.294e-9 mol/cm^2 or 2.294e-5 mol/m^2
boundarylayer = IdealDiluteSolution(boundarylayerspcs,boundarylayerrxns,solv,name="boundarylayeruid",diffusionlimited=true);
surf = IdealSurface(surfspcs,surfrxns,sitedensity,name="surface");

# %%
# Reservoir is a 100 mL (100e-6 m^3) cell
# Proton concentration is 10^-7 mol/L (10^-4 mol/m^3)
# CO2 concentration is 0.01 mol/L (10 mol/m^3), saturation solubility ~0.03 mol/L
# AVratio in experiments is 36 m^-1 but is measured by surface area/reservoir volume
# Area of the electrode is therefore 3.6e1 m^-1 * 1e2 * 1e-6 m^3 = 3.6e-3 m^2 = 36 cm^2
# Assume boundary layer thickness d_bl = 1 mm or 1e-3 m
# Volume of the boundary layer V_bl = 3.6e-3 m^2 * 1e-3 m = 3.6e-6 m^3
# Actual AVratio is therefore 3.6e-3 m^2 / 3.6e-6 m^3 = 1e3 m^-1 (reciprocal of d_bl)
# Amount of sites is 2.943e-5 mol/m^2 * 3.6e-3 m^2 = 10.595e-8 mol

C_proton = 1e-7*1e3;
C_co2 = 1e-2*1e3;
C_default = 1e-12;
V_res = 1e3;
layer_thickness = 1e-3;
A_surf = V_res*36;
V_bl = A_surf*layer_thickness;
# V_bl = V_res;
sites = sitedensity*A_surf;

# The initial conditions for individual species are moles not concentration, so we need to multiply concentration by boundary layer volume
initialcondsboundarylayer = Dict(["proton"=>C_proton*V_bl,
                                  "CO2"=>C_co2*V_bl,
                                #   "H2"=>C_default*10*V_bl,
                                #   "O=CO"=>C_default*V_bl,
                                  "V"=>V_bl,"T"=>300,"Phi"=>0.0,"d"=>0.0]);
initialcondsreservoir = Dict(["proton"=>C_proton,
                              "CO2"=>C_co2,
                              "V"=>V_res,"T"=>300]);


# Assume voltage is -1.0 V vs. R.H.E. which equates to -1.414 V vs. S.H.E. at pH=7
initialcondssurf = Dict(["CO2X"=>0.4*sites,
        # "CHO2X"=>0.1*sites,
        # "CO2HX"=>0.1*sites,
        # "OX"=>0.1*sites,
        # "OCX"=>0.1*sites,
        "vacantX"=>0.6*sites,
        # "CH2O2X"=>0.05*sites,
        # "CHOX"=>0.04*sites,
        # "CH2OX"=>0.01*sites,
        "A"=>A_surf,"T"=>300,"Phi"=>-1.414]);

# %%
domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer);
domaincat,y0cat,pcat = ConstantTAPhiDomain(phase=surf,
    initialconds=initialcondssurf);

# %%
inter,pinter = ReactiveInternalInterfaceConstantTPhi(domainboundarylayer,
  domaincat,interfacerxns,298.15,A_surf);

# %%
# start with 1mm layer thickness
diffusionlayer = ConstantReservoirDiffusion(domainboundarylayer, initialcondsreservoir, A_surf, layer_thickness);

# %%
interfaces = [inter,diffusionlayer];

# %%
@time react,y0,p = Reactor((domainboundarylayer,domaincat), (y0boundarylayer,y0cat), (0.0, 1e3), interfaces, (pboundarylayer,pcat,pinter));

# %%
fieldnames(typeof(react.ode))

# %%
fieldnames(typeof(react.ode.f.f))

# %%
react.ode.f.f

# %%
@time sol = solve(react.ode,Sundials.CVODE_BDF(),abstol=1e-20,reltol=1e-8);
println(sol.t[end]);
println(sol.retcode);

# %%
ssys = SystemSimulation(sol,(domainboundarylayer,domaincat,), interfaces,p);

# %%
concentrations(ssys.sims[1], 1e3)

# %%
"""
diffusive flux to the reservoir
"""
function flux_to_reservoir(sim,t,reservoirinterface)
    cs = concentrations(sim,t)
    # println(reservoirinterface.A);
    # println(reservoirinterface.layer_thickness);
    # println(sim.domain.diffusivity);
    # println(cs);
    # println(reservoirinterface.c);
    return reservoirinterface.A .* sim.domain.diffusivity .* (cs - reservoirinterface.c) / reservoirinterface.layer_thickness
end

"""
Integrates the flux to the reservoir and computes the concentration assuming
there is no prior concentration of that species in the reservoir
"""
function get_reservoir_concentration(sim,t,reservoirinterface,Vres)
    intg,err = quadgk(x -> flux_to_reservoir(sim,x,reservoirinterface), 0, t);
    return intg./Vres
end

# %%
# Logarithmic time scale
t_vals = 10 .^ range(-12, stop=3, length=100);

# Compute reservoir concentrations
flux_vals = [flux_to_reservoir(ssys.sims[1], t, diffusionlayer) for t in t_vals]
# conc_vals = [get_reservoir_concentration(ssys.sims[1], t, diffusionlayer, V_res) for t in t_vals]
conc_vals_bl = [concentrations(ssys.sims[1], t) for t in t_vals]
flux_matrix = hcat(flux_vals...);
# conc_matrix = hcat(conc_vals...);
conc_matrix_bl = hcat(conc_vals_bl...);


# %%
flux_matrix

# %%
conc_matrix_bl

# %%
# clf()

# for i in 1:size(conc_matrix, 1)
#     if maximum(conc_matrix[i, :]) > 1e-12
#         plot(t_vals, conc_matrix[i, :], label=ssys.sims[1].domain.phase.species[i].name)

#     end
# end

# xscale("log")
# yscale("log")
# xlabel("Time (s)")
# ylabel("Reservoir Concentration")
# legend()
# tight_layout()
# gcf()

# %%
clf()

for i in 1:size(flux_matrix, 1)
    if abs(maximum(flux_matrix[i, :])) > 1e-12
        plot(t_vals, flux_matrix[i, :], label=ssys.sims[1].domain.phase.species[i].name)

    end
end

xscale("log")
# yscale("log")
xlabel("Time (s)")
ylabel("Diffusive Flux (mol/s)")
legend()
tight_layout()
gcf()

# %%
clf()
for i in 1:size(conc_matrix_bl, 1)
    if maximum(conc_matrix_bl[i, :]) > 1e-16
        plot(t_vals, conc_matrix_bl[i, :], label=ssys.sims[1].domain.phase.species[i].name)

    end
end

xscale("log")
yscale("log")
xlabel("Time (s)")
ylabel("Boundary Layer Concentrations (mol/m^3)")
legend()
tight_layout()
gcf()

# %%
flux_to_reservoir(ssys.sims[1],1e-12,diffusionlayer)

# %%
@time res_cs = get_reservoir_concentration(ssys.sims[1],1e3,diffusionlayer,V_res)

# %%
sort(res_cs)

# %%
getfield.(ssys.sims[1].domain.phase.species,:name)

# %%
getfield.(ssys.sims[2].domain.phase.species,:name)

# %%
# Helper function
function plotX(sim, tol, t_end, exclude)
    clf()
    xs = molefractions(sim)
    maxes = maximum(xs, dims=2)

    # Filter time data up to t_end
    time_indices = findall(t -> t <= t_end, sim.sol.t)
    time_filtered = sim.sol.t[time_indices]
    xs_filtered = xs[:, time_indices]

    for i = 1:length(maxes)
        species_name = sim.domain.phase.species[i].name
        if maxes[i] > tol && !(species_name in exclude)
            plot(time_filtered, xs_filtered[i,:], label=species_name)
        end
    end
    legend()
    xlabel("Time in Sec")
    ylabel("Mole Fraction")
end

# %%
# Helper function
function plotC(sim, tol, t_end, exclude)
    clf()
    xs = concentrations(sim)
    maxes = maximum(xs, dims=2)

    # Filter time data up to t_end
    time_indices = findall(t -> t <= t_end, sim.sol.t)
    time_filtered = sim.sol.t[time_indices]
    xs_filtered = xs[:, time_indices]

    for i = 1:length(maxes)
        species_name = sim.domain.phase.species[i].name
        if maxes[i] > tol && !(species_name in exclude)
            plot(time_filtered, xs_filtered[i,:], label=species_name)
        end
    end
    legend()
    xlabel("Time in Sec")
    ylabel("Concentration")
end

# %%
exclude_species = ["H2O"]
plotX(ssys.sims[1], 1e-10, 1e3, exclude_species)
xscale("log")
yscale("log")
xlim(1e-8, 1e3)
ylim(1e-16, 5)
title("Liquid-phase Mole Fractions vs. Time on Ag111@-1.0V")
gcf()

# %%
exclude_species = ["H2O"]
plotC(ssys.sims[1], 1e-12, 1e3, exclude_species)
xscale("log")
yscale("log")
xlim(1e-8, 1e3)
ylim(1e-16, 1e2)
title("Liquid-phase Concentrations vs. Time on Ag111@-1.0V")
gcf()

# %%
exclude_species = ["H2O"]
plotX(ssys.sims[2], 1e-4, 1e3, exclude_species)
xscale("log")
yscale("log")
xlim(1e-12, 1e3)
ylim(1e-6, 5)
title("Surface Mole Fractions vs. Time on Ag111@-0.5V")
gcf()

# %%
getfluxdiagram(ssys,1e3;speciesratetolerance=1e-8)

# %%
function plotROP(ssys,name,t;N=0,tol=0.01)
    clf()
    rop = rops(ssys, name, t)
    inds = rop.nzind[reverse(sortperm(abs.(rop.nzval)))]
    if N == 0
        N = length(inds)
    elseif N > length(inds)
        N = length(inds)
    end
    inds = inds[1:N]
    mval = abs(rop[inds[1]])
    minval = mval*tol
    k = 1
    while k < length(inds) && abs(rop[inds[k]]) >= minval
        k += 1
    end
    inds = inds[1:k]
    net_rops = sum(rop[inds])
    println("Net ROPs for species $name is: $net_rops")

    for (i, j) in enumerate(inds)
        println("Showing the reaction with $i th highest ROP for species $name:")
        println(getrxnstr(ssys.reactions[j]))
        println("ROP = ", rop[inds[i]])
        println(ssys.reactions[j].kinetics)
    end

    xs = Array{Float64,1}(1:length(inds))
    barh(xs,reverse(rop[inds]))
    yticks(xs,reverse(getrxnstr.(ssys.reactions[inds])))
    xlabel("Production/Loss Rate mol/s")
    gcf()
end

# %%
function PrintKinDetail(inter, speciesname)
    println("Showing Kinetics details for reactions involving species $speciesname\n")
    for (i,rxn) in enumerate(inter.reactions)
        flag = false
        for j = 1:length(rxn.reactants)
            if rxn.reactants[j].name == speciesname
                flag = true
            end
        end
        for j = 1:length(rxn.products)
            if rxn.products[j].name == speciesname
                flag = true
            end
        end
        if flag
            println(getrxnstr(rxn))
            println(rxn.kinetics)
            kf = inter.kfs[i]
            krev = inter.krevs[i]
            kc = kf/krev
            println("kf = $kf")
            println("krev = $krev")
            println("Kc = $kc\n")
        end
    end
end

# %%
"""
Integrates the ROP in the boundary layer and computes the concentration
"""
function get_boundary_layer_concentration(sim,t,spc,Vbl,C_0)
    intg,err = quadgk(x -> sum(rops(sim,spc,t)), 0, t);
    return C_0 + intg ./ Vbl;
end

# %%
# Logarithmic time scale
t_vals = 10 .^ range(-12, stop=3, length=1000);

# Compute ROP over time
ROP_vals = [sum(rops(ssys, "O=CO", t)) for t in t_vals];
# Compute boundary layer accumulation by integration
Cbl_vals = [get_boundary_layer_concentration(ssys, t, "O=CO", V_bl, C_default) for t in t_vals];

# %%
# Plots the ROP of O=CO
clf()

plot(t_vals, ROP_vals)

xscale("log")
xlim(1e-11,1e3)
yscale("log")
ylim(1e-8,1e2)
xlabel("Time (s)")
ylabel("Rate of Progress (mol/s)")
legend()
tight_layout()
gcf()

# %%
# Plots the Boundary Layer Concentration of O=CO from ROP Integration WITHOUT Diffusion Flux Into Reservoir
clf()

plot(t_vals, Cbl_vals)

xscale("log")
xlim(1e-11,1e3)
yscale("log")
ylim(1e-13,1e1)
xlabel("Time (s)")
ylabel("Concentration (mol/m^3)")
title("Boundary Layer Accumulation of O=CO from ROP Integration")
legend()
tight_layout()
gcf()

# %%
rops(ssys,"O=CO",1)

# %%
fieldnames(typeof(ssys))

# %%
plotROP(ssys, "O=CO",1e-8;N=15,tol=0.0)

# %%
plotROP(ssys, "proton",1;N=15,tol=0.0)

# %%
plotROP(ssys, "O=CO",1;N=15,tol=0.0)

# %%
plotROP(ssys,"CH2O2X",1e-8;N=15,tol=0.0)

# %%
plotROP(ssys,"CHO2X",1;N=10,tol=0.0)

# %%
plotROP(ssys,"CO2HX",1;N=10,tol=0.0)

# %%
plotROP(ssys,"OX",1;N=10,tol=0.0)

# %%
plotROP(ssys,"OCX",1.0e-6)

# %%
concentrations(ssys,"O=CO",1)

# %%
ssys.sol

# %%
ssys.sol.u[end]

# %%
ssys.interfaces[1].rxnarray

# %%
