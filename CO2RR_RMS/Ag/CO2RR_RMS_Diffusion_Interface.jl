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
#     display_name: Julia 1.9.1
#     language: julia
#     name: julia-1.9
# ---

# %%
using DifferentialEquations
using ReactionMechanismSimulator
using PyPlot

# %%
outdict = readinput("chem300.rms")

# %%
boundarylayerspcs = outdict["gas"]["Species"]
boundarylayerrxns = outdict["gas"]["Reactions"]
surfspcs = outdict["surface"]["Species"]
surfrxns = outdict["surface"]["Reactions"]
interfacerxns = outdict[Set(["surface", "gas"])]["Reactions"]
solv = outdict["Solvents"][1];

# %%
sitedensity = 2.292e-5; # Ag111
boundarylayer = IdealDiluteSolution(boundarylayerspcs,boundarylayerrxns,solv,name="boundarylayeruid",diffusionlimited=true);
surf = IdealSurface(surfspcs,surfrxns,sitedensity,name="surface");

# %%
initialcondsboundarylayer = Dict(["proton"=>10.0^-4,
                                  "CO2"=>10.0^-3*10^6,
                                  "V"=>1.0e-3,"T"=>300,"Phi"=>0.0,"d"=>0.0]);
initialcondsreservoir = Dict(["proton"=>10.0^-4,
                              "CO2"=>10.0^-3*10^6,
                              "V"=>1.0,"T"=>300]);
AVratio = 1e5;
initialcondssurf = Dict(["CO2X"=>0.4*sitedensity*AVratio,
        "CHO2X"=>0.1*sitedensity*AVratio,
        "CO2HX"=>0.1*sitedensity*AVratio,
        "OX"=>0.1*sitedensity*AVratio,
        "OCX"=>0.1*sitedensity*AVratio,
        "vacantX"=>0.1*sitedensity*AVratio,
        "CH2O2X"=>0.05*sitedensity*AVratio,
        "CHOX"=>0.04*sitedensity*AVratio,
        "CH2OX"=>0.01*sitedensity*AVratio,
        "A"=>1.0*AVratio,"T"=>300,"Phi"=>-1.0]);

# %%
domainboundarylayer, y0boundarylayer, pboundarylayer = ConstantTVDomain(phase=boundarylayer, initialconds=initialcondsboundarylayer);
domaincat,y0cat,pcat = ConstantTAPhiDomain(phase=surf,
    initialconds=initialcondssurf);

# %%
inter,pinter = ReactiveInternalInterfaceConstantTPhi(domainboundarylayer,
  domaincat,interfacerxns,298.15,AVratio*1.0);

# %%
# start with 1mm layer thickness
diffusionlayer = ConstantReservoirDiffusion(domainboundarylayer, initialcondsreservoir, AVratio*1.0, 1e-3);

# %%
interfaces = [inter, diffusionlayer];

# %%
@time react,y0,p = Reactor((domainboundarylayer,domaincat), (y0boundarylayer,y0cat), (0.0, 1.0e2), interfaces, (pboundarylayer,pcat,pinter));


# %%
@time sol = solve(react.ode,DifferentialEquations.CVODE_BDF(),abstol=1e-16,reltol=1e-8);

# %%
sol

# %%
sol.retcode

# %%
ssys = SystemSimulation(sol,(domainboundarylayer,domaincat,), interfaces,p);

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
concentrations(ssys.sims[1])

# %%
concentrations(ssys.sims[2])

# %%
exclude_species = ["H2O"]
plotX(ssys.sims[1], 1e-6, 1e2, exclude_species)
xscale("log")
yscale("log")
xlim(1e-12, 1e2)
ylim(1e-9, 5)
title("Evolution of Liquid-phase Mole Fractions vs. Time on Ag111@-1.5V")
gcf()

# %%
exclude_species = ["H2O"]
plotC(ssys.sims[1], 1e-6, 1e2, exclude_species)
xscale("log")
yscale("log")
xlim(1e-12, 10)
ylim(1e-6, 1e8)
title("Evolution of Liquid-phase Concentrations vs. Time on Ag111@-1.5V")
gcf()

# %%
exclude_species = ["H2O"]
plotC(ssys.sims[2], 1e-6, 1e2, exclude_species)
xscale("log")
yscale("log")
xlim(1e-12, 10)
ylim(1e-6, 1e-4)
title("Evolution of Liquid-phase Concentrations vs. Time on Ag111@-1.5V")
gcf()

# %%
getfluxdiagram(ssys,1e2;speciesratetolerance=1e-6)

# %%
println(ssys.names)

# %%
plotrops(ssys,"CH2O2X",1;N=15,tol=0.0)

# %%
plotrops(ssys,"CHO2X",1;N=10,tol=0.0)

# %%
plotrops(ssys,"CO2HX",1;N=10,tol=0.0)

# %%
plotrops(ssys,"OX",1;N=10,tol=0.0)

# %%
plotrops(ssys,"OCX",1.0e-6)

# %%
for (i,rxn) in enumerate(inter.reactions)
    str = getrxnstr(rxn)
    kf = inter.kfs[i]
    krev = inter.krevs[i]
    Kc = kf/krev
    println(str)
    println("kf = $kf")
    println("krev = $krev")
    println("Kc = $Kc")
end

# %%
for (i,rxn) in enumerate(inter.reactions)
    str = getrxnstr(rxn)
    kf = inter.kfs[i]
    krev = inter.krevs[i]
    Kc = kf/krev
    println(str)
    println("kf = $kf")
    println("krev = $krev")
    println("Kc = $Kc")
end

# %%

# %%
