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
using DataStructures

# %%
rms_file = "/home/danieltori/CO2_RR_RMG/Model/PES_plot/chem114.rms"
T = 300.0

# choose PES basis: "G" for Gibbs, "H" for enthalpy
energy_mode = "H"

# %%
phaseDict = readinput(rms_file)

println("Available phases:")
println(keys(phaseDict))

# %%
gas_phase = phaseDict["gas"]
surf_phase = phaseDict["surface"]   

gas_species = gas_phase["Species"]
surf_species = surf_phase["Species"]

spcs = vcat(gas_species, surf_species)

# collect all reactions across all phases (including cross-phase like gas+surface)
rxns = ElementaryReaction[]
for (key, val) in phaseDict
    if val isa Dict && haskey(val, "Reactions")
        append!(rxns, val["Reactions"])
    end
end

println("Gas species    = ", length(gas_species))
println("Surface species = ", length(surf_species))
println("Total species   = ", length(spcs))
println("Total reactions  = ", length(rxns))

# %%
println("First 15 gas species:")
for i in 1:min(15, length(gas_species))
    println(i, "  ", gas_species[i].name)
end

println("\nFirst 20 surface species:")
for i in 1:min(20, length(surf_species))
    println(i, "  ", surf_species[i].name)
end

# %%
spc_dict = Dict{String,Any}()

for spc in spcs
    spc_dict[spc.name] = spc
end

println("Lookup built for ", length(spc_dict), " species")

# %%
Jmol_to_eV = 1.0 / 96485.00

function species_energy_eV(spc_name, spc_dict, T; mode="H")
    spc = spc_dict[spc_name]

    if mode == "H"
        E = getEnthalpy(spc.thermo, T)
    elseif mode == "G"
        E = getGibbs(spc.thermo, T)
    else
        error("mode must be \"H\" or \"G\"")
    end

    return E * Jmol_to_eV
end

# %%
test_name = "CO2"

println("Species: ", test_name)
println("H = ", species_energy_eV(test_name, spc_dict, T, mode="H"), " eV")
println("G = ", species_energy_eV(test_name, spc_dict, T, mode="G"), " eV")

# %%
function state_energy_eV(species_list, spc_dict, T; mode="H")
    total = 0.0
    for sp in species_list
        total += species_energy_eV(sp, spc_dict, T, mode=mode)
    end
    return total
end

# %%
function counter_list(lst)
    c = counter(String[])
    for x in lst
        push!(c, x)
    end
    return c
end

function reaction_species_lists(rxn)
    reactants = [sp.name for sp in rxn.reactants]
    products  = [sp.name for sp in rxn.products]
    return reactants, products
end

function find_rxn(reactants, products, rxns)
    target_r = counter_list(reactants)
    target_p = counter_list(products)

    for (i, rxn) in enumerate(rxns)
        rlist, plist = reaction_species_lists(rxn)

        if counter_list(rlist) == target_r && counter_list(plist) == target_p
            println(i, ": ", join(rlist, " + "), " <=> ", join(plist, " + "))
            return i, false
        elseif counter_list(rlist) == target_p && counter_list(plist) == target_r
            println(i, ": ", join(rlist, " + "), " <=> ", join(plist, " + "), "   [swapped]")
            return i, true
        end
    end

    error("Reaction not found for $(reactants) => $(products)")
end

# %%
function Ea_eV(rxn)
    Ea_val = 0.0

    if hasproperty(rxn, :rate) && hasproperty(rxn.rate, :Ea)
        Ea_val = rxn.rate.Ea
    elseif hasproperty(rxn, :kinetics) && hasproperty(rxn.kinetics, :Ea)
        Ea_val = rxn.kinetics.Ea
    else
        return 0.0
    end

    # if Ea is small (< 20), assume already eV
    # otherwise assume J/mol
    if Ea_val < 20
        return Float64(Ea_val)
    else
        return Float64(Ea_val) * Jmol_to_eV
    end
end

# %%
path_A = [
    (["vacantX","CO2"], ["CO2X"]),
    (["CO2X","proton"], ["CHO2X"]),
    (["CHO2X","proton"], ["HCOOH","vacantX"])
]

path_B = [
    (["vacantX","CO2"], ["CO2X"]),
    (["CO2X","proton"], ["CO2HX"]),
    (["CO2HX","proton"], ["HCOOH","vacantX"])
]

path_C = [
    (["vacantX","CO2"], ["CO2X"]),
    (["CO2X","proton"], ["CO2HX"]),
    (["CO2HX","proton"], ["H2O","OCX"])
]

path_D = [
    (["OCX","OCX"], ["XCOXCO"])
]

path_E = [
    (["OCX","OCX","proton"], ["OCX","CHOX"])
]

# %%
function build_rxn_path(path, rxns)
    rxn_path = Tuple{Int,Bool}[]
    for step in path
        idx, swapped = find_rxn(step[1], step[2], rxns)
        push!(rxn_path, (idx, swapped))
    end
    return rxn_path
end

rxn_path_A = build_rxn_path(path_A, rxns)
rxn_path_B = build_rxn_path(path_B, rxns)
rxn_path_C = build_rxn_path(path_C, rxns)
rxn_path_D = build_rxn_path(path_D, rxns)
rxn_path_E = build_rxn_path(path_E, rxns)

println("Path A = ", rxn_path_A)
println("Path B = ", rxn_path_B)
println("Path C = ", rxn_path_C)
println("Path D = ", rxn_path_D)
println("Path E = ", rxn_path_E)

# %%
function pathway_profile(path, rxn_path, rxns, spc_dict, T; mode="H")
    rows = []

    E_ref = state_energy_eV(path[1][1], spc_dict, T, mode=mode)

    for (step_no, (step, rxn_info)) in enumerate(zip(path, rxn_path))
        reactants = step[1]
        products = step[2]

        rxn_idx, swapped = rxn_info
        rxn = rxns[rxn_idx]

        E_reac_abs = state_energy_eV(reactants, spc_dict, T, mode=mode)
        E_prod_abs = state_energy_eV(products, spc_dict, T, mode=mode)

        E_reac = E_reac_abs - E_ref
        E_prod = E_prod_abs - E_ref
        ΔE = E_prod - E_reac

        Ea = Ea_eV(rxn)

        E_TS = swapped ? (E_prod + Ea) : (E_reac + Ea)

        push!(rows, Dict(
            "step" => step_no,
            "rxn_idx" => rxn_idx,
            "swapped" => swapped,
            "reactants" => reactants,
            "products" => products,
            "E_reac" => E_reac,
            "E_prod" => E_prod,
            "ΔE" => ΔE,
            "Ea" => Ea,
            "E_TS" => E_TS
        ))
    end

    return rows
end

# %%
rows_A = pathway_profile(path_A, rxn_path_A, rxns, spc_dict, T, mode=energy_mode)

for row in rows_A
    println("Step      : ", row["step"])
    println("Rxn index : ", row["rxn_idx"], row["swapped"] ? "  [swapped]" : "")
    println("Reactants : ", join(row["reactants"], " + "))
    println("Products  : ", join(row["products"], " + "))
    println("E_reac    = ", round(row["E_reac"], digits=4), " eV")
    println("E_prod    = ", round(row["E_prod"], digits=4), " eV")
    println("ΔE        = ", round(row["ΔE"], digits=4), " eV")
    println("Ea        = ", round(row["Ea"], digits=4), " eV")
    println("E_TS      = ", round(row["E_TS"], digits=4), " eV")
end

# %%
function clean_label(lst)
    # normalize OCX capitalization in labels
    pretty = [lowercase(sp) == "ocx" ? "OCX" : sp for sp in lst]
    return join(pretty, " + ")
end

function state_label(lst)
    label = clean_label(lst)
    # keep this specific label horizontal for Path E plots
    if lowercase(replace(label, " " => "")) == "ocx+ocx+proton"
        return label
    end
    if length(label) > 14
        return replace(label, " + " => "\n+\n")
    end
    return label
end

function plot_pes(rows; title="PES", color="#1f5aa6", fontsize=14, fontweight="bold", energy_mode="H")
    state_x = collect(0.0:2.0:(2.0 * length(rows)))
    state_y = Float64[rows[1]["E_reac"]]
    append!(state_y, [row["E_prod"] for row in rows])

    state_labels = String[state_label(rows[1]["reactants"])]
    append!(state_labels, [state_label(row["products"]) for row in rows])

    ts_x = collect(1.0:2.0:(2.0 * length(rows) - 1.0))
    ts_y = Float64[row["E_TS"] for row in rows]
    ts_labels = ["TS" * string(row["step"]) for row in rows]

    ymin = min(minimum(state_y), minimum(ts_y))
    ymax = max(maximum(state_y), maximum(ts_y))
    span = max(ymax - ymin, 0.5)
    top_pad = 0.16 * span
    bottom_pad = 0.18 * span
    ts_label_offset = 0.06 * span
    bar_halfwidth = 0.42
    label_size = max(fontsize + 2, 16)
    axis_label_size = max(fontsize + 4, 18)

    fig = figure(figsize=(10.5, 5.8), dpi=160)
    ax = gca()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(true)
    ax.grid(axis="y", color="#e8edf3", linewidth=0.9)
    ax.axhline(0.0, color="#b8c2cc", linewidth=1.0, linestyle="--")

    for i in eachindex(state_x)
        hlines(state_y[i], state_x[i] - bar_halfwidth, state_x[i] + bar_halfwidth, color=color, linewidth=3.0, zorder=4)
    end

    plot(state_x, state_y, linestyle="none", marker="o", markersize=4.5, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    for i in eachindex(ts_x)
        plot([state_x[i] + bar_halfwidth, ts_x[i], state_x[i + 1] - bar_halfwidth], [state_y[i], ts_y[i], state_y[i + 1]], color=color, linewidth=1.8, zorder=3)
        plot(ts_x[i], ts_y[i], linestyle="none", marker="o", markersize=6.5, markerfacecolor="#c2410c", markeredgecolor="white", markeredgewidth=0.9, zorder=6)
        text(ts_x[i], ts_y[i] + ts_label_offset, ts_labels[i], ha="center", va="bottom", fontsize=max(fontsize - 4, 8), fontweight=fontweight, color="#7c2d12")
    end

    ax.set_xticks(state_x)
    ax.set_xticklabels(state_labels, fontsize=label_size, fontweight=fontweight, color="black")
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", labelsize=label_size)
    for tick in ax.get_yticklabels()
        tick.set_fontweight(fontweight)
        tick.set_color("black")
    end

    xlabel("Reaction Coordinate", fontsize=axis_label_size, fontweight=fontweight, color="black")
    ylabel(energy_mode == "H" ? "Relative Enthalpy (eV)" : "Relative Gibbs Energy (eV)", fontsize=axis_label_size, fontweight=fontweight, color="black")
    PythonPlot.title(title, fontsize=fontsize, fontweight=fontweight, pad=12)

    xlim(minimum(state_x) - 0.7, maximum(state_x) + 0.7)
    ylim(ymin - bottom_pad, ymax + top_pad)

    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_color("#334155")

    tight_layout()
    return gcf()
end

# %%
plot_pes(rows_A, title="Path A: Formate Route", color="#1d4ed8", fontweight="bold", fontsize=14, energy_mode=energy_mode)

# %%
function full_path_summary(rows)
    lines = String[]
    for row in rows
        push!(lines, string(
            "Step ", row["step"], ": ",
            clean_label(row["reactants"]), " → ", clean_label(row["products"]),
            "   |   Ea = ", round(row["Ea"], digits=2), " eV",
            "   |   ΔE = ", round(row["ΔE"], digits=2), " eV"
        ))
    end
    return join(lines, "\n")
end

function plot_pes_fullpath(rows; title="Energy Diagram", color="#1d4ed8", fontsize=16, fontweight="black", energy_mode="H")
    weight = fontweight == "bold" ? "black" : fontweight

    state_x = collect(0.0:2.0:(2.0 * length(rows)))
    state_y = Float64[rows[1]["E_reac"]]
    append!(state_y, [row["E_prod"] for row in rows])

    state_labels = String[state_label(rows[1]["reactants"])]
    append!(state_labels, [state_label(row["products"]) for row in rows])

    ts_x = collect(1.0:2.0:(2.0 * length(rows) - 1.0))
    ts_y = Float64[row["E_TS"] for row in rows]

    ymin = min(minimum(state_y), minimum(ts_y))
    ymax = max(maximum(state_y), maximum(ts_y))
    span = max(ymax - ymin, 0.5)
    top_pad = 0.28 * span
    bottom_pad = 0.22 * span
    label_size = max(fontsize + 3, 19)
    axis_label_size = max(fontsize + 5, 21)
    note_size = max(fontsize + 1, 15)
    summary_size = max(fontsize + 1, 15)
    peak_label_offset = 0.06 * span
    bar_halfwidth = 0.40

    fig = figure(figsize=(16.5, 9.5), dpi=170)
    ax = gca()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(true)
    ax.grid(axis="y", color="#e5e7eb", linewidth=1.0)
    ax.axhline(0.0, color="#94a3b8", linewidth=1.3, linestyle="--")

    for i in eachindex(state_x)
        hlines(state_y[i], state_x[i] - bar_halfwidth, state_x[i] + bar_halfwidth, color=color, linewidth=3.8, zorder=4)
    end

    plot(state_x, state_y, linestyle="none", marker="o", markersize=5.8, markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.1, zorder=5)

    for i in eachindex(ts_x)
        plot([state_x[i] + bar_halfwidth, ts_x[i], state_x[i + 1] - bar_halfwidth], [state_y[i], ts_y[i], state_y[i + 1]], color=color, linewidth=2.4, zorder=3)
        plot(ts_x[i], ts_y[i], linestyle="none", marker="o", markersize=8.0, markerfacecolor="#c2410c", markeredgecolor="white", markeredgewidth=1.1, zorder=6)

        x_ea = ts_x[i] - 0.18
        ax.annotate("", xy=(x_ea, ts_y[i]), xytext=(x_ea, state_y[i]), arrowprops=Dict("arrowstyle" => "<->", "color" => "#7c2d12", "linewidth" => 2.0))
        ax.text(ts_x[i], ts_y[i] + peak_label_offset, string("Ea = ", round(rows[i]["Ea"], digits=2), " eV"), ha="center", va="bottom", fontsize=note_size, fontweight=weight, color="#7c2d12", bbox=Dict("boxstyle" => "round,pad=0.32", "facecolor" => "white", "edgecolor" => "#fdba74", "linewidth" => 1.3))

        x_de = state_x[i + 1] + 0.18
        ax.annotate("", xy=(x_de, state_y[i + 1]), xytext=(x_de, state_y[i]), arrowprops=Dict("arrowstyle" => "<->", "color" => "#1d4ed8", "linewidth" => 2.0))
        ax.text(x_de + 0.10, (state_y[i] + state_y[i + 1]) / 2, string("ΔE = ", round(rows[i]["ΔE"], digits=2), " eV"), ha="left", va="center", fontsize=note_size, fontweight=weight, color="#1d4ed8", bbox=Dict("boxstyle" => "round,pad=0.30", "facecolor" => "white", "edgecolor" => "#93c5fd", "linewidth" => 1.2))
    end

    ax.set_xticks(state_x)
    ax.set_xticklabels(state_labels, fontsize=label_size, fontweight=weight, color="black")
    ax.tick_params(axis="x", length=0, pad=15, width=1.4)
    ax.tick_params(axis="y", labelsize=label_size, width=1.4)
    for tick in ax.get_yticklabels()
        tick.set_fontweight(weight)
        tick.set_color("black")
    end

    xlabel("Reaction Coordinate", fontsize=axis_label_size, fontweight=weight, color="black")
    ylabel(energy_mode == "G" ? "Relative Gibbs Energy (eV)" : "Relative Enthalpy (eV)", fontsize=axis_label_size, fontweight=weight, color="black")
    PythonPlot.title(title, fontsize=fontsize + 4, fontweight=weight, pad=14)

    xlim(minimum(state_x) - 0.7, maximum(state_x) + 0.9)
    ylim(ymin - bottom_pad, ymax + top_pad)

    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_linewidth(1.4)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.28)
    fig.text(0.09, 0.05, full_path_summary(rows), ha="left", va="bottom", fontsize=summary_size, fontweight=weight, color="#1e293b", linespacing=1.5, bbox=Dict("boxstyle" => "round,pad=0.55", "facecolor" => "#f8fafc", "edgecolor" => "#cbd5e1"))
    return gcf()
end

plot_pes_fullpath(rows_A, title="Path A: Energy Diagram", color="#1d4ed8", fontweight="black", fontsize=16, energy_mode=energy_mode)

# %%
rows_B = pathway_profile(path_B, rxn_path_B, rxns, spc_dict, T, mode=energy_mode)

for row in rows_B
    println("Step      : ", row["step"])
    println("Rxn index : ", row["rxn_idx"], row["swapped"] ? "  [swapped]" : "")
    println("Reactants : ", join(row["reactants"], " + "))
    println("Products  : ", join(row["products"], " + "))
    println("E_reac    = ", round(row["E_reac"], digits=4), " eV")
    println("E_prod    = ", round(row["E_prod"], digits=4), " eV")
    println("ΔE        = ", round(row["ΔE"], digits=4), " eV")
    println("Ea        = ", round(row["Ea"], digits=4), " eV")
    println("E_TS      = ", round(row["E_TS"], digits=4), " eV")
end

# %%
plot_pes(rows_B, title="Path B: Formate Route", color="red", energy_mode=energy_mode)

# %%
plot_pes_fullpath(rows_B, title="Path B: Energy Diagram", color="#dc2626", fontweight="black", fontsize=16, energy_mode=energy_mode)

# %%
rows_C = pathway_profile(path_C, rxn_path_C, rxns, spc_dict, T, mode=energy_mode)

for row in rows_C
    println("Step      : ", row["step"])
    println("Rxn index : ", row["rxn_idx"], row["swapped"] ? "  [swapped]" : "")
    println("Reactants : ", join(row["reactants"], " + "))
    println("Products  : ", join(row["products"], " + "))
    println("E_reac    = ", round(row["E_reac"], digits=4), " eV")
    println("E_prod    = ", round(row["E_prod"], digits=4), " eV")
    println("ΔE        = ", round(row["ΔE"], digits=4), " eV")
    println("Ea        = ", round(row["Ea"], digits=4), " eV")
    println("E_TS      = ", round(row["E_TS"], digits=4), " eV")
end

# %%
plot_pes(rows_C, title="Path C: CO branch", color="purple", fontsize=14, fontweight="bold", energy_mode=energy_mode)

# %%
plot_pes_fullpath(rows_C, title="Path C: Energy Diagram", color="purple", fontweight="black", fontsize=16, energy_mode=energy_mode)

# %%
rows_D = pathway_profile(path_D, rxn_path_D, rxns, spc_dict, T, mode=energy_mode)

for row in rows_D
    println("Step      : ", row["step"])
    println("Rxn index : ", row["rxn_idx"], row["swapped"] ? "  [swapped]" : "")
    println("Reactants : ", join(row["reactants"], " + "))
    println("Products  : ", join(row["products"], " + "))
    println("E_reac    = ", round(row["E_reac"], digits=4), " eV")
    println("E_prod    = ", round(row["E_prod"], digits=4), " eV")
    println("ΔE        = ", round(row["ΔE"], digits=4), " eV")
    println("Ea        = ", round(row["Ea"], digits=4), " eV")
    println("E_TS      = ", round(row["E_TS"], digits=4), " eV")
end

# %%
plot_pes(rows_D, title="Path D: C-C coupling", color="green", energy_mode=energy_mode)

# %%
plot_pes_fullpath(rows_D, title="Path D: Energy Diagram", color="green", fontweight="black", fontsize=16, energy_mode=energy_mode)

# %%
rows_E = pathway_profile(path_E, rxn_path_E, rxns, spc_dict, T, mode=energy_mode)

for row in rows_E
    println("Step      : ", row["step"])
    println("Rxn index : ", row["rxn_idx"], row["swapped"] ? "  [swapped]" : "")
    println("Reactants : ", join(row["reactants"], " + "))
    println("Products  : ", join(row["products"], " + "))
    println("E_reac    = ", round(row["E_reac"], digits=4), " eV")
    println("E_prod    = ", round(row["E_prod"], digits=4), " eV")
    println("ΔE        = ", round(row["ΔE"], digits=4), " eV")
    println("Ea        = ", round(row["Ea"], digits=4), " eV")
    println("E_TS      = ", round(row["E_TS"], digits=4), " eV")
end

# %%
plot_pes(rows_E, title="Path E: C-C Hydrogenation", color="darkorange", energy_mode=energy_mode)

# %%
plot_pes_fullpath(rows_E, title="Path E: Energy Diagram", color="darkorange", fontweight="black", fontsize=16, energy_mode=energy_mode)

# %%
function summarize_rows(rows, name)
    println("\n", name)
    for row in rows
        println(
            "step ", row["step"],
            " | rxn ", row["rxn_idx"],
            " | Ea = ", round(row["Ea"], digits=3), " eV",
            " | ΔE = ", round(row["ΔE"], digits=3), " eV"
        )
    end
end

summarize_rows(rows_A, "Path A")
summarize_rows(rows_B, "Path B")
summarize_rows(rows_C, "Path C")
summarize_rows(rows_D, "Path D")
summarize_rows(rows_E, "Path E")
