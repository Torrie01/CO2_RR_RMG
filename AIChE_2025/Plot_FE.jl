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
using CSV, DataFrames, PythonPlot

# %%
plt = PythonPlot

# %%
df = CSV.read("final_molefractions.csv", DataFrame)

# %%
# Define parameters
species_order = [
                 "H2",
                 "CO",
                 "O=CO",
                 "CO-2",
                 "CCO",
                 "C=O"
                 ]
labels = [
          "H2",
          "CO",
          "Formate",
          "Methanol",
          "Ethanol",
          "Formaldehyde"
          ]
colors = [
          "#440154",
          "#FDAE61",
          "#31688E",
          "#35B779",
          "#FDE725",
          "#A5D32F"
          ];


# %%
potentials = names(df)[2:end]  # all potential columns
n_species = length(species_order)

# %%
# Split sim vs exp (first 5 rows = sim, next 5 = exp)
sim = df[1:n_species, :]
exp = df[n_species+1:end, :];

# %%
sim_data = Matrix(sim[[findfirst(==(s), sim.Species) for s in species_order], potentials]).*100
exp_data = Matrix(exp[[findfirst(==(s), exp.Species) for s in species_order], potentials]).*100

# %%
plt.figure(figsize=(8,4.5))
x = collect(1:length(potentials))
bar_width = 0.3;
bar_gap = 0.1;

# %%
# Left bars (simulation)
bottom_sim = zeros(length(potentials))
for (i, sp) in enumerate(species_order)
    plt.bar(x .- (bar_width/2 + bar_gap/2), sim_data[i, :],
            bar_width, bottom=bottom_sim,
            color=colors[i], label=labels[i])
    bottom_sim .+= sim_data[i, :]
end

# %%
# Right bars (experiment)
bottom_exp = zeros(length(potentials))
for (i, sp) in enumerate(species_order)
    plt.bar(x .+ (bar_width/2 + bar_gap/2), exp_data[i, :],
            bar_width, bottom=bottom_exp,
            color=colors[i], alpha=0.9,
            label="_nolegend_")
    bottom_exp .+= exp_data[i, :]
end

# %%
plt.xticks(x, potentials)
plt.xlabel("Applied Potential (V vs RHE)", fontweight="bold")
plt.ylabel("Faradaic Efficiency (%)", fontweight="bold")
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.ylim(0.0, 108)
plt.title("Simulation vs Experimental Faradaic Efficiencies on Ag(111)", pad = 12, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.text(1.04, 0.5,
         "Left = Simulation\n\nRight = Experiment",
         fontsize=9, fontweight="bold",
         ha="left", va="center",
         transform=plt.gca().transAxes)
plt.tight_layout()

# %%
plt.savefig("plt_full.png", dpi=300)
display("image/png", read("plt_full.png"))

# %%
selected_species = ["O=CO", "CO-2", "CCO"]
selected_labels  = ["Formate", "Methanol", "Ethanol"]
selected_colors = ["#31688E", "#35B779", "#FDE725",];

# %%
sel_idx = [findfirst(==(s), exp.Species) for s in selected_species]
exp_sel = exp[sel_idx, :]
exp_data_sel = Matrix(exp_sel[:, potentials]).*100;

# %%
plt.figure(figsize=(6,4))
x = collect(1:length(potentials))
bar_width = 0.3
bottom_sim = zeros(length(potentials))
for (i, sp) in enumerate(selected_species)
    plt.bar(x .- (bar_width/2), exp_data_sel[i, :],
            bar_width, bottom=bottom_sim,
            color=selected_colors[i],
            label=selected_labels[i])
    bottom_sim .+= exp_data_sel[i, :]
end

plt.xticks(x, potentials)
plt.xlabel("Applied Potential (V vs RHE)", fontweight="bold")
plt.ylabel("Faradaic Efficiency (%)", fontweight="bold")
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
plt.ylim(0.0, 10)
plt.title("Experimental Faradaic Efficiencies on Ag(111)", pad = 12, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# %%
plt.savefig("plt_partial.png", dpi=300)
display("image/png", read("plt_partial.png"))

# %%
