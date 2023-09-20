#=
# ***Modeling with Non-available components with SIR Model***
---

=#

# Import packages
using ODEPoissonProcesses

using DataFrames
using Serialization
using Random
using Serialization
using Plots
using Plots.PlotMeasures
using StatsPlots
using StatsBase
using LaTeXStrings

seed = 1
_dir = @__DIR__;
if ~Base.isdir("$(_dir)/../results/img")
    Base.mkdir("$(_dir)/../results/img")
end

# ## ***1. Load simulated data and results***
open("$(_dir)/../results/exp-5.2.4-Infected_seed$(seed).dat", "r") do io
    global results = Serialization.deserialize(io)
end;

model_names = ["sir_pois", "i_pois"]
ode_figs = []
event_figs = []

data = results["data"]

mod_sir, chain_sir = results["sir_pois"].mod, results["sir_pois"].chain
mod_i, chain_i = results["i_pois"].mod, results["i_pois"].chain

X̂s_sir = cat(ODEPoissonProcesses.get_posterior_X̂_samples(mod_sir, chain_sir)..., dims=3);
X̂s_i = cat(ODEPoissonProcesses.get_posterior_X̂_samples(mod_i, chain_i)..., dims=3);

Zs_sir = exp.(X̂s_sir)
Zs_i = exp.(X̂s_i)

θ_samples_sir = hcat(ODEPoissonProcesses.get_posterior_θ_samples(mod_sir, chain_sir)...);
θ_samples_i = hcat(ODEPoissonProcesses.get_posterior_θ_samples(mod_i, chain_i)...);

a_samples_sir = θ_samples_sir[1,:]
a_samples_i = θ_samples_i[1,:]
b_samples_sir = θ_samples_sir[2,:]
b_samples_i = θ_samples_i[2,:];

# ## ***2. Plot results***
# Plot data and estimated dynamics compared to ground-truth
p = plot(
    ODEPoissonProcesses.eventplot(data, alpha=0.75, xlims=(0, 1.)),
    ODEPoissonProcesses.eventplot(data, alpha=0.75, xlims=(0, 1.), visible_components=[2]),
    (
        ODEPoissonProcesses.draw_gt_and_est_ode(data, mod_sir, Zs_sir, qlow=0.125, qhigh=0.875, extrapolation=false);
        ylims!(0,15);
        xlabel!("time");
        ylabel!("z = exp(x)");
        annotate!(0.2, 13, text("S", :blue, "Times"));
        annotate!(0.5, 9, text("I", :red, "Times"));
        annotate!(0.8, 11, text("R", :mediumseagreen, "Times"))
    ),
    (
        ODEPoissonProcesses.draw_gt_and_est_ode(data, mod_i, Zs_i, qlow=0.125, qhigh=0.875, extrapolation=false);
        ylims!(0,15);
        xlabel!("time");
        ylabel!("z = exp(x)");
        annotate!(0.2, 13, text("S", :blue, "Times"));
        annotate!(0.5, 9, text("I", :red, "Times"));
        annotate!(0.8, 11, text("R", :mediumseagreen, "Times"))
    ),
    grid=:none,
    layout=Plots.grid(2,2, heights=[0.2, 0.8]),
    size=(800,350),
    fontfamily="Times",
    labelfontsize=16,
    tickfontsize=16,
    margin=7mm
    )

# Save figure
Plots.pdf(p, "$(_dir)/../results/img/exp-5.2.4-Infected_dynamics_seed$(seed).pdf");

# Plot estimated θ compared to ground-truth
p = plot(
    xticks=([1.5,4.5], ["a", "b"]),
    xlims=(0, 6),
    grid=:none,
    legend=:outerright,
    foreground_color_legend = nothing,
    size=(900,200),
    legendfontsize=16,
    tickfontsize=16,
    bottommargin=3mm,
    fontfamily="Times"
)
boxplot!(ones(1000), a_samples_sir, c=:black, outliers=false, label="Observations: S, I, R")
boxplot!(ones(1000)*2, a_samples_i, c=:gray, outliers=false, label="Only I observed")
boxplot!(ones(1000)*4, b_samples_sir, c=:black, outliers=false, label=:none)
boxplot!(ones(1000)*5, b_samples_i, c=:gray, outliers=false, label=:none)
plot!([0.5,2.5], [data.θ.a,data.θ.a], c=:red, lw=2, ls=:dash, label="Ground-truth")
plot!([3.5,5.5], [data.θ.b,data.θ.b], c=:red, lw=2, ls=:dash, label=:none)

# Save figure
Plots.pdf(p, "$(_dir)/../results/img/exp-5.2.4-Infected_theta_seed$(seed).pdf");
