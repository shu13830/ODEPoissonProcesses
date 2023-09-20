# # ***Evaluation of Extrapolation***

# Import packages
using ODEPoissonProcesses

using DataFrames
using Distributions
using Serialization
using Random
using Serialization
using Plots
using Plots.PlotMeasures
using Printf
using StatsPlots
using StatsBase
using LaTeXStrings

seed = 1
_dir = @__DIR__;
if ~Base.isdir("$(_dir)/../results/img")
    Base.mkdir("$(_dir)/../results/img")
end

# ## ***1. SIR***
# load simulated data and results
open("$(_dir)/../results/exp-5.2.3-SIR_seed$(seed).dat") do io
    global results = Serialization.deserialize(io)
end;

# Evaluate negative log-likelihoods
λ0s = [50.0, 100.0, 1000.0]
model_names = ["cox", "odepois"];

nll_means = Dict(name => Dict() for name in model_names)
nll_stds = Dict(name => Dict() for name in model_names)

for λ0 in λ0s
    res = results[λ0]
    data = res["data"]
    
    M′_samples = ODEPoissonProcesses.sim_extrapolation_eventcounts(data, res[model_names[1]].mod; n_simulations=100);
    
    for name in model_names
        mod, chain = res[name].mod, res[name].chain;
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂′_samples = [X̂[mod.gm.T+1:end, :] for X̂ in X̂_samples]

        nlls = []
        for M′ in M′_samples, X̂′ in X̂′_samples
            Λ̂′ = (mod.gm.λ0/mod.gm.T)' .* exp.(X̂′)
            nll=0
            for (m′, λ′) in zip(M′, Λ̂′)
                nll += -logpdf(Poisson(λ′), m′)
            end
            push!(nlls, sum(nll))
        end

        nll_means[name][λ0] = mean(nlls)
        nll_stds[name][λ0] = std(nlls)
    end
end

# DataFrame of evaluated negative log-likelihoods
df_nlls = DataFrame("Scheme" => model_names)
for λ0 in λ0s
    df_nlls[!, "λ0=$(λ0)"] = [@sprintf("%0.1f", nll_means[name][λ0])*" ± "*@sprintf("%0.1f", nll_stds[name][λ0]) for name in model_names]
end
df_nlls

# Compare ground-truth and estimated modulations
ode_figs = []
event_figs = []
titles = [L"\lambda_0=50", L"\lambda_0=100", L"\lambda_0=1000"]
for (i, λ0) in enumerate(λ0s)
    data = results[λ0]["data"]
    push!(event_figs, ODEPoissonProcesses.eventplot(data, title=titles[i], alpha=0.75, xlims=(0, 1.5)))
    for name in model_names
        res = results[λ0]
        mod, chain = res[name].mod, res[name].chain
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂s = cat(X̂_samples..., dims=3);
        Zs = exp.(X̂s)
        fig = ODEPoissonProcesses.draw_gt_and_est_ode(data, mod, Zs, qlow=0.125, qhigh=0.875)
        push!(ode_figs, fig)
    end
end

p_annotate = plot(
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("synthesized\nevents", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP-GM\n(ours)", "Times", 12))),
    layout=Plots.grid(3,1, heights=[0.19, 0.4, 0.41]),
    size=(100, 280)
    )

p = plot(
    event_figs[1], event_figs[2], event_figs[3],

    (plot(ode_figs[1]; xtick=:none, ylims=(0, 10), ylabel="exp(x)")),
    (plot(ode_figs[3]; ticks=:none, ylims=(0, 10))),
    (plot(ode_figs[5]; ticks=:none, ylims=(0, 10))),

    (plot(ode_figs[2]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 10), ylabel="exp(x)")),
    (plot(ode_figs[4]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 10), ytick=:none, xlabel="time")),
    (plot(ode_figs[6]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 10), ytick=:none)),

    guidefontfamily="Times",
    tickfontfamily="Times",
    titlefontsize=12,
    tickfontsize=9,
    grid=:none,
    legend=:none,
    labelfontsize=12,
    topmargin=0mm,
    leftmargin=0mm,
    bottommargin=2mm,
    layout=Plots.grid(3,3, heights=[0.19, 0.4, 0.41], widths=[0.34, 0.33, 0.33]),
    size=(800, 350)
)

plot(p_annotate, p, layout=Plots.grid(1,2, widths=[0.15, 0.85]))

# Save figure
Plots.png(plot(p, dpi=200), "$(_dir)/../results/img/exp-5.2.3-SIR_seed$(seed).png");

# ## ***2. Predator-Prey***
# load simulated data and results
open("$(_dir)/../results/exp-5.2.3-PredatorPrey_seed$(seed).dat") do io
    global results = Serialization.deserialize(io)
end;

# Evaluate negative log-likelihoods
λ0s = [50.0, 100.0, 1000.0]
model_names = ["cox", "odepois"];

nll_means = Dict(name => Dict() for name in model_names)
nll_stds = Dict(name => Dict() for name in model_names)

for λ0 in λ0s
    res = results[λ0]
    data = res["data"]
    
    M′_samples = ODEPoissonProcesses.sim_extrapolation_eventcounts(data, res[model_names[1]].mod; n_simulations=100);
    
    for name in model_names
        mod, chain = res[name].mod, res[name].chain;
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂′_samples = [X̂[mod.gm.T+1:end, :] for X̂ in X̂_samples]

        nlls = []
        for M′ in M′_samples, X̂′ in X̂′_samples
            Λ̂′ = (mod.gm.λ0/mod.gm.T)' .* exp.(X̂′)
            nll=0
            for (m′, λ′) in zip(M′, Λ̂′)
                nll += -logpdf(Poisson(λ′), m′)
            end
            push!(nlls, sum(nll))
        end

        nll_means[name][λ0] = mean(nlls)
        nll_stds[name][λ0] = std(nlls)
    end
end

# DataFrame of evaluated negative log-likelihoods
df_nlls = DataFrame("Scheme" => model_names)
for λ0 in λ0s
    df_nlls[!, "λ0=$(λ0)"] = [@sprintf("%0.1f", nll_means[name][λ0])*" ± "*@sprintf("%0.1f", nll_stds[name][λ0]) for name in model_names]
end
df_nlls

# Compare ground-truth and estimated modulations
ode_figs = []
event_figs = []
titles = [L"\lambda_0=50", L"\lambda_0=100", L"\lambda_0=1000"]
for (i, λ0) in enumerate(λ0s)
    data = results[λ0]["data"]
    push!(event_figs, ODEPoissonProcesses.eventplot(data, title=titles[i], alpha=0.75, xlims=(0, 1.5)))
    for name in model_names
        res = results[λ0]
        mod, chain = res[name].mod, res[name].chain
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂s = cat(X̂_samples..., dims=3);
        Zs = exp.(X̂s)
        fig = ODEPoissonProcesses.draw_gt_and_est_ode(data, mod, Zs, qlow=0.125, qhigh=0.875)
        push!(ode_figs, fig)
    end
end

p_annotate = plot(
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("synthesized\nevents", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP-GM\n(ours)", "Times", 12))),
    layout=Plots.grid(3,1, heights=[0.19, 0.4, 0.41]),
    size=(100, 280)
    )

p = plot(
    event_figs[1], event_figs[2], event_figs[3],

    (plot(ode_figs[1]; xtick=:none, ylims=(0, 5), ylabel="exp(x)")),
    (plot(ode_figs[3]; ticks=:none, ylims=(0, 5))),
    (plot(ode_figs[5]; ticks=:none, ylims=(0, 5))),

    (plot(ode_figs[2]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 5), ylabel="exp(x)")),
    (plot(ode_figs[4]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 5), ytick=:none, xlabel="time")),
    (plot(ode_figs[6]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 5), ytick=:none)),

    guidefontfamily="Times",
    tickfontfamily="Times",
    titlefontsize=12,
    tickfontsize=9,
    grid=:none,
    legend=:none,
    labelfontsize=12,
    topmargin=0mm,
    leftmargin=0mm,
    bottommargin=2mm,
    layout=Plots.grid(3,3, heights=[0.19, 0.4, 0.41], widths=[0.34, 0.33, 0.33]),
    size=(800, 350)
)

plot(p_annotate, p, layout=Plots.grid(1,2, widths=[0.15, 0.85]))

# Save figure
Plots.png(plot(p, dpi=200), "$(_dir)/../results/img/exp-5.2.3-PredatorPrey_seed$(seed).png");

p = plot(
    plot(event_figs[3], title=""),

    (plot(ode_figs[5]; xtick=:none, ylims=(0, 5), ylabel="exp(x)")),

    (plot(ode_figs[6]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 5), ylabel="exp(x)", xlabel="time")),

    guidefontfamily="Times",
    tickfontfamily="Times",
    tickfontsize=10,
    grid=:none,
    legend=:none,
    labelfontsize=14,
    topmargin=0mm,
    leftmargin=0mm,
    bottommargin=4mm,
    layout=Plots.grid(3,1, heights=[0.19, 0.4, 0.41]),
    xlims=(0,1.5)
)

Plots.pdf(
    plot(p_annotate, p, layout=Plots.grid(1,2, widths=[0.2, 0.8]), size=(700, 300)),
    "$(_dir)/../results/img/exp-5.2.3-extrapolation.pdf");

# ## ***3. Competition***
# load simulated data and results
open("$(_dir)/../results/exp-5.2.3-Competition_seed$(seed).dat") do io
    global results = Serialization.deserialize(io)
end;

# Evaluate negative log-likelihoods
λ0s = [50.0, 100.0, 1000.0]
model_names = ["cox", "odepois"];

nll_means = Dict(name => Dict() for name in model_names)
nll_stds = Dict(name => Dict() for name in model_names)

for λ0 in λ0s
    res = results[λ0]
    data = res["data"]
    
    M′_samples = ODEPoissonProcesses.sim_extrapolation_eventcounts(data, res[model_names[1]].mod; n_simulations=100);
    
    for name in model_names
        mod, chain = res[name].mod, res[name].chain;
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂′_samples = [X̂[mod.gm.T+1:end, :] for X̂ in X̂_samples]

        nlls = []
        for M′ in M′_samples, X̂′ in X̂′_samples
            Λ̂′ = (mod.gm.λ0/mod.gm.T)' .* exp.(X̂′)
            nll=0
            for (m′, λ′) in zip(M′, Λ̂′)
                nll += -logpdf(Poisson(λ′), m′)
            end
            push!(nlls, sum(nll))
        end

        nll_means[name][λ0] = mean(nlls)
        nll_stds[name][λ0] = std(nlls)
    end
end

# DataFrame of evaluated negative log-likelihoods
df_nlls = DataFrame("Scheme" => model_names)
for λ0 in λ0s
    df_nlls[!, "λ0=$(λ0)"] = [@sprintf("%0.1f", nll_means[name][λ0])*" ± "*@sprintf("%0.1f", nll_stds[name][λ0]) for name in model_names]
end
df_nlls

# Compare ground-truth and estimated modulations
ode_figs = []
event_figs = []
titles = [L"\lambda_0=50", L"\lambda_0=100", L"\lambda_0=1000"]
for (i, λ0) in enumerate(λ0s)
    data = results[λ0]["data"]
    push!(event_figs, ODEPoissonProcesses.eventplot(data, title=titles[i], alpha=0.75, xlims=(0, 1.5)))
    for name in model_names
        res = results[λ0]
        mod, chain = res[name].mod, res[name].chain
        X̂_samples = ODEPoissonProcesses.get_posterior_X̂_samples(mod, chain);
        X̂s = cat(X̂_samples..., dims=3);
        Zs = exp.(X̂s)
        fig = ODEPoissonProcesses.draw_gt_and_est_ode(data, mod, Zs, qlow=0.125, qhigh=0.875)
        push!(ode_figs, fig)
    end
end

p_annotate = plot(
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("synthesized\nevents", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP", "Times", 12))),
    (plot(ticks=:none, axis=false); annotate!(0.5, 0.5, Plots.text("LGCP-GM\n(ours)", "Times", 12))),
    layout=Plots.grid(3,1, heights=[0.19, 0.4, 0.41]),
    size=(100, 280)
    )

p = plot(
    event_figs[1], event_figs[2], event_figs[3],

    (plot(ode_figs[1]; xtick=:none, ylims=(0, 8), ylabel="exp(x)")),
    (plot(ode_figs[3]; ticks=:none, ylims=(0, 8))),
    (plot(ode_figs[5]; ticks=:none, ylims=(0, 8))),

    (plot(ode_figs[2]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 8), ylabel="exp(x)")),
    (plot(ode_figs[4]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 8), ytick=:none, xlabel="time")),
    (plot(ode_figs[6]; xtick=[0.0,0.5,1.0,1.5], ylims=(0, 8), ytick=:none)),

    guidefontfamily="Times",
    tickfontfamily="Times",
    titlefontsize=12,
    tickfontsize=9,
    grid=:none,
    legend=:none,
    labelfontsize=12,
    topmargin=0mm,
    leftmargin=0mm,
    bottommargin=2mm,
    layout=Plots.grid(3,3, heights=[0.19, 0.4, 0.41], widths=[0.34, 0.33, 0.33]),
    size=(800, 350)
)

plot(p_annotate, p, layout=Plots.grid(1,2, widths=[0.15, 0.85]))

# Save figure
Plots.png(plot(p, dpi=200), "$(_dir)/../results/img/exp-5.2.3-Competition_seed$(seed).png");
