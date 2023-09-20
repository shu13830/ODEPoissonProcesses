#=
# ***Application 2: competition of JavaScript web frameworks***
---

=#

# Install packages
import Pkg
Pkg.add("PythonPlot")

# Import packages
using ODEPoissonProcesses
using CSV
using DataFrames
using Distributions
using Random
using Serialization
using PythonPlot

seed=1
_dir = @__DIR__;
if ~Base.isdir("$(_dir)/../results/img")
    Base.mkdir("$(_dir)/../results/img");
end;

# ## ***1. Data preparation***
## Load data
df = CSV.read("$(_dir)/../data/jsframeworks.csv", header=3, DataFrame);
rename!(df, ["Month", "Angular", "jQuery", "React", "Vue.js"]);

## preprocess values
## truncate values smaller than 1 to 0
@show last(df, 5);
for i in 2:size(df)[2]
    parsed = df[!, "$(names(df)[i])"] .|> x -> tryparse(Int, x)
    truncated = parsed .|> x -> x==nothing ? 0 : x
    df[!, "$(names(df)[i])"] = truncated
end
@show last(df, 5);

## preprocess date info.
## convert Date to float
dates = df[!, 1]
decimaldates = ODEPoissonProcesses.decimaldate.(dates)
df[!, 1] = decimaldates
@show last(df, 5);

# ### ***Plot data***
time_points = df[!, 1]
framework_names = ["jQuery", "Angular", "React", "Vue.js"]
cm = PythonPlot.pyplot.colormaps["tab10"]
values = Dict(i => df[!, name] for (i, name) in enumerate(framework_names))

ls = ["dotted", "solid", "dashed", "dashdot"] #line_style
_colors = [cm.colors[0], cm.colors[3], cm.colors[9], cm.colors[2]]
fig, ax = PythonPlot.pyplot.subplots(figsize=(8,3.5))
[ax.plot(time_points, values[i], lw=2, ls=ls, label=name, c=c) for (i, (ls, c, name)) in enumerate(zip(ls,_colors,framework_names))]
ax.legend(loc="upper left", handlelength=2, prop=Dict("family"=>"Times", "size"=>20), frameon=false)
ax.set_xticks([2005, 2010, 2015, 2020])
ax.set_xticklabels([2005, 2010, 2015, 2020], font="Times", fontsize=18)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100], font="Times", fontsize=18)
ax.set_xlabel("Year", fontsize=24, font="Times")
ax.set_ylabel("Interest", fontsize=24, font="Times")
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)
PythonPlot.tight_layout()

## Save figure
PythonPlot.savefig("$(_dir)/../results/img/data_jsframeworks.pdf", dpi=300)
PythonPlot.pyplot.show();

# ## ***2. Define models***
times = Dict(i => vcat([fill(t, count) for (count, t) in zip(values[i], time_points)]...) for i in 1:length(values))
classes = Dict(i=>name for (i, name) in enumerate(framework_names));
U = 21  # number of inducing points 
T = 100  # number of observation points
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## ODE guided Poisson process with LGCP-based Gradient Matching
odepois = CompetitionPoissonProcess(
    times, classes; U=U, T=T, γ=γ, 
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])

# ## ***3. Inference***
# MCMC iteration settings
n_burnin = 10000
n_thinning = 20
n_samples = 1000
n_total_iter = n_burnin + n_thinning * n_samples;

# Execute inference
Random.seed!(seed)
chain_odepois = Chain(
    odepois, n_burnin=n_burnin, n_thinning=n_thinning, 
    blocks=[HMCBlock(:y, L=5), HMCBlock(:x, L=5), HMCBlock(:θ, L=5)])
chain_odepois = train!(odepois, n_total_iter, chain_odepois);

# ## ***4. Store results***
results =  Dict(
    "data" => df,
    "odepois" => (mod=odepois, chain=chain_odepois)
)

# ## ***5. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.3.2-WebFrameworks_seed$(seed).dat", "w") do io
    Serialization.serialize(io, results)
end

# ## ***6. Plot results***
# Calculate posterior mean of A (competitive coefficient matrix)
mod, chain = results["odepois"].mod, results["odepois"].chain
C = mod.data.C
samples = ODEPoissonProcesses.get_posterior_θ_samples(mod, chain);
A_posterior_mean = ODEPoissonProcesses.competitive_coef_matrix(mean(samples)[2*C+1:end], C);
A_posterior_mean = float.(A_posterior_mean);

# Draw a hetmap
fig = PythonPlot.figure(layout="tight", figsize=(4.5,4.5))
PythonPlot.imshow(A_posterior_mean, cmap="gray_r")
ticks = [0, 1, 2, 3]
PythonPlot.xticks(ticks, framework_names, font="Times", fontsize=24, rotation=30)
PythonPlot.yticks(ticks, framework_names, font="Times", fontsize=24)
fig.savefig("$(_dir)/../results/img/A_postmean_jsframeworks.pdf", dpi=300)
fig.show();