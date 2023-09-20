#=
# ***Application 1: Patent Application Event Analysis***
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
using Plots
using Plots.PlotMeasures
using PythonPlot

seed = 1
_dir = @__DIR__;
if ~Base.isdir("$(_dir)/../results/img")
    Base.mkdir("$(_dir)/../results/img");
end;

# ## ***1. Data preparation***
# Load data
df = CSV.read("$(_dir)/../data/patents.csv", header=1, DataFrame);
first(df, 3)

# Preprocess date info.
## convert Date to float
df.date = ODEPoissonProcesses.float_times(string.(df.date), "YYYYMMDD")
first(df, 3)

# Prepear data as inpur for ODEPoissonProcesses
companies = sort(collect(Set(df.co)))
classes = Dict(i => String(c) for (i, c) in enumerate(companies))
times = Dict(i => Float64[t for (t, c) in zip(df.date, df.co) if c == co] for (i, co) in enumerate(companies));

# ### ***Plot data***
p1 = Plots.plot(
    yticks=([1,2,3,4,5], companies),
    yflip=true,
    ylims=(0.5,5.5),
    xlabel="Patent application date",
)
for (i, co) in enumerate(companies)
    n_pats = length(times[i])
    scatter!(times[i], fill(i, n_pats), color=:black, m=:+)
end

p = Plots.plot(
    p1,
    size=(800,350),
    bottommargin=12mm,
    tickfontsize=16,
    labelfontsize=18,
    fontfamily="Times",
    legend=:none,
)

# Save figure
Plots.pdf(p, "$(_dir)/../results/img/data_patents.pdf")

# ## ***2. Define models***
U = 21  # number of inducing points 
T = 100  # number of observation points
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## ODE guided Poisson process with LGCP-based Gradient Matching
odepois = CompetitionPoissonProcess(
    times, classes; U=U, T=T, γ=γ, from_to=(2011.0, maximum(df.date)), 
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
);

# ## ***5. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.3.1-Patnets_seed$(seed).dat", "w") do io
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
ticks = [0, 1, 2, 3, 4]
PythonPlot.xticks(ticks, companies, font="Times", fontsize=20, rotation=45)
PythonPlot.yticks(ticks, companies, font="Times", fontsize=20)
fig.savefig("$(_dir)/../results/img/A_postmean_patents.pdf", dpi=300)
fig.show();