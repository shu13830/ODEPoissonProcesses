#=
# ***Application 3: MovieTrends***
---

=#

# Import packages
using ODEPoissonProcesses
using JSON
using DataFrames
using Distributions
using Random
using Serialization
using Plots
using Plots.PlotMeasures
using Printf

seed=1
_dir = @__DIR__;
if ~Base.isdir("$(_dir)/../results/img")
    Base.mkdir("$(_dir)/../results/img");
end;

# ## ***1. Data preparation***
## Load data
tweets = JSON.parsefile("$(_dir)/../data/tweets.json");
keys(tweets)

# ### ***Plot data***
times_minions = float.(tweets["minions"])
times_avengers = float.(tweets["avengers_end_game"])
p = Plots.plot(
        Plots.histogram(
            times_minions, bins=90,
            title="Minions \\(2015\\)", label=:none, 
            ylabel="# of tweets/day", ylims=(0, 25),
            xlabel="day",
            xlims=(-30, 90),
            xticks=([-30, 0, 30, 60, 90], [0, 30, 60, 90, 120])
        ),
        Plots.histogram(
            times_avengers, bins=90,
            title="Avengers: End Game \\(2019\\)", label=:none,
            ylims=(0, 175),
            xlabel="day",
            xlims=(-30, 90),
            xticks=([-30, 0, 30, 60, 90], [0, 30, 60, 90, 120])
        ),
        fontfamily="Times",
        labelfontsize=12,
        tickfontsize=10,
        titlefontsize=12,
        linewidth=0.3,
        linecolor=Gray(0.4),
        fillcolor=Gray(0.4),
        size=(750,200),
        leftmargin=4mm,
        bottommargin=5mm
    )

## Save figure
Plots.pdf(p, "$(_dir)/../results/img/data_movietweets.pdf");

p

# ## ***2. Define models***
U = 21  # number of inducing points
T = 100  # number of observation points
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## ODE guided Poisson process with LGCP-based Gradient Matching
## NOTE: Only I can be observed in SIR model
i_pois_minions = InfectedPoissonProcess(
    times_minions, λ0=float(length(times_minions)), from_to=(-30., 90.), U=U, T=T, γ=γ,
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])
i_pois_avengers = InfectedPoissonProcess(
    times_avengers, λ0=float(length(times_avengers)), from_to=(-30., 90.), U=U, T=T, γ=γ,
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])

# ## ***3. Inference***
# MCMC iteration settings
n_burnin = 10000
n_thinning = 20
n_samples = 1000
n_total_iter = n_burnin + n_thinning * n_samples;

# Execute inference
Random.seed!(seed)
chain_minions = Chain(
    i_pois_minions, n_burnin=n_burnin, n_thinning=n_thinning, 
    blocks=[HMCBlock(:y, L=5), HMCBlock(:x, L=5), HMCBlock(:θ, L=5)])
chain_minions = train!(i_pois_minions, n_total_iter, chain_minions);

chain_avengers = Chain(
    i_pois_avengers, n_burnin=n_burnin, n_thinning=n_thinning, 
    blocks=[HMCBlock(:y, L=5), HMCBlock(:x, L=5), HMCBlock(:θ, L=5)])
chain_avengers = train!(i_pois_avengers, n_total_iter, chain_avengers);

# ## ***4. Store results***
results =  Dict(
    "data" => tweets,
    "odepois_minions" => (mod=i_pois_minions, chain=chain_minions),
    "odepois_avengers" => (mod=i_pois_avengers, chain=chain_avengers)
)

# ## ***5. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.3.3-MovieTrends_seed$(seed).dat", "w") do io
    Serialization.serialize(io, results)
end

# ## ***6. Check results***
# Calculate posterior parameter statistics
mod_minions, chain_minions = results["odepois_minions"].mod, results["odepois_minions"].chain
mod_avengers, chain_avengers = results["odepois_avengers"].mod, results["odepois_avengers"].chain

samples_minions = ODEPoissonProcesses.get_posterior_θ_samples(mod_minions, chain_minions);
samples_avengers = ODEPoissonProcesses.get_posterior_θ_samples(mod_avengers, chain_avengers);

θ_postmean_minions = mean(samples_minions);
θ_postmean_avengers = mean(samples_avengers);

θ_poststd_minions = std(samples_minions);
θ_poststd_avengers = std(samples_avengers);

## DataFrame of posterior parameter statistics
DataFrame(
    "Movie Production" => ["Minions", "Avengers"],
    "a" => [
        @sprintf("%0.2f", θ_postmean_minions[1])*" ± "*@sprintf("%0.2f", 2 * θ_poststd_minions[1]),
        @sprintf("%0.2f", θ_postmean_avengers[1])*" ± "*@sprintf("%0.2f", 2 * θ_poststd_avengers[1])
        ],
    "b" => [
        @sprintf("%0.2f", θ_postmean_minions[2])*" ± "*@sprintf("%0.2f", 2 * θ_poststd_minions[2]),
        @sprintf("%0.2f", θ_postmean_avengers[2])*" ± "*@sprintf("%0.2f", 2 * θ_poststd_avengers[2])
        ]
)
