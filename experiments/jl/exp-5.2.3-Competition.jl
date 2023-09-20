#=
# ***Extrapolation with Competition Model***
---

=#

using ODEPoissonProcesses
using StatsBase
using Plots
using Random
using Serialization

# ## ***1. Simulate event data***
seed = 1
λ0 = 50.
dat = sim_Competition_event_data(seed=seed, λ0=λ0);

## plot simulated data
plot(size=(800,200), legend=:outerleft)
scatter!(dat.times[1], zeros(length(dat.times[1])) .+ 2, m=:vline, label="A")
scatter!(dat.times[2], zeros(length(dat.times[2])) .+ 1, m=:vline, label="B")
scatter!(dat.times[3], zeros(length(dat.times[3])), m=:vline, label="C")
ylims!(-1,3)

# ## ***2. Define models***
times = Dict(1=>dat.times[1], 2=>dat.times[2], 3=>dat.times[3])
classes = Dict(1=>"A", 2=>"B", 3=>"C")
U = 21  # number of inducing points 
T = 100  # number of observation points
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## NOTE: set `ex_time` for extrapolation
## ODE guided Poisson process with LGCP-based Gradient Matching
odepois = CompetitionPoissonProcess(
    times, classes; λ0=λ0, U=U, T=T, γ=γ, 
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3], ex_time=0.5)

## kernel parameter grid search for LGCP
a_options = [0.5, 1.0, 1.5, 2.0]
l_options = [0.1, 0.15, 0.2, 0.25]
δ_options = [0.1]

ϕ_patterns = []
lls = []
for a in a_options, l in l_options, δ in δ_options
    cox = CompetitionPoissonProcess(
        times, classes; λ0=λ0, U=U, T=T, γ=γ,
        base_kernel=base_kernel, ascale=a, lscale=l, δ=δ)
    chain_cox = Chain(
        cox, n_burnin=1500, n_thinning=50,
        blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10)])
    chain_cox = train!(cox, 2000, chain_cox, do_gm=false)
    push!(ϕ_patterns, (a, l, δ))
    push!(lls, mean(chain_cox[:ll][1501:2000]))
end

## Find the best kernel parameter setting
ϕ_best_lgcp = ϕ_patterns[maximum(lls) .== lls][1]

## Cox process with LGCP (do not use Gradient Matching)
cox = CompetitionPoissonProcess(
    times, classes; λ0=λ0, U=U, T=T, γ=γ,
    base_kernel=base_kernel, ascale=ϕ_best_lgcp[1], lscale=ϕ_best_lgcp[2], δ=ϕ_best_lgcp[3], ex_time=0.5)

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
    blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
chain_odepois = train!(odepois, n_total_iter, chain_odepois);

## NOTE: do not use gradient matching scheme
chain_cox = Chain(
    cox, n_burnin=n_burnin, n_thinning=n_thinning,
    blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10)])
chain_cox = train!(cox, n_total_iter, chain_cox, do_gm=false);

# ## ***4. Store results***
results =  Dict(
    λ0 => Dict(
        "data" => dat,
        "odepois" => (mod=odepois, chain=chain_odepois),
        "cox" => (mod=cox, chain=chain_cox)
    )
);

# ## ***5. Experiments with different $\lambda_0$***
Random.seed!(seed)
for λ0 in [100., 1000.]
    ## simulate data
    dat = sim_Competition_event_data(seed=seed, λ0=λ0);

    ## define models
    times = Dict(1=>dat.times[1], 2=>dat.times[2], 3=>dat.times[3])
    classes = Dict(1=>"A", 2=>"B", 3=>"C")

    odepois = CompetitionPoissonProcess(
        times, classes; λ0=λ0, U=U, T=T, γ=γ, 
        base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3], ex_time=0.5)
    
    ϕ_patterns = []
    lls = []
    for a in a_options, l in l_options, δ in δ_options
        cox = CompetitionPoissonProcess(
            times, classes; λ0=λ0, U=U, T=T, γ=γ,
            base_kernel=base_kernel, ascale=a, lscale=l, δ=δ)
        chain_cox = Chain(
            cox, n_burnin=1500, n_thinning=50,
            blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10)])
        chain_cox = train!(cox, 2000, chain_cox, do_gm=false)
        push!(ϕ_patterns, (a, l, δ))
        push!(lls, mean(chain_cox[:ll][1501:2000]))
    end
    ϕ_best_lgcp = ϕ_patterns[maximum(lls) .== lls][1]        
    cox = CompetitionPoissonProcess(
        times, classes; λ0=λ0, U=U, T=T, γ=γ,
        base_kernel=base_kernel, ascale=ϕ_best_lgcp[1], lscale=ϕ_best_lgcp[2], δ=ϕ_best_lgcp[3], ex_time=0.5)

    ## inference
    chain_odepois = Chain(
        odepois, n_burnin=n_burnin, n_thinning=n_thinning,
        blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
    chain_odepois = train!(odepois, n_total_iter, chain_odepois)

    chain_cox = Chain(
        cox, n_burnin=n_burnin, n_thinning=n_thinning,
        blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10)])
    chain_cox = train!(cox, n_total_iter, chain_cox, do_gm=false)

    ## Store results
    results[λ0] = Dict(
        "data" => dat,
        "odepois" => (mod=odepois, chain=chain_odepois),
        "cox" => (mod=cox, chain=chain_cox)
    );
end;

# ## ***6. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.2.3-Competition_seed$(seed).dat", "w") do io
    Serialization.serialize(io, results)
end
