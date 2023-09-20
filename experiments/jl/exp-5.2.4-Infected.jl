#=
# ***Modeling with Non-available components with SIR Model***
---

=#

using ODEPoissonProcesses
using Plots
using Random
using Serialization

# ## ***1. Simulate event data***
seed = 1
λ0 = 1000.
dat = sim_SIR_event_data(seed=seed, λ0=λ0);

## plot simulated data
plot(size=(800,200), legend=:outerleft)
scatter!(dat.times[1], zeros(length(dat.times[1])) .+ 2, m=:vline, label="S")
scatter!(dat.times[2], zeros(length(dat.times[2])) .+ 1, m=:vline, label="I")
scatter!(dat.times[3], zeros(length(dat.times[3])), m=:vline, label="R")
ylims!(-1,3)

# ## ***2. Define models***
times = Dict(1=>dat.times[1], 2=>dat.times[2], 3=>dat.times[3])
classes = Dict(1=>"S", 2=>"I", 3=>"R")
U = 21  # number of inducing points 
T = 100  # number of observation points
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## ODE guided Poisson process with LGCP-based Gradient Matching
## NOTE: S, I and R can be observed
sir_pois = SIRPoissonProcess(
    times, classes; λ0=λ0, U=U, T=T, γ=γ, 
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])
## NOTE: Only I can be observed
i_pois = InfectedPoissonProcess(
    times[2], λ0=λ0, from_to=(0., 1.), U=U, T=T, γ=γ, 
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3]);

# ## ***3. Inference***
# MCMC iteration settings
n_burnin = 10000
n_thinning = 20
n_samples = 1000
n_total_iter = n_burnin + n_thinning * n_samples;

# Execute inference
Random.seed!(seed)
chain_sir = Chain(
    sir_pois, n_burnin=n_burnin, n_thinning=n_thinning,
    blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
chain_sir = train!(sir_pois, n_total_iter, chain_sir)

Random.seed!(seed)
chain_i = Chain(
    i_pois, n_burnin=n_burnin, n_thinning=n_thinning,
    blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:xθ, L=10), HMCBlock(:θ, L=10)])
chain_i = train!(i_pois, n_total_iter, chain_i);

# ## ***4. Store results***
results =  Dict(
    "data" => dat,
    "sir_pois" => (mod=sir_pois, chain=chain_sir),
    "i_pois" => (mod=i_pois, chain=chain_i)
);

# ## ***5. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.2.4-Infected_seed$(seed).dat", "w") do io
    Serialization.serialize(io, results)
end
