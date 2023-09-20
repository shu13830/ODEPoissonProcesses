#=
# ***ODE Parameter Estimation with Competition Model***
---

=#

using ODEPoissonProcesses
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
T_coarse = 20  # number of observation points for coarsly discretized GPGM
base_kernel = :RBF
ϕ = [5.0, 0.15, 0.1]
γ = 0.1

## ODE guided Poisson process with LGCP-based Gradient Matching
odepois = CompetitionPoissonProcess(
    times, classes; λ0=λ0, U=U, T=T, γ=γ, 
    base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])
## Conventional ODE Model with GP-based Gradient Matching (coarsly discretized observation points)
odemod_coarse = CompetitionModel(
    times, classes; λ0=λ0, U=U, T=T_coarse, γ=γ, 
    base_kernel=base_kernel)  # NOTE: kernel parameters and σ are optimized
## Conventional ODE Model with GP-based Gradient Matching (finely discretized observation points)
odemod_fine = CompetitionModel(
    times, classes; λ0=λ0, U=U, T=T, γ=γ, 
    base_kernel=base_kernel);  # NOTE: kernel parameters and σ are optimized

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
chain_odepois = train!(odepois, n_total_iter, chain_odepois)

chain_coarse = Chain(
    odemod_coarse, n_burnin=n_burnin, n_thinning=n_thinning, 
    blocks=[HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
chain_coarse = train!(odemod_coarse, n_total_iter, chain_coarse)

chain_fine = Chain(
    odemod_fine, n_burnin=n_burnin, n_thinning=n_thinning, 
    blocks=[HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
chain_fine = train!(odemod_fine, n_total_iter, chain_fine);

# ## ***4. Store results***
results =  Dict(
    λ0 => Dict(
        "data" => dat,
        "odepois" => (mod=odepois, chain=chain_odepois),
        "odemod_coarse" => (mod=odemod_coarse, chain=chain_coarse),
        "odemod_fine" => (mod=odemod_fine, chain=chain_fine)
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
        base_kernel=base_kernel, ascale=ϕ[1], lscale=ϕ[2], δ=ϕ[3])
    odemod_coarse = CompetitionModel(
        times, classes; λ0=λ0, U=U, T=T_coarse, γ=γ, 
        base_kernel=base_kernel)
    odemod_fine = CompetitionModel(
        times, classes; λ0=λ0, U=U, T=T, γ=γ, 
        base_kernel=base_kernel)

    ## inference
    chain_odepois = Chain(
        odepois, n_burnin=n_burnin, n_thinning=n_thinning, 
        blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
    chain_odepois = train!(odepois, n_total_iter, chain_odepois)
    
    chain_coarse = Chain(
        odemod_coarse, n_burnin=n_burnin, n_thinning=n_thinning, 
        blocks=[HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
    chain_coarse = train!(odemod_coarse, n_total_iter, chain_coarse)
    
    chain_fine = Chain(
        odemod_fine, n_burnin=n_burnin, n_thinning=n_thinning, 
        blocks=[HMCBlock(:x, L=10), HMCBlock(:θ, L=10)])
    chain_fine = train!(odemod_fine, n_total_iter, chain_fine)

    ## Store results
    results[λ0] = Dict(
        "data" => dat,
        "odepois" => (mod=odepois, chain=chain_odepois),
        "odemod_coarse" => (mod=odemod_coarse, chain=chain_coarse),
        "odemod_fine" => (mod=odemod_fine, chain=chain_fine)
    );
end;

# ## ***6. Save results***
_dir = @__DIR__
if ~Base.isdir("$(_dir)/../results")
    Base.mkdir("$(_dir)/../results")
end
open("$(_dir)/../results/exp-5.2.2-Competition_seed$(seed).dat", "w") do io
    Serialization.serialize(io, results)
end
