@kwdef mutable struct SampleBlock
    targstate::Symbol
    algo::Symbol
    ϵ::Union{Nothing,Float64} = nothing  # step size
    L::Union{Nothing,Int} = nothing      # number of leapfrog steps
    a::Union{Nothing,Float64} = nothing  # tempering rate
end

function HMCBlock(targstate::Symbol; ϵ::Float64=0.001, L::Int=10)
    return SampleBlock(targstate=targstate, algo=:HMC, ϵ=ϵ, L=L)
end

function NUTSBlock(targstate::Symbol; ϵ::Float64=0.001)
    return SampleBlock(targstate=targstate, algo=:NUTS, ϵ=ϵ)
end

function tHMCBlock(targstate::Symbol; ϵ::Float64=0.001, L::Int=10, a::Float64=1.1)
    return SampleBlock(targstate=targstate, algo=:HMC, ϵ=ϵ, L=L, a=a)
end

function tNUTSBlock(targstate::Symbol; ϵ::Float64=0.001, a::Float64=1.1)
    return SampleBlock(targstate=targstate, algo=:NUTS, ϵ=ϵ, a=a)
end

function ESSBlock(targstate::Symbol)
    return SampleBlock(targstate=targstate, algo=:ESS)
end

function GESSBlock(targstate::Symbol)
    return SampleBlock(targstate=targstate, algo=:ESS)
end


"""
    ChainSetting

A mutable structure that holds the settings for the MCMC chain in the sampling process.

# Fields
- `n_burnin::Int`: The number of iterations to discard as the "burn-in" phase.
- `n_β_adjust_iters::Int`: The number of iterations for adjusting β parameter.
- `n_β_adjust_stepsize::Float64`: The step size for adjusting the β parameter.
- `blocks::Vector{SampleBlock}`: The blocks of sampling settings.
- `burnin::Bool = true`: A flag indicating whether to perform the "burn-in" phase.
- `n_keep_state_rec::Int = 1000`: The number of state records to keep.
- `n_thinning::Int = 10`: The thinning interval for the chain.
- `do_gm::Bool = true`: A flag indicating whether to do geometric mixing.
- `should_weight_γ = true`: A flag indicating whether to weight γ parameter.
- `should_diagonalize_A = true`: A flag indicating whether to diagonalize A matrix.

# Notes
- The structure is mutable, so the fields can be updated during the sampling process.
- Accepted symbols for `algo` are `:HMC`, `:NUTS`, `:ESS`, `:GESS`.

"""
@kwdef mutable struct ChainSetting
    n_burnin::Int
    n_β_adjust_iters::Int
    n_β_adjust_stepsize::Float64
    blocks::Vector{SampleBlock}
    burnin::Bool = true
    n_keep_state_rec::Int = 1000
    n_thinning::Int = 10
    do_gm::Bool = true
    should_weight_γ = false
    should_diagonalize_A = true
end

"""
    ChainHistory

A mutable structure that holds the history of the MCMC chain during the sampling process.

# Fields
- `n_iter::Int = 0`: The total number of iterations performed in the MCMC process.
- `accept_counter::DefaultDict{Tuple{Symbol,Symbol},Int}`: A dictionary that maps pairs of target state and algorithm to the number of accepted proposals.
- `reject_counter::DefaultDict{Tuple{Symbol,Symbol},Int}`: A dictionary that maps pairs of target state and algorithm to the number of rejected proposals.
- `accepts::DefaultDict{Tuple{Symbol,Symbol},Vector{Bool}}`: A dictionary that maps pairs of target state and algorithm to a vector of booleans representing whether each proposal was accepted.
- `state_rec::NamedTuple`: A named tuple holding the vectors of states for each of the parameters y, x, θ, σ, ϕ throughout the chain.
- `hmc_stats::DefaultDict{Tuple{Symbol,Symbol},Vector{NamedTuple}}`: A dictionary that maps pairs of target state and algorithm to a vector of named tuples representing the statistics of the Hamiltonian Monte Carlo sampling for each proposal.

# Notes
- The structure is mutable, so the fields can be updated during the sampling process.
- The dictionaries use pairs of target state and algorithm as keys to keep track of the sampling process for each combination of target state and algorithm.

"""
@kwdef mutable struct ChainHistory
    n_iter::Int = 0
    accept_counter::DefaultDict{Tuple{Symbol,Symbol},Int} = DefaultDict{Tuple{Symbol,Symbol},Int,Int}(0)  # key: (targstate, algo)
    reject_counter::DefaultDict{Tuple{Symbol,Symbol},Int} = DefaultDict{Tuple{Symbol,Symbol},Int,Int}(0)  # key: (targstate, algo)
    accepts::DefaultDict{Tuple{Symbol,Symbol},Vector{Bool}} = DefaultDict{Tuple{Symbol,Symbol},Vector{Bool}}(() -> Bool[])  # key: (targstate, algo)
    state_rec::NamedTuple = (
        y=Vector{Float64}[], x=Vector{Float64}[], θ=Vector{Float64}[], σ=Vector{Float64}[], ϕ=Vector{Float64}[]
    )
    hmc_stats::DefaultDict{Tuple{Symbol,Symbol},Vector{NamedTuple}} = DefaultDict{Tuple{Symbol,Symbol},Vector{NamedTuple}}(() -> NamedTuple[])  # key: (targstate, algo)
end


function set_chain!(
    chain::MCMCChains.Chains,
    mod::Union{ODEPoissonProcess,ODEModel};
    kwargs...
)
    if ~(:setting in keys(chain.info))
        @info "ChainSetting is not set in Chains.info.setting. Set ChainSetting."
        chain = Chain(mod)
    end
    for arg in keys(kwargs)
        if Symbol(arg) in fieldnames(ChainSetting)
            if typeof(kwargs[arg]) == fieldtype(ChainSetting,arg)
                setfield!(chain.info[:setting], Symbol(arg), kwargs[arg])
            else
                try
                    setfield!(chain.info[:setting], Symbol(arg), convert(fieldtype(ChainSetting,arg), kwargs[arg]))                   
                catch
                    error("Invalid argument type: $(arg) must be $(fieldtype(ChainSetting,arg))")
                end
            end
        elseif Symbol(arg) in fieldnames(ChainHistory)
            if typeof(kwargs[arg]) == fieldtype(ChainHistory,arg)
                setfield!(chain.info[:history], Symbol(arg), kwargs[arg])
            else
                try
                    setfield!(chain.info[:history], Symbol(arg), convert(fieldtype(ChainHistory,arg), kwargs[arg]))
                catch
                    error("Invalid argument type: $(arg) must be $(fieldtype(ChainHistory,arg))")
                end
            end
        else
            error("Invalid argument: $(arg)")
        end
    end
    return chain
end


"""
    Chain(mod::Union{ODEPoissonProcess, ODEModel}; n_burnin::Int, kwargs...) -> Chain

Construct a Markov chain Monte Carlo (MCMC) chain for the given model with specified parameters.

# Arguments
- `mod`: A model of type `ODEPoissonProcess` or `ODEModel` to be used for creating the MCMC chain.
- `n_burnin`: The number of iterations to discard as "burn-in" phase. It should be greater than or equal to 1000.

# Keyword Arguments
- `kwargs...`: Other optional parameters to be passed to the function.

# Returns
- An MCMC chain initialized with dummy records and adjusted parameters.

The MCMC chain is initialized with dummy records that depend on the type of the model. The chain is then updated with provided model.

# Throws
- Throws an assertion error if `n_burnin` is less than 1000.

# Notes
- The model can be of type `ODEPoissonProcess` or `ODEModel`.
- The `n_burnin` parameter is used to specify the number of initial iterations to be discarded (burn-in phase). It is used to ensure that the chain has reached a stable state before making inferences.

"""
function Chain(
    mod::Union{ODEPoissonProcess,ODEModel};
    n_burnin::Int,
    blocks::Vector{SampleBlock},
    kwargs...
)
    @assert n_burnin >= 1000 "n_burnin must be greater than or equal to 1000"
    # create dummy records
    if mod isa ODEPoissonProcess
        names = ["ll", "ll_m⎸x̂", "ll_x̂⎸xϕ", "ll_x⎸ϕ", "ll_x⎸ϕθγ", "ll_θ"]
    else  # mod isa ODEModel
        names = ["ll", "ll_y⎸zσϕ", "ll_z⎸ϕ", "ll_z⎸ϕθγ", "ll_θ"]
    end
    # initialize chain with dummy records
    lls = zeros(2, length(names))
    chain = MCMCChains.Chains(lls, names)
    n_β_adjust_iters = div(n_burnin, 2)
    n_β_adjust_stepsize = 1/n_β_adjust_iters
    chain = setinfo(
        chain,
        (
            setting=ChainSetting(
                n_burnin=n_burnin, 
                n_β_adjust_iters=n_β_adjust_iters, 
                n_β_adjust_stepsize=n_β_adjust_stepsize,
                blocks = blocks
            ),
            history=ChainHistory()
        )
    )
    chain = set_chain!(chain, mod; kwargs...)
    return chain
end

function train!(
    mod::Union{ODEPoissonProcess,ODEModel},
    n_iter::Int,
    chain::MCMCChains.Chains;
    show_progress::Bool=true,
    σ::Union{Float64,Nothing}=nothing,
    γ::Union{Float64,Nothing}=nothing,
    kwargs...
)

    chain = set_chain!(chain, mod; kwargs...)
    setting = chain.info[:setting]
    history = chain.info[:history]

    if history.n_iter == 0
        mod.gm.β[1] = 0.0
    end

    if ~isnothing(σ)
        @assert σ >= 0 "σ must be positive."
        mod.gm.σ[:] .= σ
    end
    if ~isnothing(γ)
        @assert γ > 0 "γ must be positive."
        mod.gm.γ[1] = γ
    end
    for (targstate, targdist) in mod.gm.targdists
        targdist.do_gm = setting.do_gm
    end
    for s in mod.gm.gpcache
        s.should_diagonalize_A = setting.should_diagonalize_A
    end
    if mod isa ODEPoissonProcess
        for (targstate, targdist) in mod.gm.targdists
            targdist.should_weight_γ = setting.should_weight_γ
        end
    end

    p = show_progress ? Progress(n_iter, showspeed=true) : nothing
    for n in 1:n_iter
        history.n_iter += 1
        if history.n_iter > setting.n_burnin
            # end of burnin iteration
            setting.burnin = false
            mod.gm.β[1] = 1.0
        else
            setting.burnin = true
            if mod.gm.β[1] < 1.0
                if history.n_iter <= setting.n_β_adjust_iters
                    mod.gm.β[1] += setting.n_β_adjust_stepsize
                end
                # if rem(history.n_iter, setting.n_β_adjust_iters) == 0
                #     mod.gm.β[1] += setting.n_β_adjust_stepsize
                # end
            end
            if mod.gm.β[1] > 1.0
                mod.gm.β[1] = 1.0
            end
        end

        gibbs!(mod, setting, history)

        new_lls = calc_lls(mod)
        chain = record_lls(chain, new_lls)
        if n % setting.n_thinning == 0
            record_states!(mod, setting, history)
        end

        show_progress && next!(p; showvalues=show_values(mod, setting, history, n, new_lls))
    end
    return chain
end

function show_values(
    mod::Union{ODEPoissonProcess,ODEModel},
    setting::ChainSetting,
    history::ChainHistory,
    n::Int,
    lls::NamedTuple
)
    # NOTE: check if gradient matching works well
    errors = get_gradient_errors(mod)
    gm1 = check_gradient_matching_validity(mod, errors, 1.0)
    gm2 = check_gradient_matching_validity(mod, errors, 2.0)
    gm3 = check_gradient_matching_validity(mod, errors, 3.0)

    iter_vals = [(:iter, n), (:total_iter, history.n_iter)]
    ll_vals = mod isa ODEPoissonProcess ?
        [
            (:ll, @sprintf("%.3f", lls.ll)),
            (:ll_m⎸x̂, @sprintf("%.3f", lls.ll_m⎸y) * " (" * @sprintf("%.3f", lls.ll_m⎸y / sum(mod.gm.M .!= nothing)) * "/window)"),
            (:ll_x̂⎸xϕ, @sprintf("%.3f", lls.ll_y⎸xσmϕ) * " (" * @sprintf("%.3f", lls.ll_y⎸xσmϕ / length(mod.gm.Y)) * "/variable)"),
            (:ll_x⎸ϕ, @sprintf("%.3f", lls.ll_x⎸mϕ) * " (" * @sprintf("%.3f", lls.ll_x⎸mϕ / mod.data.C) * "/category)"),
            (:ll_grad, @sprintf("%.3f", lls.ll_x⎸mϕθγ)),
            (:lp_θ, @sprintf("%.3f", lls.ll_θ) * "\n")
        ] :
        [
            (:ll, @sprintf("%.3f", lls.ll)),
            (:ll_y⎸zσϕ, @sprintf("%.3f", lls.ll_y⎸xσmϕ) * " (" * @sprintf("%.3f", lls.ll_y⎸xσmϕ / length(mod.gm.Y)) * "/variable)"),
            (:ll_z⎸ϕ, @sprintf("%.3f", lls.ll_x⎸mϕ) * " (" * @sprintf("%.3f", lls.ll_x⎸mϕ / mod.data.C) * "/category)"),
            (:ll_grad, @sprintf("%.3f", lls.ll_x⎸mϕθγ)),
            (:lp_θ, @sprintf("%.3f", lls.ll_θ) * "\n")
        ]

    block_showvals = []
    for (i, blk) in enumerate(setting.blocks)
        t = blk.targstate
        a = blk.algo
        append!(block_showvals, "\n")
        append!(block_showvals, repeat(" ", 36))
        append!(block_showvals, string(t) * " - ")
        append!(block_showvals, string(a))
    end
    block_showvals = join(block_showvals)

    accept_showvals = join(
        "\n" * repeat(" ", 36) * 
        "$(k=> length(v) > 100 ? sum(v[end-99:end]) :
            length(v) > 10 ? sum(v) / length(v) * 100 :
            "?")" * "%" 
        for (k, v) in history.accepts
    )
    L_showvals = join(
        isnothing(blk.L) ? "" :
            "\n" * repeat(" ", 36) * 
            "$(blk.targstate)-$(blk.algo): " * 
            (blk.algo in [:NUTS, :tNUTS] ?
                (length(history.hmc_stats[(blk.targstate,blk.algo)]) > 0 ?
                    "$(history.hmc_stats[(blk.targstate,blk.algo)][end].n_steps)" :
                    "$(blk.L)") * " (being adjusted)" :
                "$(blk.L)" * (setting.burnin ? " (fixed)" : " (being adjusted, but L*ϵ is constant)"))
            for blk in setting.blocks
    )
    ϵ_showvals = join(
        isnothing(blk.ϵ) ? "" :
            "\n" * repeat(" ", 36) * "$(blk.targstate)-$(blk.algo): " * @sprintf("%.5f", blk.ϵ) *
            (setting.burnin ? " (being adjusted)" : " (being adjusted, but L*ϵ is constant)")
            for blk in setting.blocks
    )

    common_vals = [
        ("sample blocks", block_showvals),
        ("accept % (in recent 100 trials)", accept_showvals),
        ("L", L_showvals),
        ("ϵ", ϵ_showvals),
        ("burnin", setting.burnin),
        ("gradient matching", setting.do_gm),
        ("γ weighting", setting.should_weight_γ),
        ("A diagonalizing", setting.should_diagonalize_A),
        ("β (inverse temperature)",
            @sprintf("%.3f", mod.gm.β[1]) * (mod.gm.β[1] < 1 ? " (being adjusted)" : " (fixed)")),
        ("% grad errors in ±1 std", @sprintf("%.1f", gm1 * 100)),
        ("% grad errors in ±2 std", @sprintf("%.1f", gm2 * 100)),
        ("% grad errors in ±3 std", @sprintf("%.1f", gm3 * 100)),
        ("σ", mod.gm.σ),
        ("gp mean", [s.m for s in mod.gm.gpcache]),
        ("kernel", [
            s.ϕ.base_kernel isa RBFKernel ? "RBF" :
            s.ϕ.base_kernel isa Matern52Kernel ? "Matern52" :
            s.ϕ.base_kernel
            for s in mod.gm.gpcache]),
        ("ϕ_1", [s.ϕ.α for s in mod.gm.gpcache]),
        ("ϕ_2", [s.ϕ.𝓁 for s in mod.gm.gpcache]),
        ("ϕ_3", [s.ϕ.δ for s in mod.gm.gpcache]),
        ("γ", @sprintf("%.3f", mod.gm.γ[1]))
    ]
    return [iter_vals; ll_vals; common_vals]
end


function gibbs!(
    mod::Union{ODEPoissonProcess,ODEModel}, 
    setting::ChainSetting, 
    history::ChainHistory
)
    block = rand(setting.blocks)
    sample_vars!(mod, setting, history, block)
end

function check_gradient_matching_validity(
    mod::Union{ODEPoissonProcess,ODEModel}, 
    errors, 
    n_std::T
) where {T<:Real}
    @unpack ode, γ, X, θ, gpcache = mod.gm
    res = []
    for (c, err) in enumerate(errors)
        @unpack A = gpcache[c]
        err_std = sqrt.(diag(A) .+ γ[1]^2) * n_std
        push!(res, abs.(err) .< err_std)
    end
    res = vcat(res...)
    return sum(res) / length(res)
end

function get_gradient_errors(
    mod::Union{ODEPoissonProcess,ODEModel}
)
    @unpack ode, γ, X, θ, gpcache = mod.gm
    U, C = size(X)
    θvec = get_state_vector(mod, [:θ])
    Ds = [s.D for s in gpcache]
    ms = [s.m for s in gpcache]
    if mod isa ODEPoissonProcess
        errors = get_gradient_errors(U, C, Ds, ms, ode, θ, X, θvec, LGCPGMScheme())
    else
        errors = get_gradient_errors(U, C, Ds, ms, ode, θ, X, θvec, GPGMScheme())
    end
    errors
end

"""
Adjustment of step size ε:
ε is reduced when rejected for 10 consecutive times and increased when accepted for 10 consecutive times.
This process is only carried out during burn-in to avoid disrupting the Markov chain.
"""
function adjust_ϵ_heuristically!(setting::ChainSetting, history::ChainHistory, block::SampleBlock)
    @unpack targstate, algo = block
    if setting.burnin
        if algo in [:HMC, :NUTS, :tHMC, :tNUTS]
            if history.accept_counter[(targstate,algo)] > 10
                block.ϵ *= 1.1
            end
            if history.reject_counter[(targstate,algo)] > 3
                block.ϵ /= 1.1
            end
        end
    else # NOTE: after burnin, keep L * ϵ constant not to break the Markov chain
        if algo in [:HMC, :tHMC]
            if history.accept_counter[(targstate,algo)] > 6
                if block.L > 5  # NOTE: assume minimum L is 5
                    # subtract 1 from L
                    block.L -= 1
                    # make ϵ larger to keep L * ϵ constant
                    block.ϵ *= (block.L+1) / block.L
                end
            end
            if history.reject_counter[(targstate,algo)] > 4
                if block.L < 30  # NOTE: assume maximum L is 30
                    # add 1 to L
                    block.L += 1
                    # make ϵ smaller to keep L * ϵ constant
                    block.ϵ *= (block.L-1) / block.L
                end
            end
        end
    end
end


"""
Adjustment of step size ε 2:
If ε is too large, it may lead to the occurrence of internal variables as Inf or -Inf during sampling, 
or result in variables being forced to zero, causing the sampler to throw errors. In such cases, 
an exception handling is implemented by halving ε.
"""
function adjust_ϵ_to_deal_with_overflow!(block::SampleBlock)
    @warn "Stepsize for $(block.targstate) is too wide and the integrator propose non-finite value internally." *
        "To deal with this, stepsize was set to the half value."
    block.ϵ /= 2
end


"""
    sample_vars!(
        mod::Union{ODEPoissonProcess,ODEModel}, 
        setting::ChainSetting, 
        history::ChainHistory, 
        targstate::Symbol,
        algo::Symbol
    )

Perform sampling of the variables in the given model using a specified MCMC algorithm.

# Arguments
- `mod`: A model of type `ODEPoissonProcess` or `ODEModel` for which sampling is performed.
- `setting`: A `ChainSetting` instance that holds the settings for the MCMC chain.
- `history`: A `ChainHistory` instance that keeps track of the history of the chain.
- `targstate`: A symbol representing the target state for sampling.
- `algo`: A symbol representing the MCMC algorithm to be used for sampling. 

# Behavior
The function will perform sampling according to the specified algorithm. If the sampling is successful, it will update the model's state. If the sampling fails, it will adjust the sampling parameters and try again. It records sampling history, including the number of accepted and rejected proposals.

# Throws
- Throws an assertion error if `targstate` is not among the accepted symbols for the given model type.
- Throws an assertion error if `algo` is not among the accepted sampling algorithms.

# Notes
- Accepted symbols for `targstate` depend on the model type.
- Accepted symbols for `algo` are `:HMC`, `:NUTS`, `:ESS`, `:GESS`.

"""
function sample_vars!(
    mod::Union{ODEPoissonProcess,ODEModel}, 
    setting::ChainSetting, 
    history::ChainHistory, 
    block::SampleBlock
)
    targstate = block.targstate
    algo = block.algo  # use default algorithm

    if mod isa ODEPoissonProcess
        @assert targstate in [:y, :x, :θ, :xθ, :yx, :yxθ, :ϕ, :yxθϕ, :yxϕ, :xθϕ]
    else
        @assert targstate in [:y, :x, :θ, :σ, :xθ, :yx, :yxθ, :ϕ, :yxθσϕ, :yxθϕ, :yxϕ, :xθϕ]
    end
    @assert algo in [:HMC, :NUTS, :tHMC, :tNUTS, :ESS, :GESS]
    targets = Symbol[Symbol(s) for s in string(targstate)]

    v = get_state_vector(mod, targets)

    if algo in [:HMC, :NUTS, :tHMC, :tNUTS]
        adjust_ϵ_heuristically!(setting, history, block)
    end

    try
        targ = mod.gm.targdists[targstate]

        if algo in [:HMC, :NUTS, :tHMC, :tNUTS]
            # metric
            metric = DiagEuclideanMetric(targ.D)            
            ℓπ = v -> targ.lp(targ, v)
            ∂ℓπ∂θ = v -> (ℓπ(v), targ.g(targ, v))
            hamiltonian = AdvancedHMC.Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
            integrator = algo in [:HMC, :NUTS] ? Leapfrog(block.ϵ) :
                algo in [:tHMC, :tNUTS] ? TemperedLeapfrog(block.ϵ, block.a) :
                nothing
            proposal = algo in [:HMC, :tHMC] ? StaticTrajectory(integrator, block.L) :
                    algo in [:NUTS, :tNUTS] ? NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator) :
                    nothing
            samples, stats = AdvancedHMC.sample(hamiltonian, proposal, v, 1, verbose=false)
            push!(history.hmc_stats[(targstate, algo)], stats[end])
            global v_new = samples[end]
        elseif algo in [:ESS, :GESS]
            # gaussian approximation to the target density
            μ = zeros(targ.D)
            Σ = Matrix(LinearAlgebra.I(targ.D))
            if algo==:ESS
                pseudo_prior = MvNormal(μ, Σ)
                # log residual error of above approximation to the target density
                pseudo_loglikelihood = v -> targ.lp(targ, v) - logpdf(pseudo_prior, v)
            elseif algo==:GESS
                # see [Nishihara+2014] Parallel MCMC with generalized elliptical slice sampling 
                # conditionally sample scale parameter
                dof = 10.0
                α = (targ.D + dof) / 2
                β = 1 / 2 * (dof + (v .- μ)' * (Σ \ (v .- μ)))
                scale = rand(InverseGamma(α, β))
                # gaussian approximation to the target density
                pseudo_prior = MvNormal(μ, scale * Σ)
                # log residual error of above approximation to the target density
                pseudo_loglikelihood = v -> targ.lp(targ, v) - logpdf(MvTDist(dof, μ, Σ), v)
            end
            ess = ESSModel(pseudo_prior, pseudo_loglikelihood)
            global v_new = EllipticalSliceSampling.sample(ess, ESS(), 1, init_params=v, discard_initial=true)[1]
        end
        global acc = v == v_new ? false : true
    catch e
        if typeof(e) <: AssertionError || typeof(e) <: ArgumentError
            @warn "sampling is failed:"
            @show e
            if setting.burnin
                adjust_ϵ_to_deal_with_overflow!(block)
            end
        elseif typeof(e) <: DomainError
            @warn "sampling is failed due to the occurrence of Inf or -Inf."
            @show e
        else
            rethrow(e)
        end
        global acc = false
    end

    if acc
        reflect_states!(v_new, mod, targets)
        push!(history.accepts[(targstate,algo)], true)
        history.accept_counter[(targstate,algo)] += 1
        history.reject_counter[(targstate,algo)] = 0
    else
        push!(history.accepts[(targstate,algo)], false)
        history.accept_counter[(targstate,algo)] = 0
        history.reject_counter[(targstate,algo)] += 1
    end
end
