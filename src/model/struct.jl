struct ModelInput
    times::Dict{Int,Vector{Float64}}             # normalized time points
    times_orig::Dict{Int,Vector{Float64}}        # original time points
    classes::Dict{Int,String}                    # classes
    from::Float64
    to::Float64
    C::Int                                       # number of classes

    function ModelInput(
        times::Dict{Int,Vector{Float64}},
        from_to::Union{Nothing,Tuple{Float64,Float64}},
        classes::Dict{Int,String}
    )
        # check input length
        @assert length(times) == (length(classes)) "times and classes must have same length."

        times_concat = vcat(values(times)...)
        t_min, t_max = minimum(times_concat), maximum(times_concat)
        # normalize times
        if isnothing(from_to)
            from = t_min
            to = t_max
        else
            from, to = from_to
            @assert from <= t_min && t_max <= to
        end
        times_normalized = Dict(i => normalize_times(sort(_times), from, to) for (i, _times) in times)
        # sort inputs
        # get class information
        C = length(times_normalized)

        new(times_normalized, times, classes, from, to, C)
    end
end


mutable struct LGCPGM
    M::Matrix{Union{Nothing,Int}}            # number of observations in each time window
    Y::Matrix{Float64}                       # State at observation time points
    X::Matrix{Float64}                       # State at inducing time points
    θ::ODEParamSetting                       # ODE parameters
    σ::Vector{Float64}                       # std of Gaussian observation noise (basically the value is set to zero. Positive values are set only in burn-in for efficient state search)
    γ::Vector{Float64}                       # std of Gaussian noise for ODE gradient probabilistic model
    β::Vector{Float64}                       # inverse temperature for relaxing gradient matching constraint
    ode::Function                            # Ordinary Differential Equation. NOTE: needs to be implemented for each ODE model
    observed_ids::Vector{Int}                # indices of observed data points
    λ0::Vector{Float64}                      # coefficients of intensity function
    Δt::Float64                              # interval of observation time points
    T::Int                                   # number of interpolation observation time points to evaluate Poisson likelihood
    t::Vector{Float64}                       # interpolation observation time points to evaluate Poisson likelihood 0 < w < 1
    T⁺::Int                                  # number of observation time points to evaluate Poisson likelihood
    t⁺::Vector{Float64}                      # observation time points to evaluate Poisson likelihood 0 < w⁺
    Δu::Float64                              # interval of inducing points
    U::Int                                   # number of interpolation inducing time points to evaluate gradient
    u::Vector{Float64}                       # interpolation inducing time points to evaluate gradient 0 < u < 1
    U⁺::Int                                  # number of inducing time points to evaluate gradient
    u⁺::Vector{Float64}                      # inducing time points to evaluate gradient 0 < u⁺
    targdists::Dict{Symbol,LGCPGMTarget}    # target distributions
    gpcache::Vector{SparseGPCache{Float64}}  # Sparse GP cache

    function LGCPGM(
        data::ModelInput,
        ode::Function,
        observed_ids::Vector{Int},
        init_ode_param_setting_fn::Function,
        λ0::Union{Nothing,T1,Vector{T1}},
        is_λ0_unified::Bool,
        T::Int,
        U::Int,
        ex_time::T1,
        # followings are input for state initialization
        γ::T1,
        m::T1,
        ode_param_priors::Vector{ScaledLogitNormal},
        base_kernel::Symbol,
        ascale::T1,
        lscale::T1,
        δ::T1,
        should_diagonalize_A::Bool,
    ) where {T1<:Real}

        @assert T > U
        @unpack C, times = data
        λ0 = initialize_λ0(λ0, is_λ0_unified, C, times)
        Δt, t, t⁺, Δu, u, u⁺ = initialize_time_points(T, U, ex_time)
        T⁺ = length(t⁺)
        U⁺ = length(u⁺)
        M = calc_ocurrences(Δt, t⁺, T, T⁺, C, times, observed_ids)
        gpcache = SparseGPCache{T1}[
            SparseGPCache(
                m, base_kernel, ascale, lscale, δ, U⁺, u⁺, t⁺; 
                should_diagonalize_A=should_diagonalize_A)
            for c in 1:C]
        Y = zeros(T⁺, C)
        X = zeros(U⁺, C)
        θ = init_ode_param_setting_fn(C, ode_param_priors...)
        θ_normal_priors = get_θ_normal_priors(θ)
        σ = zeros(C)
        γ = [γ]
        β = [1.0]
        targdists = Dict{Symbol,LGCPGMTarget}()
        for targstate in [:yxθ, :yx, :xθ, :y, :x, :θ, :yxθϕ, :yxϕ, :xθϕ, :ϕ]
            targdists[targstate] = LGCPGMTarget(
                targstate, T, T⁺, U⁺, C, M, Y, X, θ, gpcache, θ_normal_priors, σ, γ, β, ode, λ0
            )
        end

        return new(
            M, Y, X, θ, σ, γ, β, ode, observed_ids, λ0, Δt, T, t, T⁺, t⁺, Δu, U, u, U⁺, u⁺,
            targdists, gpcache
        )
    end
end


mutable struct ODEPoissonProcess
    data::ModelInput
    gm::LGCPGM

    function ODEPoissonProcess(
        data::ModelInput,
        ode::Function,
        observed_ids::Vector{Int},
        init_ode_param_setting_fn::Function,
        λ0::Union{Nothing,T1,Vector{T1}},
        is_λ0_unified::Bool,
        T::Int,
        U::Int,
        ex_time::T1,
        γ::T1,
        m::T1,
        ode_param_priors::Vector{ScaledLogitNormal},
        base_kernel::Symbol,
        ascale::T1,
        lscale::T1,
        δ::T1,
        should_diagonalize_A::Bool
    ) where {T1<:Real}
        lgcpgm = LGCPGM(
            data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
            γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
        new(data, lgcpgm)
    end
end


mutable struct GPGM
    Y::Matrix{Float64}                       # State at observation time points
    X::Matrix{Float64}                       # State at inducing time points
    θ::ODEParamSetting                       # ODE parameters
    σ::Vector{Float64}                       # std of Gaussian observation noise
    γ::Vector{Float64}                       # std of Gaussian noise for ODE gradient probabilistic model
    β::Vector{Float64}                       # inverse temperature for relaxing gradient matching constraint
    ode::Function                            # Ordinary Differential Equation. NOTE: needs to be implemented for each ODE model
    observed_ids::Vector{Int}                # indices of observed data points
    λ0::Vector{Float64}                      # coefficients of intensity function
    Δt::Float64                              # interval of observation time points
    T::Int                                   # number of interpolation observation time points to evaluate Poisson likelihood
    t::Vector{Float64}                       # interpolation observation time points to evaluate Poisson likelihood 0 < w < 1
    T⁺::Int                                  # number of observation time points to evaluate Poisson likelihood
    t⁺::Vector{Float64}                      # observation time points to evaluate Poisson likelihood 0 < w⁺
    Δu::Float64                              # interval of inducing time points
    U::Int                                   # number of interpolation inducing time points to evaluate gradient
    u::Vector{Float64}                       # interpolation inducing time points to evaluate gradient 0 < u < 1
    U⁺::Int                                  # number of inducing time points to evaluate gradient
    u⁺::Vector{Float64}                      # inducing time points to evaluate gradient 0 < u⁺
    targdists::Dict{Symbol,GPGMTarget}       # target distributions
    gpcache::Vector{SparseGPCache{Float64}}  # Sparse GP caches

    function GPGM(
        data::ModelInput,
        ode::Function,
        observed_ids::Vector{Int},
        init_ode_param_setting_fn::Function,
        λ0::Union{Nothing,T1,Vector{T1}},
        is_λ0_unified::Bool,
        T::Int,
        U::Int,
        ex_time::T1,
        # followings are input for state initialization
        γ::T1,
        m::T1,
        ode_param_priors::Vector{ScaledLogitNormal},
        base_kernel::Symbol,
        ascale::T1,
        lscale::T1,
        δ::T1,
        should_diagonalize_A::Bool
    ) where {T1<:Real}

        @unpack C, times = data
        σ = fill(0.01, C)   # dummy
        λ0 = initialize_λ0(λ0, is_λ0_unified, C, times)
        Δt, t, t⁺, Δu, u, u⁺ = initialize_time_points(T, U, ex_time)
        T⁺ = length(t⁺)
        U⁺ = length(u⁺)
        Y = calc_observed_Y(Δt, t⁺, T, T⁺, C, times, observed_ids, λ0)
        gpcache = SparseGPCache{T1}[
            SparseGPCache(
                m, base_kernel, ascale, lscale, δ, U⁺, u⁺, t⁺;
                should_diagonalize_A=should_diagonalize_A)
            for c in 1:C]
        X = zeros(U⁺, C)
        θ = init_ode_param_setting_fn(C, ode_param_priors...)
        θ_normal_priors = get_θ_normal_priors(θ)
        γ = [γ]
        β = [1.0]
        targdists = Dict{Symbol,GPGMTarget}()
        for targstate in [:yxθ, :yx, :xθ, :y, :x, :θ, :yxθσϕ, :yxθϕ, :yxϕ, :xθϕ, :σ, :ϕ]
            targdists[targstate] = GPGMTarget(
                targstate, T, T⁺, U⁺, C, Y, X, θ, gpcache, θ_normal_priors, σ, γ, β, ode
            )
        end

        return new(
            Y, X, θ, σ, γ, β, ode, observed_ids, λ0, Δt, T, t, T⁺, t⁺, Δu, U, u, U⁺, u⁺,
            targdists, gpcache
        )
    end
end


mutable struct ODEModel
    data::ModelInput
    gm::GPGM

    function ODEModel(
        data::ModelInput,
        ode::Function,
        observed_ids::Vector{Int},
        init_ode_param_setting_fn::Function,
        λ0::Union{Nothing,T1,Vector{T1}},
        is_λ0_unified::Bool,
        T::Int,
        U::Int,
        ex_time::T1,
        γ::T1,
        m::T1,
        ode_param_priors::Vector{ScaledLogitNormal},
        base_kernel::Symbol,
        ascale::T1,
        lscale::T1,
        δ::T1,
        opt_hp::Bool,
        should_diagonalize_A::Bool
    ) where {T1<:Real}
        gpgm = GPGM(
            data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
            γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
        if opt_hp
            optimize_hyperparams!(gpgm)
        end
        new(data, gpgm)
    end

end


Base.show(io::IO, s::LGCPGM) = print(io, "Log Gaussian Cox Process-based Gradient Matching (ODE: $(s.ode))")
Base.show(io::IO, s::ODEPoissonProcess) = print(io, "ODE-guided Poisson Processes (ODE: $(s.gm.ode))")
Base.show(io::IO, s::GPGM) = print(io, "Gaussian Process based Gradient Matching (ODE: $(s.ode))")
Base.show(io::IO, s::ODEModel) = print(io, "ODE Model (ODE: $(s.gm.ode))")


param2tuple(θ::AbstractODEParamsFullSupport) = Tuple([getfield(θ, n) for n in 1:nfields(θ)])
tuple2param(mod::ODEPoissonProcess, θ::Tuple) = typeof(mod.gm.θ.params)(θ...)

function get_θ_normal_priors(θ::ODEParamSetting)
    normal_priors = Normal{Float64}[]
    for (prior, n) in zip(getfields(θ.priors), θ.paramlengths)
        for _ in 1:n
            push!(normal_priors, prior.normal)
        end
    end
    return normal_priors
end

function initialize_λ0(
    λ0::Union{Nothing,T,Vector{T}}, is_λ0_unified::Bool, C::Int, times::Dict{Int,Vector{T}}
) where {T<:Real}

    if isnothing(λ0)
        λ0 = is_λ0_unified ?
            fill(StatsBase.geomean([length(times[c]) for c in 1:C]), C) :
            [float(length(times[c])) for c in 1:C]
    elseif typeof(λ0) <: T
        λ0 = fill(λ0, C)
    else
        @assert length(λ0) == C "Invalid length of λ0."
    end
    @assert all(λ0 .> 0) "λ0 must be positive value. When you model unavailable components, " *
                        "You must specify λ0 explicitly."
    return λ0
end

function initialize_time_points(
    T::Int,
    U::Int,
    ex_time::T1
) where {T1<:Real}

    Δt = 1 / T # window width
    t = Δt .* (collect(1:T) .- 0.5)
    T_ex = round(ex_time / Δt, RoundNearestTiesAway)
    t_ex = 1 .+ Δt .* (collect(1:T_ex) .- 0.5)
    t⁺ = vcat([t, t_ex]...)
    Δu = 1 / (U - 1)
    u = Δu .* (collect(1:U) .- 1) .+ 1e-10
    U_ex = round(ex_time / Δu, RoundNearestTiesAway)
    if U_ex * Δu > ex_time
        U_ex = U_ex - 1
    end
    u_ex = 1 .+ Δu .* collect(1:U_ex) .+ 1e-10
    u⁺ = vcat([u, u_ex]...)
    # NOTE: 1e-10 is added to u and u_ex
    # This prevents observational time points from equating with inducing points, 
    # mitigating white noise influence on the kernel values. 
    # This results in eliminating zero components in the diagonal of the predictive covariance in the sparse Gaussian process.
    return (Δt, t, t⁺, Δu, u, u⁺)
end

function calc_ocurrences(
    Δt::T1, t⁺::Vector{T1}, T::Int, T⁺::Int, C::Int, times::Dict{Int,Vector{T1}}, observed_ids::Vector{Int}
) where {T1<:Real}

    M = Matrix{Union{Nothing,Int}}(nothing, T⁺, C)
    for c in 1:C
        if c in observed_ids
            M[:, c] = Union{Nothing,Int}[
                i > 1 ?
                nothing :
                sum(i - Δt / 2 .<= times[c] .< i + Δt / 2)
                for i in t⁺]
            M[T, c] += sum(times[c] .== 1)
        else
            M[:, c] .= nothing
        end
    end
    return M
end


function calc_observed_Y(
    Δt::T1, t⁺::Vector{T1}, T::Int, T⁺::Int, C::Int, times::Dict{Int,Vector{T1}}, observed_ids::Vector{Int}, λ0::Vector{T1}
) where {T1<:Real}

    Y = zeros(T⁺, C)
    for c in 1:C
        if c in observed_ids
            Y[:, c] = [sum(i - Δt / 2 .<= times[c] .< i + Δt / 2) for i in t⁺]
            Y[T, c] += sum(times[c] .== 1)
            Y[:, c] .*= T / λ0[c]
        else
            error("ODE Model currently supports the problem where all components are observed.")
        end
    end
    return Y
end


function optimize_hyperparams!(gm::GPGM)
    @unpack Y, X, σ, gpcache, T, U, t, u, t⁺, u⁺ = gm
    for c in 1:length(σ)
        @unpack m, ϕ = gpcache[c]
        y = Y[1:T, c]
        if ϕ.base_kernel isa KernelFunctions.RBFKernel
            k = GaussianProcesses.SEIso(log(ϕ.𝓁), log(ϕ.α)) + GaussianProcesses.Noise(log(ϕ.δ))
        elseif ϕ.base_kernel isa KernelFunctions.Matern52Kernel
            k = GaussianProcesses.Matern52Iso(log(ϕ.𝓁), log(ϕ.α)) + GaussianProcesses.Noise(log(ϕ.δ))
        end
        _mean = GaussianProcesses.MeanConst(m)
        gpe = GaussianProcesses.GPE(t, y, _mean, k, 0.0)
        GaussianProcesses.optimize!(gpe, method=ConjugateGradient(), domean=false, kern=true, noise=true)   # Optimise the hyperparameters
        # opt_m = gpe.mean.β
        opt_ascale = sqrt(gpe.kernel.kleft.σ2)
        opt_lscale = sqrt(gpe.kernel.kleft.ℓ2)
        opt_δ = sqrt(gpe.kernel.kright.σ2)
        opt_σ = gpe.logNoise.value |> exp
        update_kernel_params!(gpcache[c], KernelParams(ϕ.base_kernel, opt_ascale, opt_lscale, opt_δ))
        σ[c] = opt_σ
    end
end


function get_θ₊(mod::Union{ODEPoissonProcess,ODEModel}, θ::AbstractODEParamsFullSupport)
    θ_priors = mod.gm.θ.priors
    θ₊ = get_θ₊(θ, θ_priors)  # NOTE: needs to be implemented for each ODE model
    return θ₊
end

function get_Λ(X::Matrix{T}) where {T<:Real}
    Λ = exp.(X)
    return Λ
end


# function optimize_kernel_params!(mod::ODEPoissonProcess)
#     _chain = Chain(mod, target=[:y, :x], do_gm=false, burnin=true, n_burnin=1000)
#     train!(mod, 1000, _chain, show_progress=true)

#     @unpack C = mod.data
#     @unpack Y, X, gpcache, λ0, M, T, U, t, u = mod.gm

#     for c in 1:C

#         base_kernel = gpcache[c].ϕ.base_kernel
#         gp_mean = gpcache[c].m
#         m = M[1:T, c] 
#         _λ0 = λ0[c]
#         a_min = 0.1
#         l_min = 0.1
#         w_min = 0.001

#         function _nll(v::AbstractVector)
#             y, x = v[1:T], v[T+1:T+U]
#             a, l, w = exp.(v[T+U+1:T+U+3])
#             a += a_min
#             l += l_min
#             w += w_min
#             k = a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
#             Kuu = kernelmatrix(k, u) |> Hermitian |> Matrix |> PDMat
#             Ktu = kernelmatrix(k, t, u)
#             Ktt = kernelmatrix(k, t)
#             L⁻¹ = Kuu.chol.L |> inv
#             Kuu⁻¹ = L⁻¹' * L⁻¹
#             KtuKuu⁻¹ = Ktu * Kuu⁻¹
#             diagK̂ = diag(Ktt - Ktu * (Kuu \ Ktu')) .+ 1e-6
#             gp = MvNormal(fill(gp_mean, U), Kuu)
#             pred_mean = gp_mean .- KtuKuu⁻¹ * (x .- gp_mean)
#             λ = (_λ0 / T) .* exp.(y)
#             ll = 0
#             ll += logpdf(gp, x)
#             ll += [logpdf(Normal(_m, _σ), _y) for (_m, _σ, _y) in zip(pred_mean, sqrt.(diagK̂), y)] |> sum
#             ll += [logpdf(Poisson(_λ), _m) for (_λ, _m) in zip(λ, m)] |> sum
#             nll = -ll
#             return nll
#         end

#         initial_v = [
#             vec(Y[1:T, c])...,
#             vec(X[1:U, c])...,
#             log(max(a_min, gpcache[c].ϕ.α - a_min)),
#             log(max(l_min, gpcache[c].ϕ.𝓁 - l_min)),
#             log(max(w_min, gpcache[c].ϕ.δ - w_min))
#         ]
#         res = optimize(_nll, initial_v, LBFGS(), Optim.Options(show_trace=true))
#         optimal_v = Optim.minimizer(res)
#         opt_a, opt_l, opt_w = exp.(optimal_v[T+U+1:T+U+3])
#         opt_a += a_min
#         opt_l += l_min
#         opt_w += w_min
#         update_kernel_params!(gpcache[c], KernelParams(base_kernel, opt_a, opt_l, opt_w))
#     end
# end