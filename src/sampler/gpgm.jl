Base.@kwdef mutable struct GPGMTarget
    D::Int                            # dimension of target distribution
    targstate::Symbol                 # target state to sample. {:yxθ, :xθ, :x, :θ}
    lp::Union{Function,Any} = []      # calculate log probability
    g::Union{Function,Any} = []       # calculate gradient of log probability
    H::Union{Function,Any} = []       # calculate Hessian matrix of log probability
    len_y::Int                        # length of y
    do_gm::Bool
end


function GPGMTarget(
    targstate::Symbol,
    T::Int,
    T⁺::Int,
    U⁺::Int,
    C::Int,
    Y::Matrix{Float64},
    X::Matrix{Float64},
    θ::ODEParamSetting,
    gpcache::Vector{SparseGPCache{Float64}},
    θ_normal_priors::Vector{Normal{Float64}},
    σ::Vector{Float64},
    γ::Vector{Float64},
    β::Vector{Float64},
    ode::Function,
)
    @assert length(γ) == 1
    @assert length(β) == 1

    len_y = C * (T⁺ - T)
    len_x = length(X)
    len_θ = sum(θ.paramlengths)
    len_σ = C
    len_ϕ = 3 * C

    # log  probability
    function lp(v::Vector{<:Number}; do_gm::Bool, should_sum::Bool)
        scheme = GPGMScheme()
        θvec = get_θ_vec(θ)

        (
            _ϕvec, _ms, _Kuus, _Kuu⁻¹s, _Kuu′s, _Ds, _As, _KtuKuu⁻¹s, _diagK̂s
        ) = get_gp_variables(v, targstate, C, gpcache, len_ϕ)

        Luus = [s.Luu for s in gpcache]
        _Y, _X, _θvec, _σ = get_states_from_vector(
            v, targstate, T, T⁺, U⁺, C, Luus, θ, scheme, Y, X, θvec, σ)

        y_mean_vec = [m .+ KtuKuu⁻¹ * (x .- m) for (m, KtuKuu⁻¹, x) in zip(_ms, _KtuKuu⁻¹s, eachcol(_X))]
        y_std_vec = [sqrt.(diagK̂ .+ σ_c^2) for (diagK̂, σ_c) in zip(_diagK̂s, _σ)]

        finite_gps = [MvNormal(fill(m, U⁺), Kuu) for (m, Kuu) in zip(_ms, _Kuus)]

        error_covs = [
            s.should_diagonalize_A ?
                diagm(diag(A)) + γ[1]^2 * I :
                A + γ[1]^2 * I
            for (A, s) in zip(_As, gpcache)]
        grad_error_dists = [MvNormal(zeros(U⁺), Aγ) for Aγ in error_covs]
        grad_errors = get_gradient_errors(U⁺, C, _Ds, _ms, ode, θ, _X, _θvec, scheme)

        # calculate log probability
        _lp₍y⎸xσmϕ₎ = lp₍y⎸xσmϕ₎(_Y, y_mean_vec, y_std_vec)
        _lp₍x⎸mϕ₎ = lp₍x⎸mϕ₎(_X, finite_gps)

        if do_gm
            _lp₍x⎸mϕθγ₎ = β[1] * lp₍x⎸mϕθγ₎(grad_errors, grad_error_dists)
            _lp₍θ₎ = lp₍θ₎(_θvec, θ_normal_priors)
        else
            _lp₍x⎸mϕθγ₎ = 0
            _lp₍θ₎ = 0
        end
        lps = (
            lp₍y⎸xσmϕ₎=_lp₍y⎸xσmϕ₎,
            lp₍x⎸mϕ₎=_lp₍x⎸mϕ₎,
            lp₍x⎸mϕθγ₎=_lp₍x⎸mϕθγ₎,
            lp₍θ₎=_lp₍θ₎
        )

        if should_sum
            return sum(lps)
        else
            lps
        end
    end

    function logp(targ::GPGMTarget, v::Vector{<:Number}; should_sum::Bool=true)
        lp(v; do_gm=targ.do_gm, should_sum=should_sum)
    end

    # gradient of log probability
    function dlp(v::Vector{<:Number}; do_gm::Bool)
        scheme = GPGMScheme()
        θvec = get_θ_vec(θ)

        (
            _ϕvec, _ms, _Kuus, _Kuu⁻¹s, _Kuu′s, _Ds, _As, _KtuKuu⁻¹s, _diagK̂s
        ) = get_gp_variables(v, targstate, C, gpcache, len_ϕ)

        Luus = [s.Luu for s in gpcache]
        _Y, _X, _θvec, _σ = get_states_from_vector(
            v, targstate, T, T⁺, U⁺, C, Luus, θ, scheme, Y, X, θvec, σ)

        y_mean_vec = [m .+ KtuKuu⁻¹ * (x .- m) for (m, KtuKuu⁻¹, x) in zip(_ms, _KtuKuu⁻¹s, eachcol(_X))]
        y_std_vec = [sqrt.(diagK̂ .+ σ_c^2) for (diagK̂, σ_c) in zip(_diagK̂s, _σ)]

        finite_gps = [MvNormal(fill(m, U⁺), Kuu) for (m, Kuu) in zip(_ms, _Kuus)]

        grads = []
        # calculate derivative of log probability
        if occursin("y", string(targstate))
            _dlp₍y⎸xσmϕ₎dy = dlp₍y⎸xσmϕ₎dy(_Y, y_mean_vec, y_std_vec, T, scheme)
            push!(grads, _dlp₍y⎸xσmϕ₎dy)
        end

        if do_gm && (
            occursin("x", string(targstate)) ||
            occursin("θ", string(targstate)) ||
            occursin("ϕ", string(targstate))
        )

            error_covs = [
                s.should_diagonalize_A ?
                    diagm(diag(A)) + γ[1]^2 * I :
                    A + γ[1]^2 * I
                for (A, s) in zip(_As, gpcache)]
            grad_error_dists = [MvNormal(zeros(U⁺), Aγ) for Aγ in error_covs]
            grad_errors = get_gradient_errors(U⁺, C, _Ds, _ms, ode, θ, _X, _θvec, scheme)
            θℝ = typeof(θ.params)(_θvec)
            _∂Lg∂ẋ₍gp₎ = vcat([
                Distributions.gradlogpdf(grad_error_dist, err)
                for (grad_error_dist, err)
                in
                zip(grad_error_dists, grad_errors)
            ]...)    # Nx length
            _∂Lg∂Gode = reshape(-_∂Lg∂ẋ₍gp₎, size(_X)...)  # U × C/M            
        end

        if occursin("x", string(targstate))
            _dlp₍y⎸xσmϕ₎dx = dlp₍y⎸xσmϕ₎dx(_Y, y_mean_vec, y_std_vec, Luus, _KtuKuu⁻¹s)
            _dlp₍x⎸mϕ₎dx = dlp₍x⎸mϕ₎dx(_X, finite_gps)
            _dlp₍x⎸mϕθγ₎dx = do_gm ?
                            β[1] * dlp₍x⎸mϕθγ₎dx(_X, _∂Lg∂Gode, θℝ, θ.priors, Luus, _Ds, scheme) :
                            zeros(len_x)

            push!(grads, 
                _dlp₍y⎸xσmϕ₎dx .+ 
                _dlp₍x⎸mϕ₎dx .+ 
                _dlp₍x⎸mϕθγ₎dx
            )
        end

        if occursin("θ", string(targstate))
            _dlp₍x⎸mϕθγ₎dθ = do_gm ?
                        β[1] * dlp₍x⎸mϕθγ₎dθ(_X, _∂Lg∂Gode, θℝ, θ.priors, scheme) :
                        zeros(len_θ)
            _dlp₍θ₎dθ = do_gm ?
                        dlp₍θ₎dθ(_θvec, θ_normal_priors) :
                        zeros(len_θ)
            push!(grads, _dlp₍x⎸mϕθγ₎dθ .+ _dlp₍θ₎dθ)
        end

        if occursin("σ", string(targstate))
            _dlp₍y⎸xσmϕ₎dσ = dlp₍y⎸xσmϕ₎dσ(_Y, _σ, y_mean_vec, y_std_vec)
            push!(grads, _dlp₍y⎸xσmϕ₎dσ)
        end

        if occursin("ϕ", string(targstate))
            _dlp₍y⎸xσmϕ₎dϕ = dlp₍y⎸xσmϕ₎dϕ(_Y, _ϕvec, gpcache, y_mean_vec, y_std_vec)
            _dlp₍x⎸mϕ₎dϕ = dlp₍x⎸mϕ₎dϕ(_X, _ϕvec, finite_gps, gpcache)
            _dlp₍x⎸mϕθγ₎dϕ = do_gm ?
                            β[1] * dlp₍x⎸mϕθγ₎dϕ(_X, _ϕvec, grad_errors, grad_error_dists, _Kuu′s, _Kuu⁻¹s, gpcache) :
                            zeros(len_ϕ)
            push!(grads, _dlp₍y⎸xσmϕ₎dϕ .+ _dlp₍x⎸mϕ₎dϕ .+ _dlp₍x⎸mϕθγ₎dϕ)
        end

        return vcat(grads...)
    end

    function dlogp(targ::GPGMTarget, v::Vector{<:Number})
        float.(dlp(v; do_gm=targ.do_gm))
    end

    dlp_with_gm(v::Vector{<:Number}) = dlp(v, do_gm=true)
    dlp_without_gm(v::Vector{<:Number}) = dlp(v, do_gm=false)

    # hessian of log probability
    d2logp_with_gm(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_with_gm, v)
    d2logp_without_gm(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_without_gm, v)

    function d2logp(targ::GPGMTarget, v::Vector{<:Number})
        if targ.do_gm
            float.(d2logp_with_gm(v))
        else
            float.(d2logp_without_gm(v))
        end
    end

    D = 0
    if occursin("y", string(targstate))
        D += len_y
    end
    if occursin("x", string(targstate))
        D += len_x
    end
    if occursin("θ", string(targstate))
        D += len_θ
    end
    if occursin("σ", string(targstate))
        D += len_σ
    end
    if occursin("ϕ", string(targstate))
        D += len_ϕ
    end

    return GPGMTarget(D, targstate, logp, dlogp, d2logp, len_y, true)
end


function ode_gradmean(
    X::Matrix{T1},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors,
    U⁺::Int,
    C::Int,
    ode::Function
) where {T1<:Real}

    @assert size(X) == (U⁺, C)
    θ₊ = get_θ₊(θ, θ_priors)
    F1 = ode(X, θ₊)  # NOTE: needs to be implemented for each ODE model
    @assert all(isfinite.(F1)) "non-finite values exist"
    return F1
end


function ∂F1∂x(
    X::Matrix{T1},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors
) where {T1<:Real}

    θ₊ = get_θ₊(θ, θ_priors)
    grads = ∂F1∂x(X, θ₊)
    return grads  # [Nx × C] × (Nx*C)
end

function ∂F1∂θ(
    X::Matrix{T1},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors
) where {T1<:Real}

    θ₊ = get_θ₊(θ, θ_priors)
    grads = ∂F1∂θ(X, θ₊, θ, θ_priors)  # NOTE: needs to be implemented for each ODE model
    return grads  # [Nx × C] * length(θ)
end
