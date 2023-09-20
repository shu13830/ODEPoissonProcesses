abstract type AbstractGradientMatchingScheme end
struct LGCPGMScheme <: AbstractGradientMatchingScheme end
struct GPGMScheme <: AbstractGradientMatchingScheme end

Base.@kwdef mutable struct LGCPGMTarget
    D::Int                        # dimension of target distribution
    targstate::Symbol             # target state to sample. {:yxθ, :xθ, :x, :θ}
    lp::Union{Function,Any} = []  # calculate log probability
    g::Union{Function,Any} = []   # calculate gradient of log probability
    H::Union{Function,Any} = []   # calculate Hessian matrix of log probability
    len_y::Int                    # length of y
    do_gm::Bool                   # whether to do gradient matching
    should_weight_γ::Bool         # whether to weight γ
end

function LGCPGMTarget(
    targstate::Symbol,
    T::Int,
    T⁺::Int,
    U⁺::Int,
    C::Int,
    M::Matrix{Union{Nothing,Int}},
    Y::Matrix{Float64},
    X::Matrix{Float64},
    θ::ODEParamSetting,
    gpcache::Vector{SparseGPCache{Float64}},
    θ_normal_priors::Vector{Normal{Float64}},
    σ::Vector{Float64},
    γ::Vector{Float64},
    β::Vector{Float64},
    ode::Function,
    λ0::Vector{Float64},
)
    @assert length(γ) == 1
    @assert length(β) == 1

    len_y = length(Y)
    len_x = length(X)
    len_θ = sum(θ.paramlengths)
    len_ϕ = 3 * C

    # log probability
    function lp(v::Vector{<:Number}; do_gm::Bool, should_sum::Bool, should_weight_γ::Bool)
        scheme = LGCPGMScheme()
        θvec = get_θ_vec(θ)

        (
            _ϕvec, _ms, _Kuus, _Kuu⁻¹s, _Kuu′s, _Ds, _As, _KtuKuu⁻¹s, _diagK̂s
        ) = get_gp_variables(v, targstate, C, gpcache, len_ϕ)

        Luus = [s.Luu for s in gpcache]
        _Y, _X, _θvec, _σ = get_states_from_vector(
            v, targstate, T, T⁺, U⁺, C, Luus, θ, scheme, Y, X, θvec, σ)

        Λ = (λ0 ./ T)' .* exp.(_Y)

        y_mean_vec = [m .+ KtuKuu⁻¹ * (x .- m) for (m, KtuKuu⁻¹, x) in zip(_ms, _KtuKuu⁻¹s, eachcol(_X))]
        y_std_vec = [sqrt.(diagK̂ .+ σ_c^2) for (diagK̂, σ_c) in zip(_diagK̂s, _σ)]

        finite_gps = [MvNormal(fill(m, U⁺), Kuu) for (m, Kuu) in zip(_ms, _Kuus)]

        if should_weight_γ
            error_covs = [
                s.should_diagonalize_A ?
                    diagm(diag(A)) + Diagonal((γ[1] ./ exp.(x)).^2) :
                    A + Diagonal((γ[1] ./ exp.(x)).^2)
                for (A, s, x) in zip(_As, gpcache, eachcol(_X))]
        else
            error_covs = [
                s.should_diagonalize_A ?
                    diagm(diag(A)) + γ[1]^2 * I :
                    A + γ[1]^2 * I
                for (A, s) in zip(_As, gpcache)]
        end

        grad_error_dists = [MvNormal(zeros(U⁺), Aγ) for Aγ in error_covs]
        grad_errors = get_gradient_errors(U⁺, C, _Ds, _ms, ode, θ, _X, _θvec, scheme)

        # calculate log probability
        _lp₍m⎸y₎ = lp₍m⎸y₎(M, Λ)
        _lp₍y⎸xσmϕ₎ = lp₍y⎸xσmϕ₎(_Y, y_mean_vec, y_std_vec)
        _lp₍x⎸mϕ₎ = lp₍x⎸mϕ₎(_X, finite_gps)

        if do_gm
            _lp₍x⎸mϕθγ₎ = β[1] * lp₍x⎸mϕθγ₎(grad_errors, grad_error_dists)
            _lp₍θ₎ = lp₍θ₎(_θvec, θ_normal_priors)
        else
            _lp₍x⎸mϕθγ₎ = 0.
            _lp₍θ₎ = 0.
        end
        lps = (
            lp₍m⎸y₎=_lp₍m⎸y₎,
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

    function logp(targ::LGCPGMTarget, v::Vector{<:Number}; should_sum::Bool=true)
        lp(v; do_gm=targ.do_gm, should_sum=should_sum, should_weight_γ=targ.should_weight_γ)
    end

    # gradient of log probability
    function dlp(v::Vector{<:Number}; do_gm::Bool, should_weight_γ::Bool)
        scheme = LGCPGMScheme()
        θvec = get_θ_vec(θ)

        (
            _ϕvec, _ms, _Kuus, _Kuu⁻¹s, _Kuu′s, _Ds, _As, _KtuKuu⁻¹s, _diagK̂s
        ) = get_gp_variables(v, targstate, C, gpcache, len_ϕ)

        Luus = [s.Luu for s in gpcache]
        _Y, _X, _θvec, _σ = get_states_from_vector(
            v, targstate, T, T⁺, U⁺, C, Luus, θ, scheme, Y, X, θvec, σ)

        Λ = (λ0 ./ T)' .* exp.(_Y)

        y_mean_vec = [m .+ KtuKuu⁻¹ * (x .- m) for (m, KtuKuu⁻¹, x) in zip(_ms, _KtuKuu⁻¹s, eachcol(_X))]
        y_std_vec = [sqrt.(diagK̂ .+ σ_c^2) for (diagK̂, σ_c) in zip(_diagK̂s, _σ)]

        finite_gps = [MvNormal(fill(m, U⁺), Kuu) for (m, Kuu) in zip(_ms, _Kuus)]

        grads = []
        # calculate derivative of log probability
        if occursin("y", string(targstate))
            _dlp₍m⎸y₎dy = dlp₍m⎸y₎dy(M, Λ)
            _dlp₍y⎸xσmϕ₎dy = dlp₍y⎸xσmϕ₎dy(_Y, y_mean_vec, y_std_vec, T, scheme)
            push!(grads, _dlp₍m⎸y₎dy .+ _dlp₍y⎸xσmϕ₎dy)
        end

        if do_gm && (
            occursin("x", string(targstate)) ||
            occursin("θ", string(targstate)) ||
            occursin("ϕ", string(targstate))
        )

            if should_weight_γ
                error_covs = [
                    s.should_diagonalize_A ?
                        diagm(diag(A)) + Diagonal((γ[1] ./ exp.(x)).^2) :
                        A + Diagonal((γ[1] ./ exp.(x)).^2)
                    for (A, s, x) in zip(_As, gpcache, eachcol(_X))]
            else
                error_covs = [
                    s.should_diagonalize_A ?
                        diagm(diag(A)) + γ[1]^2 * I :
                        A + γ[1]^2 * I
                    for (A, s) in zip(_As, gpcache)]
            end

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
            if should_weight_γ
                _dlp₍x⎸mϕθγ₎dx = do_gm ?
                    β[1] * dlp₍x⎸mϕθγ₎dx(
                        _X, _∂Lg∂Gode, grad_error_dists, grad_errors, γ[1], θℝ, θ.priors, Luus, _Ds, scheme
                    ) :
                    zeros(len_x)
            else
                _dlp₍x⎸mϕθγ₎dx = do_gm ?
                    β[1] * dlp₍x⎸mϕθγ₎dx(_X, _∂Lg∂Gode, θℝ, θ.priors, Luus, _Ds, scheme) :
                    zeros(len_x)
            end

            push!(grads, _dlp₍y⎸xσmϕ₎dx .+ _dlp₍x⎸mϕ₎dx .+ _dlp₍x⎸mϕθγ₎dx)
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

        if occursin("ϕ", string(targstate))
            _dlp₍y⎸xσmϕ₎dϕ = dlp₍y⎸xσmϕ₎dϕ(_Y, _ϕvec, gpcache, y_mean_vec, y_std_vec)
            _dlp₍x⎸mϕ₎dϕ = dlp₍x⎸mϕ₎dϕ(_X, _ϕvec, finite_gps, gpcache)
            _dlp₍x⎸mϕθγ₎dϕ = do_gm ?
                β[1] * dlp₍x⎸mϕθγ₎dϕ(
                    _X, _ϕvec, grad_errors, grad_error_dists, _Kuu′s, _Kuu⁻¹s, gpcache
                ) :
                zeros(len_ϕ)

            push!(grads, _dlp₍y⎸xσmϕ₎dϕ .+ _dlp₍x⎸mϕ₎dϕ .+ _dlp₍x⎸mϕθγ₎dϕ)
        end

        return vcat(grads...)
    end

    function dlogp(targ::LGCPGMTarget, v::Vector{<:Number})
        float.(dlp(v; do_gm=targ.do_gm, should_weight_γ=targ.should_weight_γ))
    end

    dlp_with_gm(v::Vector{<:Number}) = dlp(v, do_gm=true, should_weight_γ=false)
    dlp_without_gm(v::Vector{<:Number}) = dlp(v, do_gm=false, should_weight_γ=false)
    dlp_with_gm_weightγ(v::Vector{<:Number}) = dlp(v, do_gm=true, should_weight_γ=true)
    dlp_without_gm_weightγ(v::Vector{<:Number}) = dlp(v, do_gm=false, should_weight_γ=true)

    # hessian of log probability
    d2logp_with_gm(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_with_gm, v)
    d2logp_without_gm(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_without_gm, v)
    d2logp_with_gm_weightγ(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_with_gm_weightγ, v)
    d2logp_without_gm_weightγ(v::Vector{<:Number}) = ForwardDiff.jacobian(dlp_without_gm_weightγ, v)

    function d2logp(targ::LGCPGMTarget, v::Vector{<:Number})
        if targ.do_gm
            if targ.should_weight_γ
                d2lp = float.(d2logp_with_gm_weightγ(v))
            else
                d2lp = float.(d2logp_with_gm(v))
            end
        else
            if targ.should_weight_γ
                d2lp = float.(d2logp_without_gm_weightγ(v))
            else
                d2lp = float.(d2logp_without_gm(v))
            end
        end
        d2lp
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
    if occursin("ϕ", string(targstate))
        D += len_ϕ
    end

    return LGCPGMTarget(D, targstate, logp, dlogp, d2logp, len_y, true, false)
end

function reflect_ϕ!(
    ϕvec::Vector{T1}, C::Int, gpcache::Vector{SparseGPCache{T1}}
) where {T1<:Real}

    ϕmat = reshape(ϕvec, 3, C)
    for c in 1:C
        a, l, w = ϕmat[:, c]
        ϕ = KernelParams(gpcache[c].ϕ.base_kernel, a, l, w)
        update_kernel_params!(gpcache[c], ϕ)
    end
end

function get_states_from_vector(
    v::AbstractVector,
    targstate::Symbol,
    T::Int,
    T⁺::Int,
    U⁺::Int,
    C::Int,
    Luus::Vector{<:LowerTriangular},
    θ::ODEParamSetting,
    scheme::Union{LGCPGMScheme,GPGMScheme},
    Y::Matrix{T1},
    X::Matrix{T2},
    θvec::Vector{T3},
    σ::Vector{T4}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}

    if occursin("y", string(targstate))
        if scheme isa LGCPGMScheme
            len_y = T⁺ * C
            _y = v[1:len_y]
            _Y = reshape(_y, T⁺, C)
        else  # scheme isa GPGMScheme
            len_y = (T⁺ - T) * C
            _y_ext = v[1:len_y]
            _Yext = reshape(_y_ext, T⁺ - T, C)
            _Y = vcat(Y[1:T, :], _Yext)
        end
    else
        len_y = 0
        _Y = Y  # use cached Y
    end

    if occursin("x", string(targstate))
        len_x = U⁺ * C
        _x_mapped = v[len_y+1:len_y+len_x]
        _X_mapped = reshape(_x_mapped, U⁺, C)
        _X = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(_X_mapped))
        for (c, x) in enumerate(eachcol(_X_mapped))
            _X[:, c] = Luus[c] * x
        end
    else
        len_x = 0
        _X = X  # use cached X
    end

    if occursin("θ", string(targstate))
        len_θ = sum(θ.paramlengths)
        _θvec = v[len_y+len_x+1:len_y+len_x+len_θ]
    else
        len_θ = 0
        _θvec = θvec  # use cached θvec
    end

    if occursin("σ", string(targstate))
        @assert scheme isa GPGMScheme
        len_σ = C
        _σ = v[len_y+len_x+len_θ+1:len_y+len_x+len_θ+len_σ] .|> exp
    else
        len_σ = 0
        _σ = σ  # use cached σ
    end

    return _Y, _X, _θvec, _σ
end

function get_gp_variables(
    v::AbstractVector, targstate::Symbol, C::Int, gpcache::Vector{SparseGPCache{T1}}, len_ϕ::Int
) where {T1<:Real}
    if occursin("ϕ", string(targstate))
        ms = Float64[]
        Kuus = AbstractMatrix[]
        Kuu⁻¹s = AbstractMatrix[]
        Kuu′s = AbstractMatrix[]
        Ds = AbstractMatrix[]
        As = AbstractMatrix[]
        KtuKuu⁻¹s = AbstractMatrix[]
        diagK̂s = AbstractVector[]

        ϕvec = v[end-len_ϕ+1:end] .|> exp
        ϕmat = reshape(ϕvec, 3, C)
        for c in 1:C
            a, l, w = ϕmat[:, c]
            base_kernel = gpcache[c].ϕ.base_kernel
            @unpack u, t = gpcache[c]
            k = a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
            Kuu = k.(u, u') |> PDMat
            Luu = Kuu.chol.L
            Luu⁻¹ = Luu |> inv
            Kuu⁻¹ = Luu⁻¹' * Luu⁻¹
            Kuu′ = dKxy_dx(a, l, w, u, u, base_kernel)
            Kuu″ = d2Kxy_dxdy(a, l, w, u, u, base_kernel)
            D = calc_gp_gradient_mean_linear_operator(Kuu′, Kuu⁻¹)
            A = calc_gp_gradient_cov(Kuu″, Kuu′, Kuu)
            if u == t
                KtuKuu⁻¹ = diagm(ones(length(t)))
                K̂ = zeros(length(t), length(t))
            else
                KtuKuu⁻¹ = k.(t, u') * Kuu⁻¹
                K̂ = diagm(diag(k.(t, t') - k.(t, u') * Kuu⁻¹ * k.(u, t')))
            end
            diagK̂ = diag(K̂)

            push!(ms, gpcache[c].m)
            push!(Kuus, Kuu)
            push!(Kuu⁻¹s, Kuu⁻¹)
            push!(Kuu′s, Kuu′)
            push!(Ds, D)
            push!(As, A)
            push!(KtuKuu⁻¹s, KtuKuu⁻¹)
            push!(diagK̂s, diagK̂)
        end
    else
        ϕvec = get_ϕ_vec(gpcache)
        ms = [gpcache[c].m for c in 1:C]
        Kuus = [gpcache[c].Kuu for c in 1:C]
        Kuu⁻¹s = [gpcache[c].Kuu⁻¹ for c in 1:C]
        Kuu′s = [gpcache[c].Kuu′ for c in 1:C]
        Ds = [gpcache[c].D for c in 1:C]
        As = [gpcache[c].A for c in 1:C]
        KtuKuu⁻¹s = [gpcache[c].KtuKuu⁻¹ for c in 1:C]
        diagK̂s = [gpcache[c].diagK̂ for c in 1:C]
    end
    return ϕvec, ms, Kuus, Kuu⁻¹s, Kuu′s, Ds, As, KtuKuu⁻¹s, diagK̂s
end

# ========================================================================
# log probability functions
# ========================================================================
# --- lp₍m⎸y₎ ----------------------------------------------------------
function lp₍m⎸y₎(
    M::Matrix{Union{Nothing,Int}},
    Λ::Matrix{T1}
) where {T1<:Real}

    lp = 0
    for (m, λ) in zip(eachcol(M), eachcol(Λ))
        lp += lp₍m⎸y₎(m[:], λ[:])
    end

    return lp
end  # TEST

function lp₍m⎸y₎(
    m::Vector{Union{Nothing,Int}},
    λ::Vector{T1}
) where {T1<:Real}

    lp = 0
    for (m_t, λ_t) in zip(m, λ)
        if ~isnothing(m_t)
            lp += logpdf(Poisson(λ_t), m_t)
        end
    end

    return lp
end  # TEST

# --- lp₍y⎸xσmϕ₎ ---------------------------------------------------------
function lp₍y⎸xσmϕ₎(
    Y::Matrix{T1},
    y_mean_vec::Vector{Vector{T2}},
    y_std_vec::Vector{Vector{T3}}
) where {T1<:Real,T2<:Real,T3<:Real}

    lp = 0
    for (y, y_mean, y_std) in zip(eachcol(Y), y_mean_vec, y_std_vec)
        lp += lp₍y⎸xσmϕ₎(y[:], y_mean, y_std)
    end

    return lp
end  # TEST

function lp₍y⎸xσmϕ₎(
    y::Vector{T1},
    y_mean::Vector{T2},
    y_std::Vector{T3}
) where {T1<:Real,T2<:Real,T3<:Real}

    lp = 0
    for (_μ, _std, _y) in zip(y_mean, y_std, y)
        lp += logpdf(Normal(_μ, _std), _y)
    end

    return lp
end  # TEST

# --- lp₍x⎸mϕ₎ -----------------------------------------------------------
function lp₍x⎸mϕ₎(
    X::Matrix{T1},
    finite_gps::Vector{<:AbstractMvNormal}
) where {T1<:Real}

    lp = 0
    for (x, finite_gp) in zip(eachcol(X), finite_gps)
        lp += logpdf(finite_gp, x)
    end
    return lp
end  # TEST

# --- lp₍θ₎ -------------------------------------------------------------
function lp₍θ₎(
    θvec::Vector{T1},
    θ_normal_priors::Vector{Normal{T2}}
) where {T1<:Real,T2<:Real}

    lp = sum([logpdf(normal_prior, _θ) for (_θ, normal_prior) in zip(θvec, θ_normal_priors)])
    return lp
end  # TEST

# --- lp₍x⎸mϕθγ₎ ---------------------------------------------------------
function lp₍x⎸mϕθγ₎(
    grad_errors::Vector{Vector{T1}},
    grad_error_dists::Vector{<:AbstractMvNormal}
) where {T1<:Real}

    lp = 0
    for (c, err) in enumerate(grad_errors)
        lp += logpdf(grad_error_dists[c], err)
    end
    return lp
end  # TEST

function get_gradient_errors(
    U⁺::Int,
    C::Int,
    Ds::Vector{<:AbstractMatrix},
    ms::Vector{T1},
    ode::Function,
    θ::ODEParamSetting,
    X::Matrix{T2},
    θvec::Vector{T3},
    scheme::Union{LGCPGMScheme,GPGMScheme}
) where {T1<:Real,T2<:Real,T3<:Real}

    θℝ = typeof(θ.params)(θvec)
    θ_priors = θ.priors
    if scheme isa LGCPGMScheme
        odegradmean = ode_gradmean_ℝ(X, θℝ, θ_priors, U⁺, C, ode)
    elseif scheme isa GPGMScheme
        odegradmean = ode_gradmean(X, θℝ, θ_priors, U⁺, C, ode)
    else
        error("scheme must be LGCPGMScheme or GPGMScheme")
    end
    gp_gradmean = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
    for (c, x) in enumerate(eachcol(X))
        D, m = Ds[c], ms[c]
        gp_gradmean[:, c] = D * (x .- m)
    end
    err_mat = gp_gradmean .- odegradmean
    errors = [err[:] for err in eachcol(err_mat)]
    return errors
end  # TEST

# ========================================================================
# 1st derivative of log probability
# ========================================================================
# --- dlp₍m⎸y₎ ---------------------------------------------------------
function dlp₍m⎸y₎dy(
    M::Matrix{Union{Nothing,Int}},
    Λ::Matrix{T1}
) where {T1<:Real}

    dlp = Vector{Union{Float64,ForwardDiff.Dual}}[]
    for (m, λ) in zip(eachcol(M), eachcol(Λ))
        push!(dlp, dlp₍m⎸y₎dy(m[:], λ[:]))
    end

    return vcat(dlp...)
end  # TEST

function dlp₍m⎸y₎dy(
    m::Vector{Union{Nothing,Int}},
    λ::Vector{T1}
) where {T1<:Real}

    dlp = Union{Float64,ForwardDiff.Dual}[]
    for (m_t, λ_t) in zip(m, λ)
        if ~isnothing(m_t)
            gradlp_m┆y_λ = gradlogpmf_λ(Poisson(λ_t), m_t)
        else
            gradlp_m┆y_λ = 0
        end
        grad_λ_y = λ_t
        g = gradlp_m┆y_λ * grad_λ_y  # m - λ
        push!(dlp, g)
    end

    return dlp
end  # TEST


# --- dlp₍y⎸xσmϕ₎ --------------------------------------------------------
function dlp₍y⎸xσmϕ₎dy(
    Y::Matrix{T1},
    y_mean_vec::Vector{Vector{T2}},
    y_std_vec::Vector{Vector{T3}},
    T::Int,
    scheme::Union{LGCPGMScheme,GPGMScheme}
) where {T1<:Real,T2<:Real,T3<:Real}

    dlp = Vector{Union{Float64,ForwardDiff.Dual}}[]
    for (y, y_mean, y_std) in zip(eachcol(Y), y_mean_vec, y_std_vec)
        if scheme isa LGCPGMScheme
            push!(dlp, dlp₍y⎸xσmϕ₎dy(y[:], y_mean, y_std))
        elseif scheme isa GPGMScheme
            push!(dlp, dlp₍y⎸xσmϕ₎dy(y[T+1:end], y_mean[T+1:end], y_std[T+1:end]))
        else
            error("scheme must be LGCPGMScheme or GPGMScheme")
        end
    end

    return vcat(dlp...)
end  # TEST

function dlp₍y⎸xσmϕ₎dy(
    y::Vector{T1},
    y_mean::Vector{T2},
    y_std::Vector{T3}
) where {T1<:Real,T2<:Real,T3<:Real}

    dlp = Union{Float64,ForwardDiff.Dual}[]
    for (_μ, _std, _y) in zip(y_mean, y_std, y)
        push!(dlp, Distributions.gradlogpdf(Normal(_μ, _std), _y))
    end

    return dlp
end  # TEST

function dlp₍y⎸xσmϕ₎dx(
    Y::Matrix{T1},
    y_mean_vec::Vector{Vector{T2}},
    y_std_vec::Vector{Vector{T3}},
    Luus::Vector{<:LowerTriangular},
    KtuKuu⁻¹s::Vector{<:AbstractMatrix}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}

    dlp = Vector{Union{Float64,ForwardDiff.Dual}}[]
    for (y, y_mean, y_std, Luu, KtuKuu⁻¹) in zip(eachcol(Y), y_mean_vec, y_std_vec, Luus, KtuKuu⁻¹s)
        push!(dlp, dlp₍y⎸xσmϕ₎dx(y[:], y_mean, y_std, Luu, KtuKuu⁻¹))
    end

    return vcat(dlp...)
end  # TEST

function dlp₍y⎸xσmϕ₎dx(
    y::Vector{T1},
    y_mean::Vector{T2},
    y_std::Vector{T3},
    Luu::LowerTriangular{T4,Matrix{T4}},
    KtuKuu⁻¹::Matrix{T5}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real,T5<:Real}

    dlp_y⎸xσmϕ_m = -1 * Union{Float64,ForwardDiff.Dual}[
        Distributions.gradlogpdf(Normal(_μ, _std), _y)
        for (_μ, _std, _y) in zip(y_mean, y_std, y)
    ]
    dlp = Luu' * (KtuKuu⁻¹' * dlp_y⎸xσmϕ_m)
    return dlp
end  # TEST

function dlp₍y⎸xσmϕ₎dσ(
    Y::Matrix{T1},
    σ::Vector{T2},
    y_mean_vec::Vector{Vector{T3}},
    y_mean_std::Vector{Vector{T4}},
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    C = size(Y, 2)
    dlp = []
    for c in 1:C
        y = Y[:, c]
        push!(dlp, dlp₍y⎸xσmϕ₎dσ(y, σ[c], y_mean_vec[c], y_mean_std[c]))
    end
    return dlp
end  # TEST

function dlp₍y⎸xσmϕ₎dσ(
    y::Vector{T1}, σ::T2, y_mean::Vector{T3}, y_std::Vector{T4}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}

    dlp_y⎸xσmϕ_predvar = sum([gradlogpdf_σ²(Normal(Ey, std), _y) for (Ey, std, _y) in zip(y_mean, y_std, y)])
    dpredvar_dσ = 2 * σ
    dσ_dlogσ = σ
    dlp_y⎸xσmϕ_dlogσ = dlp_y⎸xσmϕ_predvar * dpredvar_dσ * dσ_dlogσ
    return dlp_y⎸xσmϕ_dlogσ
end  # TEST

function dlp₍y⎸xσmϕ₎dϕ(
    Y::Matrix{T1},
    ϕvec::Vector{T2},
    gpcache::Vector{SparseGPCache{T3}},
    y_mean_vec::Vector{Vector{T4}},
    y_std_vec::Vector{Vector{T5}},
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real,T5<:Real}

    C = size(Y, 2)
    ϕmat = reshape(ϕvec, 3, C)
    _dlp₍y⎸xσmϕ₎dϕ = []
    for c in 1:C
        y = Y[:, c]
        base_kernel = gpcache[c].ϕ.base_kernel
        @unpack u, t = gpcache[c]
        a, l, w = ϕmat[:, c]
        y_mean = y_mean_vec[c]
        y_std = y_std_vec[c]
        diag_∂K̂∂loga = ∂K̂∂a(a, l, w, t, u, base_kernel) .* a |> diag
        diag_∂K̂∂logl = ∂K̂∂l(a, l, w, t, u, base_kernel) .* l |> diag
        diag_∂K̂∂logw = ∂K̂∂w(a, l, w, t, u, base_kernel) .* w |> diag
        push!(_dlp₍y⎸xσmϕ₎dϕ,
            dlp₍y⎸xσmϕ₎dϕ(y, y_mean, y_std, diag_∂K̂∂loga, diag_∂K̂∂logl, diag_∂K̂∂logw)
        )
    end
    return vcat(_dlp₍y⎸xσmϕ₎dϕ...)
end  # TEST

function dlp₍y⎸xσmϕ₎dϕ(
    y::Vector{T1}, y_mean::Vector{T2}, y_std::Vector{T3},
    diag_∂K̂∂loga::Vector{T4}, diag_∂K̂∂logl::Vector{T4}, diag_∂K̂∂logw::Vector{T4}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    dlp_y⎸xσmϕ_predvar = [gradlogpdf_σ²(Normal(Ey, std), _y)
                          for (Ey, std, _y) in zip(y_mean, y_std, y)]
    dlp_y⎸xσmϕ_dloga = sum(dlp_y⎸xσmϕ_predvar .* diag_∂K̂∂loga)
    dlp_y⎸xσmϕ_dlogl = sum(dlp_y⎸xσmϕ_predvar .* diag_∂K̂∂logl)
    dlp_y⎸xσmϕ_dlogw = sum(dlp_y⎸xσmϕ_predvar .* diag_∂K̂∂logw)
    return [dlp_y⎸xσmϕ_dloga, dlp_y⎸xσmϕ_dlogl, dlp_y⎸xσmϕ_dlogw]
end  # TEST

gradlogpdf_σ²(d::Normal, x::Real) = -0.5 / d.σ^2 + (x - d.μ)^2 / (2 * d.σ^4)  # TEST

# --- dlp₍x⎸mϕ₎ --------------------------------------------------------
function dlp₍x⎸mϕ₎dx(
    X::Matrix{T1},
    finite_gps::Vector{<:AbstractMvNormal}
) where {T1<:Real}

    dlp = Vector{Union{Float64,ForwardDiff.Dual}}[]
    for (x, finite_gp) in zip(eachcol(X), finite_gps)
        push!(dlp, Distributions.gradlogpdf(finite_gp, x))
    end
    return vcat(dlp...)
end  # TEST

function dlp₍x⎸mϕ₎dϕ(
    X::Matrix{T1},
    ϕvec::Vector{T2},
    finite_gps::Vector{<:AbstractMvNormal},
    gpcache::Vector{SparseGPCache{T3}}
) where {T1<:Real,T2<:Real,T3<:Real}

    C = size(X, 2)
    ϕmat = reshape(ϕvec, 3, C)
    _dlp₍x⎸mϕ₎dϕ = []
    for c in 1:C
        x = X[:, c]
        base_kernel = gpcache[c].ϕ.base_kernel
        @unpack u = gpcache[c]
        a, l, w = ϕmat[:, c]
        finite_gp = finite_gps[c]
        dlp₍x⎸mϕ₎dΣ = gradlogpdf_Σ(finite_gp, x)
        ∂K∂loga = ∂K∂a(a, l, u, u, base_kernel) .* a
        ∂K∂logl = ∂K∂l(a, l, u, u, base_kernel) .* l
        ∂K∂logw = ∂K∂w(w, u, u, base_kernel) .* w
        push!(
            _dlp₍x⎸mϕ₎dϕ,
            dlp₍x⎸mϕ₎dϕ(dlp₍x⎸mϕ₎dΣ, ∂K∂loga, ∂K∂logl, ∂K∂logw)
        )
    end

    return vcat(_dlp₍x⎸mϕ₎dϕ...)
end  # TEST

function dlp₍x⎸mϕ₎dϕ(
    dlp₍x⎸mϕ₎dΣ::Matrix{T1}, ∂K∂loga::Matrix{T2}, ∂K∂logl::Matrix{T2}, ∂K∂logw::Matrix{T2}
) where {T1<:Real,T2<:Real}
    dlp₍x⎸mϕ₎dloga = dlp₍x⎸mϕ₎dΣ .* ∂K∂loga |> sum
    dlp₍x⎸mϕ₎dlogl = dlp₍x⎸mϕ₎dΣ .* ∂K∂logl |> sum
    dlp₍x⎸mϕ₎dlogw = dlp₍x⎸mϕ₎dΣ .* ∂K∂logw |> sum
    return [dlp₍x⎸mϕ₎dloga, dlp₍x⎸mϕ₎dlogl, dlp₍x⎸mϕ₎dlogw]
end  # TEST

function gradlogpdf_Σ(d::Distributions.MvNormal, x::Vector{T}) where {T<:Real}
    Σ = d.Σ
    L⁻¹ = inv(Σ.chol.L)
    Σinv = L⁻¹' * L⁻¹
    Σinv_diff = Σinv * (x - d.μ)
    return -0.5 * (Σinv - Σinv_diff * Σinv_diff')
end  # TEST

# --- dlp₍θ₎ ------------------------------------------------------------
function dlp₍θ₎dθ(
    θvec::Vector{T1},
    θ_normal_priors::Vector{Normal{T2}}
) where {T1<:Real,T2<:Real}

    dlp = Union{Float64,ForwardDiff.Dual}[Distributions.gradlogpdf(normal_prior, _θ) for (_θ, normal_prior) in zip(θvec, θ_normal_priors)]
    return dlp
end  # TEST

# --- dlp₍x⎸mϕθγ₎ --------------------------------------------------------
"""
`dlp₍x⎸mϕθγ₎dx`
"""
function dlp₍x⎸mϕθγ₎dx(
    X::Matrix{T1},
    ∂Lg∂Gode::Matrix{T2},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors,
    Luus::Vector{<:LowerTriangular},
    Ds::Vector{<:AbstractMatrix},
    scheme::Union{LGCPGMScheme,GPGMScheme}
) where {T1<:Real,T2<:Real}

    _∂Lg∂Ggp = -∂Lg∂Gode
    ∂Lg∂x_gp = vcat([vec(sum(g .* Ds[c], dims=1)) for (c, g) in enumerate(eachcol(_∂Lg∂Ggp))]...)

    if scheme isa LGCPGMScheme
        Λ = get_Λ(X)
        θ₊ = get_θ₊(θ, θ_priors)
        _∂Gode∂λ = ∂G1∂λ(Λ, θ₊)  # NOTE: needs to be implemented for each ODE model
        _∂λ∂x = vec(Λ)
        ∂Lg∂x_ode = [(vec(∂Lg∂Gode)' * vec(g1)) * g2 for (g1, g2) in zip(_∂Gode∂λ, _∂λ∂x)]
        # _∂Gode∂x = ∂G1∂x(X, θ, θ_priors)
        # ∂Lg∂x_ode = [vec(∂Lg∂Gode)' * vec(g) for g in _∂Gode∂x]  # vecter inner product = sum of Hadamard product
    elseif scheme isa GPGMScheme
        _∂Gode∂x = ∂F1∂x(X, θ, θ_priors)
        ∂Lg∂x_ode = [vec(∂Lg∂Gode)' * vec(g) for g in _∂Gode∂x]
    else
        error("scheme must be one of LGCPGMScheme, GPGMScheme")
    end

    ∂Lg∂x = reshape(∂Lg∂x_gp + ∂Lg∂x_ode, size(X)...)
    ∂Lg∂x = vcat([Luus[c]' * v for (c, v) in enumerate(eachcol(∂Lg∂x))]...)

    return ∂Lg∂x
end  # TEST

"""
γ weighting version of `dlp₍x⎸mϕθγ₎dx`
"""
function dlp₍x⎸mϕθγ₎dx(
    X::Matrix{T1},
    ∂Lg∂Gode::Matrix{T2},
    err_dists::Vector{<:AbstractMvNormal},
    errors::Vector{Vector{T3}},
    γ::T4,
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors,
    Luus::Vector{<:LowerTriangular},
    Ds::Vector{<:AbstractMatrix},
    scheme::Union{LGCPGMScheme,GPGMScheme},
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}

    _∂Lg∂Ggp = -∂Lg∂Gode
    ∂Lg∂x_gp = vcat([vec(sum(g .* Ds[c], dims=1)) for (c, g) in enumerate(eachcol(_∂Lg∂Ggp))]...)

    if scheme isa LGCPGMScheme
        Λ = get_Λ(X)
        θ₊ = get_θ₊(θ, θ_priors)
        _∂Gode∂λ = ∂G1∂λ(Λ, θ₊)  # NOTE: needs to be implemented for each ODE model
        _∂λ∂x = vec(Λ)
        ∂Lg∂x_ode = [(vec(∂Lg∂Gode)' * vec(g1)) * g2 for (g1, g2) in zip(_∂Gode∂λ, _∂λ∂x)]
        # _∂Gode∂x = ∂G1∂x(X, θ, θ_priors)
        # ∂Lg∂x_ode = [vec(∂Lg∂Gode)' * vec(g) for g in _∂Gode∂x]
    elseif scheme isa GPGMScheme
        _∂Gode∂x = ∂F1∂x(X, θ, θ_priors)
        ∂Lg∂x_ode = [vec(∂Lg∂Gode)' * vec(g) for g in _∂Gode∂x]
        else
        error("scheme must be one of LGCPGMScheme, GPGMScheme")
    end

    ∂Lg∂x_Aγ = []
    for (err_dist, err, x) in zip(err_dists, errors, eachcol(X))
        ∂Lg∂Aγ = gradlogpdf_Σ(err_dist, err[:])
        ∂Aγ∂x = - 2 * (γ ./ exp.(x)).^2
        # ∂Aγ∂γexp₍x₎⁻¹ = 2 * γ ./ exp.(x)
        # ∂γexp₍x₎⁻¹∂exp₍x₎ = - γ ./ (exp.(x)).^2
        # ∂exp₍x₎∂x = exp.(x)
        append!(∂Lg∂x_Aγ, diag(∂Lg∂Aγ) .* ∂Aγ∂x)
    end

    ∂Lg∂x = reshape(∂Lg∂x_gp + ∂Lg∂x_ode + ∂Lg∂x_Aγ, size(X)...)
    ∂Lg∂x = vcat([Luus[c]' * v for (c, v) in enumerate(eachcol(∂Lg∂x))]...)

    return ∂Lg∂x
end  # TEST

function dlp₍x⎸mϕθγ₎dθ(
    X::Matrix{T},
    ∂Lg∂Gode::Matrix{T2},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors,
    scheme::Union{LGCPGMScheme,GPGMScheme}
) where {T<:Real,T2<:Real}

    if scheme isa LGCPGMScheme
        _∂Gode∂θ = ∂G1∂θ(X, θ, θ_priors)  # [T × C/M] * length(θ.params)
    elseif scheme isa GPGMScheme
        _∂Gode∂θ = ∂F1∂θ(X, θ, θ_priors)  # [T × C/M] * length(θ.params)
    else
        error("scheme must be one of LGCPGMScheme, GPGMScheme")
    end
    ∂Lg∂θ = [sum(∂Lg∂Gode .* g) for g in _∂Gode∂θ]

    return ∂Lg∂θ
end  # TEST

function dlp₍x⎸mϕθγ₎dϕ(
    X::Matrix{T1},
    ϕvec::Vector{T2},
    grad_errors::Vector{Vector{T3}},
    grad_errors_dists::Vector{<:AbstractMvNormal},
    Kuu′s::Vector{<:AbstractMatrix},
    Kuu⁻¹s::Vector{<:AbstractMatrix},
    gpcache::Vector{SparseGPCache{T4}}
) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}

    C = size(X, 2)
    ϕmat = reshape(ϕvec, 3, C)
    _dlp₍x⎸mϕθγ₎dϕ = []
    for c in 1:C
        base_kernel = gpcache[c].ϕ.base_kernel
        @unpack u, should_diagonalize_A = gpcache[c]
        a, l, w = ϕmat[:, c]

        err = grad_errors[c]
        err_dist = grad_errors_dists[c]

        Kuu′, Kuu⁻¹ = Kuu′s[c], Kuu⁻¹s[c]

        ∂Lg∂Dx = -Distributions.gradlogpdf(err_dist, err)
        ∂Dx∂D = X[:, c]
        ∂Lg∂D = ∂Lg∂Dx * ∂Dx∂D'
        ∂Lg∂Aγ = gradlogpdf_Σ(err_dist, err[:])

        ∂D∂loga = dD_da(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel) .* a
        ∂D∂logl = dD_dl(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel) .* l
        ∂D∂logw = dD_dw(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel) .* w

        ∂A∂loga = dA_da(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel; should_diagonalize_A=should_diagonalize_A) .* a
        ∂A∂logl = dA_dl(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel; should_diagonalize_A=should_diagonalize_A) .* l
        ∂A∂logw = dA_dw(a, l, w, u, u, Kuu′, Kuu⁻¹, base_kernel; should_diagonalize_A=should_diagonalize_A) .* w

        push!(
            _dlp₍x⎸mϕθγ₎dϕ,
            dlp₍x⎸mϕθγ₎dϕ(∂Lg∂D, ∂D∂loga, ∂D∂logl, ∂D∂logw, ∂Lg∂Aγ, ∂A∂loga, ∂A∂logl, ∂A∂logw)
        )
    end
    return vcat(_dlp₍x⎸mϕθγ₎dϕ...)
end  # TEST

function dlp₍x⎸mϕθγ₎dϕ(
    ∂Lg∂D::Matrix{T1},
    ∂D∂loga::Matrix{T1},
    ∂D∂logl::Matrix{T1},
    ∂D∂logw::Matrix{T1},
    ∂Lg∂Aγ::Matrix{T1},
    ∂A∂loga::Matrix{T1},
    ∂A∂logl::Matrix{T1},
    ∂A∂logw::Matrix{T1}
) where {T1<:Real}

    ∂Lg∂loga = sum(∂Lg∂Aγ .* ∂A∂loga) + sum(∂Lg∂D .* ∂D∂loga)
    ∂Lg∂logl = sum(∂Lg∂Aγ .* ∂A∂logl) + sum(∂Lg∂D .* ∂D∂logl)
    ∂Lg∂logw = sum(∂Lg∂Aγ .* ∂A∂logw) + sum(∂Lg∂D .* ∂D∂logw)
    return [∂Lg∂loga, ∂Lg∂logl, ∂Lg∂logw]
end  # TEST

# ========================================================================
# Others
# ========================================================================
function ode_gradmean_ℝ(
    X::Matrix{T},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors,
    U⁺::Int,
    C::Int,
    ode::Function
) where {T<:Real}

    @assert size(X) == (U⁺, C)

    Λ = get_Λ(X)  # NOTE: U × C shaped
    θ₊ = get_θ₊(θ, θ_priors)
    F1 = ode(Λ, θ₊)  # NOTE: needs to be implemented for each ODE model
    @assert all(isfinite.(F1)) "non-finite values exist"
    G1 = F1 ./ Λ
    return G1
end

function ∂Lg∂Ggp(
    err_mat::Matrix{T},
    gradient_error_dist::MvNormal
) where {T<:Real}
    _∂Lg∂Ggp = []
    for err in eachcol(err_mat)
        push!(_∂Lg∂Ggp, Distributions.gradlogpdf(gradient_error_dist, err))
    end
    return hcat(_∂Lg∂Ggp...)
end

function ∂G1∂θ(
    X::Matrix{T},
    θ::AbstractODEParamsFullSupport,
    θ_priors::AbstractODEParamPriors
) where {T<:Real}

    Λ = get_Λ(X)
    θ₊ = get_θ₊(θ, θ_priors)
    grads = ∂G1∂θ(Λ, θ₊, θ, θ_priors)  # NOTE: needs to be implemented for each ODE model
    return grads  # [Nx × C] * length(θ)
end

function ∂θ∂θℝ(θ_ℝ::Union{T,Vector{T}}, prior::ScaledLogitNormal) where {T<:Real}
    return grad_scaled_logistic_x.(θ_ℝ, prior.min, prior.max; rate=prior.rate, mid=prior.mid)
end

function grad_scaled_logistic_x(
    x::Union{Float64,ForwardDiff.Dual}, min::Float64=0.0, max::Float64=1.0;
    rate::Float64=1.0, mid::Float64=0.0
)

    f = scaled_logistic(x, 0.0, 1.0, rate=rate, mid=mid)
    return rate * (max - min) * f .* (1.0 - f)
end
