function ∂G1∂λ(Λ::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
    U, C = size(Λ)
    @unpack r, η, a = θ₊
    C = length(r)
    A = competitive_coef_matrix(a, C)
    grads = []
    for c_λ in 1:C, t_λ in 1:U
        # grad = zeros(T, U, C)
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(Λ))
        grad .= 0.0
        for c_g in 1:C
            grad[t_λ, c_g] = -r[c_g] * A[c_g, c_λ] ./ η[c_g]
        end
        push!(grads, grad)
    end
    return grads
end


function ∂F1∂x(
    X::Matrix{T}, θ₊::CompetitionParams
) where {T<:Real}

    U⁺, C = size(X)
    @unpack r, η, a = θ₊
    C = length(r)
    A = competitive_coef_matrix(a, C)
    grads = []
    for c_x in 1:C, t_x in 1:U⁺
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
        grad .= 0.0
        for c_g in 1:C
            if c_g == c_x
                grad[t_x, c_g] = r[c_g] .* (1 .- (X[t_x, c_g] .+ X[t_x, :]' * A[c_g, :]) ./ η[c_g])
            else
                grad[t_x, c_g] = -r[c_g] * X[t_x, c_g] * A[c_g, c_x] / η[c_g]
            end
        end
        push!(grads, grad)
    end
    return grads
end


function ∂G1∂θ(
    Λ::Matrix{T},
    θ₊::CompetitionParams,
    θ::CompetitionParamsFullSupport,
    θ_priors::CompetitionParamPriors
) where {T<:Real}

    function _∂G1∂rℝ(Λ::Matrix{T}, θ₊::CompetitionParams, ∂r∂rℝ::Vector{T2}) where {T<:Real,T2<:Real}

        function _∂G1∂r(Λ::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            A = competitive_coef_matrix(a, C)
            grads = []
            for c in 1:C
                v = 1 .- (Λ * A[c, :]) ./ η[c]
                grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(Λ))
                grad .= 0.0
                grad[:, c] .= v
                push!(grads, grad)
            end
            return grads
        end

        ∂G1∂r = _∂G1∂r(Λ, θ₊)  # [T × C] * C
        ∂G1∂rℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂G1∂r, ∂r∂rℝ)]  # [T × C] * C
        return ∂G1∂rℝ
    end

    function _∂G1∂ηℝ(Λ::Matrix{T}, θ₊::CompetitionParams, ∂η∂ηℝ::Vector{T2}) where {T<:Real,T2<:Real}

        function _∂G1∂η(Λ::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            A = competitive_coef_matrix(a, C)
            grads = []
            for c in 1:C
                v = r[c] .* (Λ * A[c, :]) ./ η[c]^2
                grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(Λ))
                grad .= 0.0
                grad[:, c] .= v
                push!(grads, grad)
            end
            return grads
        end

        ∂G1∂η = _∂G1∂η(Λ, θ₊)  # [T × C] * C
        ∂G1∂ηℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂G1∂η, ∂η∂ηℝ)]  # [T × C] * C
        return ∂G1∂ηℝ
    end

    function _∂G1∂aℝ(Λ::Matrix{T}, θ₊::CompetitionParams, ∂a∂aℝ::Vector{T2}) where {T<:Real,T2<:Real}

        function _∂G1∂a(Λ::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            grads = []
            for j in 1:C, i in 1:C
                if i != j
                    v = -r[i] .* Λ[:, j] ./ η[i]
                    grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(Λ))
                    grad .= 0.0
                    grad[:, i] .= v
                    push!(grads, grad)
                end
            end
            return grads
        end

        ∂G1∂a = _∂G1∂a(Λ, θ₊)  # [T × C] * C^2-C
        ∂G1∂aℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂G1∂a, ∂a∂aℝ)]  # [T × C] * C^2-C
        return ∂G1∂aℝ
    end

    @unpack r_ℝ, η_ℝ, a_ℝ = θ
    @unpack r_prior, η_prior, a_prior = θ_priors
    ∂r∂rℝ = ∂θ∂θℝ(r_ℝ, r_prior)  # C length
    ∂η∂ηℝ = ∂θ∂θℝ(η_ℝ, η_prior)  # C length
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)  # C^2-C length

    ∂G1∂rℝ = _∂G1∂rℝ(Λ, θ₊, ∂r∂rℝ)  # [T × C] * C
    ∂G1∂ηℝ = _∂G1∂ηℝ(Λ, θ₊, ∂η∂ηℝ)  # [T × C] * C
    ∂G1∂aℝ = _∂G1∂aℝ(Λ, θ₊, ∂a∂aℝ)  # [T × C] * C^2-C

    return vcat([∂G1∂rℝ, ∂G1∂ηℝ, ∂G1∂aℝ]...)  # [T × C] * (C + C + C^2-C)
end


function ∂F1∂θ(
    X::Matrix{T},
    θ₊::CompetitionParams,
    θ::CompetitionParamsFullSupport,
    θ_priors::CompetitionParamPriors
) where {T<:Real}

    function _∂F1∂rℝ(
        X::Matrix{T}, θ₊::CompetitionParams, ∂r∂rℝ::Vector{T2}
    ) where {T<:Real,T2<:Real}

        function _∂F1∂r(X::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            A = competitive_coef_matrix(a, C)
            grads = []
            for c in 1:C
                v = X[:, c] .* (1 .- (X * A[c, :] ./ η[c]))
                grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
                grad .= 0.0
                grad[:, c] .= v
                push!(grads, grad)
            end
            return grads
        end

        ∂F1∂r = _∂F1∂r(X, θ₊)  # [T × C] * C
        ∂F1∂rℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂F1∂r, ∂r∂rℝ)]  # [T × C] * C
        return ∂F1∂rℝ
    end

    function _∂F1∂ηℝ(
        X::Matrix{T}, θ₊::CompetitionParams, ∂η∂ηℝ::Vector{T2}
    ) where {T<:Real,T2<:Real}

        function _∂F1∂η(X::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            A = competitive_coef_matrix(a, C)
            grads = []
            for c in 1:C
                v = r[c] .* X[:, c] .* (X * A[c, :]) ./ η[c]^2
                grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
                grad .= 0.0
                grad[:, c] .= v
                push!(grads, grad)
            end
            return grads
        end

        ∂F1∂η = _∂F1∂η(X, θ₊)  # [T × C] * C
        ∂F1∂ηℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂F1∂η, ∂η∂ηℝ)]  # [T × C] * C
        return ∂F1∂ηℝ
    end

    function _∂F1∂aℝ(
        X::Matrix{T}, θ₊::CompetitionParams, ∂a∂aℝ::Vector{T2}
    ) where {T<:Real,T2<:Real}

        function _∂F1∂a(X::Matrix{T}, θ₊::CompetitionParams) where {T<:Real}
            @unpack r, η, a = θ₊
            C = length(r)
            grads = []
            for j in 1:C, i in 1:C
                if i != j
                    v = -r[i] .* X[:, i] .* X[:, j] ./ η[i]
                    grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
                    grad .= 0.0
                    grad[:, i] .= v
                    push!(grads, grad)
                end
            end
            return grads
        end

        ∂F1∂a = _∂F1∂a(X, θ₊)  # [T × C] * C^2-C
        ∂F1∂aℝ = Matrix[g1 .* g2 for (g1, g2) in zip(∂F1∂a, ∂a∂aℝ)]  # [T × C] * C^2-C
        return ∂F1∂aℝ
    end

    @unpack r_ℝ, η_ℝ, a_ℝ = θ
    @unpack r_prior, η_prior, a_prior = θ_priors
    ∂r∂rℝ = ∂θ∂θℝ(r_ℝ, r_prior)  # C length
    ∂η∂ηℝ = ∂θ∂θℝ(η_ℝ, η_prior)  # C length
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)  # C^2-C length

    ∂F1∂rℝ = _∂F1∂rℝ(X, θ₊, ∂r∂rℝ)  # [T × C] * C
    ∂F1∂ηℝ = _∂F1∂ηℝ(X, θ₊, ∂η∂ηℝ)  # [T × C] * C
    ∂F1∂aℝ = _∂F1∂aℝ(X, θ₊, ∂a∂aℝ)  # [T × C] * C^2-C

    return vcat([∂F1∂rℝ, ∂F1∂ηℝ, ∂F1∂aℝ]...)  # [T × C] * (C + C + C^2-C)
end
