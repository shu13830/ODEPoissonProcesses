"""
λ₁=exp(x₁): population of prey
λ₂=exp(x₂): population of predator

Lotka-Volterra Predator-Prey equation mapped to real space
    dx₁/dt = a - b λ₂
    dx₂/dt = - c + d λ₁

derivative of Lotka-Volterra Predator-Prey equation mapped to real space
    ∇λ₁(dx₁/dt) = 0,    ∇λ₂(dx₁/dt) = -b
    ∇λ₁(dx₂/dt) = d,  ∇λ₂(dx₂/dt) = 0

"""
function ∂G1∂λ(
    Λ::Matrix{T}, θ₊::PredatorPreyParams
) where {T<:Real}

    U, C = size(Λ)
    @unpack a, b, c, d = θ₊
    grads = []
    for c in 1:C, t in 1:U
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, U, C)
        grad .= 0.0
        if c == 1
            grad[t, 1] = 0.0    # ∇λ₁(dλ₁/dt) = 0
            grad[t, 2] = d    # ∇λ₁(dλ₂/dt) = d
        elseif c == 2
            grad[t, 1] = -b  # ∇λ₂(dλ₁/dt) = - b
            grad[t, 2] = 0.0    # ∇λ₂(dλ₂/dt) = 0
        else
            error("Invalid index of class. class must be one of (1, 2)")
        end
        push!(grads, grad)
    end
    return grads
end

"""
λ₁: population of prey
λ₂: population of predator

Lotka-Volterra Predator-Prey equation
    dλ₁/dt = a λ₁ - b λ₁ λ₂
    dλ₂/dt = -c λ₂ + d λ₁ λ₂

derivative of Lotka-Volterra Predator-Prey equation
    ∇λ₁(dλ₁/dt) = a - b λ₂,  ∇λ₂(dλ₁/dt) = - b λ₁
    ∇λ₁(dλ₂/dt) = d λ₂,     ∇λ₂(dλ₂/dt) = - c + d λ₁
"""
function ∂F1∂x(
    X::Matrix{T}, θ₊::PredatorPreyParams
) where {T<:Real}

    U⁺, C = size(X)
    @unpack a, b, c, d = θ₊
    grads = []
    for c in 1:C, t in 1:U⁺
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, U⁺, C)
        grad .= 0.0
        if c == 1
            grad[t, 1] = a .- b .* max(X[t, 2], 0.0)    # ∇λ₁(dλ₁/dt) = a - b λ₂
            grad[t, 2] = d .* max(X[t, 2], 0.0)         # ∇λ₁(dλ₂/dt) = d λ₂
        elseif c == 2
            grad[t, 1] = -b .* max(X[t, 1], 0.0)       # ∇λ₂(dλ₁/dt) = - b λ₁
            grad[t, 2] = -c .+ d .* max(X[t, 1], 0.0)  # ∇λ₂(dλ₂/dt) = -c + d λ₁
        else
            error("Invalid index of class. class must be one of (1, 2)")
        end
        push!(grads, grad)
    end
    return grads
end

"""
Lotka-Volterra Predator-Prey equation
    dλ₁/dt = a λ₁ - b λ₁ λ₂
    dλ₂/dt = -c λ₂ + d λ₁ λ₂

Lotka-Volterra Predator-Prey equation mapped to real space
    dx₁/dt = a - b λ₂
    dx₂/dt = - c + d λ₁

derivative of Lotka-Volterra Predator-Prey equation
    ∇a(dλ₁/dt) = λ₁,  ∇b(dλ₁/dt) = - λ₁ λ₂,  ∇c(dλ₁/dt) = 0,     ∇d(dλ₁/dt) = 0
    ∇a(dλ₂/dt) = 0,   ∇b(dλ₂/dt) = 0,        ∇c(dλ₂/dt) = - λ₂,  ∇d(dλ₂/dt) = λ₁ λ₂

derivative of Lotka-Volterra Predator-Prey equation mapped to real space
    ∇a(dλ₁/dt) = 1,  ∇b(dλ₁/dt) = - λ₂,  ∇c(dλ₁/dt) = 0,    ∇d(dλ₁/dt) = 0
    ∇a(dλ₂/dt) = 0,  ∇b(dλ₂/dt) = 0,     ∇c(dλ₂/dt) = - 1,  ∇d(dλ₂/dt) = λ₁

"""
function ∂G1∂θ(
    Λ::Matrix{T},
    θ₊::PredatorPreyParams,
    θ::PredatorPreyParamsFullSupport,
    θ_priors::PredatorPreyParamPriors
) where {T<:Real}

    function _∂G1∂aℝ(Λ::Matrix{T}, θ₊::PredatorPreyParams, ∂a∂aℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂a(Λ::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b, c, d = θ₊
            grads[:, 1] .= 1    # ∇a(dλ₁/dt) = 1,
            # grads[:, 2] .= 0  # ∇a(dλ₂/dt) = 0
            return grads
        end

        ∂G1∂a = _∂G1∂a(Λ, θ₊)  # [T × C]
        ∂G1∂aℝ = ∂G1∂a .* ∂a∂aℝ   # [T × C]
        return ∂G1∂aℝ
    end

    function _∂G1∂bℝ(Λ::Matrix{T}, θ₊::PredatorPreyParams, ∂b∂bℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂b(Λ::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b, c, d = θ₊
            grads[:, 1] = -Λ[:, 2]  # ∇b(dλ₁/dt) = - λ₂
            # grads[:, 2] .= 0       # ∇b(dλ₂/dt) = 0
            return grads
        end

        ∂G1∂b = _∂G1∂b(Λ, θ₊)  # [T × C]
        ∂G1∂bℝ = ∂G1∂b .* ∂b∂bℝ           # [T × C]
        return ∂G1∂bℝ
    end

    function _∂G1∂cℝ(Λ::Matrix{T}, θ₊::PredatorPreyParams, ∂c∂cℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂c(Λ::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b, c, d = θ₊
            # grads[:, 1] .= 0  # ∇c(dλ₁/dt) = 0
            grads[:, 2] .= -1  # ∇c(dλ₂/dt) = - 1
            return grads
        end

        ∂G1∂c = _∂G1∂c(Λ, θ₊)  # [T × C]
        ∂G1∂cℝ = ∂G1∂c .* ∂c∂cℝ           # [T × C]
        return ∂G1∂cℝ
    end

    function _∂G1∂dℝ(Λ::Matrix{T}, θ₊::PredatorPreyParams, ∂d∂dℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂d(Λ::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b, c, d = θ₊
            # grads[:, 1] .= 0     # ∇d(dλ₁/dt) = 0
            grads[:, 2] = Λ[:, 1]  # ∇d(dλ₂/dt) = λ₁
            return grads
        end

        ∂G1∂d = _∂G1∂d(Λ, θ₊)  # [T × C]
        ∂G1∂dℝ = ∂G1∂d .* ∂d∂dℝ           # [T × C]
        return ∂G1∂dℝ
    end

    @unpack a_ℝ, b_ℝ, c_ℝ, d_ℝ = θ
    @unpack a_prior, b_prior, c_prior, d_prior = θ_priors
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)
    ∂b∂bℝ = ∂θ∂θℝ(b_ℝ, b_prior)
    ∂c∂cℝ = ∂θ∂θℝ(c_ℝ, c_prior)
    ∂d∂dℝ = ∂θ∂θℝ(d_ℝ, d_prior)

    ∂G1∂aℝ = _∂G1∂aℝ(Λ, θ₊, ∂a∂aℝ)
    ∂G1∂bℝ = _∂G1∂bℝ(Λ, θ₊, ∂b∂bℝ)
    ∂G1∂cℝ = _∂G1∂cℝ(Λ, θ₊, ∂c∂cℝ)
    ∂G1∂dℝ = _∂G1∂dℝ(Λ, θ₊, ∂d∂dℝ)

    return [∂G1∂aℝ, ∂G1∂bℝ, ∂G1∂cℝ, ∂G1∂dℝ]  # Vector of T × C shaped matrices
end



function ∂F1∂θ(
    X::Matrix{T},
    θ₊::PredatorPreyParams,
    θ::PredatorPreyParamsFullSupport,
    θ_priors::PredatorPreyParamPriors
) where {T<:Real}

    function _∂F1∂aℝ(X::Matrix{T}, θ₊::PredatorPreyParams, ∂a∂aℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂a(X::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b, c, d = θ₊
            grads[:, 1] = max.(X[:, 1], 0.0)  # ∇a(dλ₁/dt) = λ₁,
            grads[:, 2] .= 0.0                # ∇a(dλ₂/dt) = 0
            return grads
        end

        ∂F1∂a = _∂F1∂a(X, θ₊)  # [T × C]
        ∂F1∂aℝ = ∂F1∂a .* ∂a∂aℝ           # [T × C]
        return ∂F1∂aℝ
    end

    function _∂F1∂bℝ(X::Matrix{T}, θ₊::PredatorPreyParams, ∂b∂bℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂b(X::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b, c, d = θ₊
            grads[:, 1] = -max.(X[:, 1], 0.0) .* max.(X[:, 2], 0.0)  # ∇b(dλ₁/dt) = - λ₁ λ₂
            # grads[:, 2] .= 0                                    # ∇b(dλ₂/dt) = 0
            return grads
        end

        ∂F1∂b = _∂F1∂b(X, θ₊)  # [T × C]
        ∂F1∂bℝ = ∂F1∂b .* ∂b∂bℝ           # [T × C]
        return ∂F1∂bℝ
    end

    function _∂F1∂cℝ(X::Matrix{T}, θ₊::PredatorPreyParams, ∂c∂cℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂c(X::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b, c, d = θ₊
            # grads[:, 1] .= 0                 # ∇c(dλ₁/dt) = 0
            grads[:, 2] = -max.(X[:, 2], 0.0)  # ∇c(dλ₂/dt) = - λ₂
            return grads
        end

        ∂F1∂c = _∂F1∂c(X, θ₊)  # [T × C]
        ∂F1∂cℝ = ∂F1∂c .* ∂c∂cℝ           # [T × C]
        return ∂F1∂cℝ
    end

    function _∂F1∂dℝ(X::Matrix{T}, θ₊::PredatorPreyParams, ∂d∂dℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂d(X::Matrix{T}, θ₊::PredatorPreyParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b, c, d = θ₊
            # grads[:, 1] .= 0                                  # ∇d(dλ₁/dt) = 0
            grads[:, 2] = max.(X[:, 1], 0.0) .* max.(X[:, 2], 0.0)  # ∇d(dλ₂/dt) = λ₁ λ₂
            return grads
        end

        ∂F1∂d = _∂F1∂d(X, θ₊)  # [T × C]
        ∂F1∂dℝ = ∂F1∂d .* ∂d∂dℝ   # [T × C]
        return ∂F1∂dℝ
    end

    @unpack a_ℝ, b_ℝ, c_ℝ, d_ℝ = θ
    @unpack a_prior, b_prior, c_prior, d_prior = θ_priors
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)
    ∂b∂bℝ = ∂θ∂θℝ(b_ℝ, b_prior)
    ∂c∂cℝ = ∂θ∂θℝ(c_ℝ, c_prior)
    ∂d∂dℝ = ∂θ∂θℝ(d_ℝ, d_prior)

    ∂F1∂aℝ = _∂F1∂aℝ(X, θ₊, ∂a∂aℝ)
    ∂F1∂bℝ = _∂F1∂bℝ(X, θ₊, ∂b∂bℝ)
    ∂F1∂cℝ = _∂F1∂cℝ(X, θ₊, ∂c∂cℝ)
    ∂F1∂dℝ = _∂F1∂dℝ(X, θ₊, ∂d∂dℝ)

    return [∂F1∂aℝ, ∂F1∂bℝ, ∂F1∂cℝ, ∂F1∂dℝ]  # Vector of T × C shaped matrices
end  # NOTE: not tested
