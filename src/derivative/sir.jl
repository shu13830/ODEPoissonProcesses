"""
SIR equations
    dΛₛ/dt = - a Λₛ Λᵢ
    dΛᵢ/dt = a Λₛ Λᵢ - b Λᵢ
    dΛᵣ/dt = b Λᵢ

SIR equations mapped to real space
    dXₛ/dt = - a Λᵢ
    dXᵢ/dt = a Λₛ - b
    dXᵣ/dt = b Λᵢ / Λᵣ

derivative of SIR equations
    ∇Λₛ(dΛₛ/dt) = - a Λᵢ,  ∇Λᵢ(dΛₛ/dt) = - a Λₛ,    ∇Λᵣ(dΛₛ/dt) = 0
    ∇Λₛ(dΛᵢ/dt) = a Λᵢ,    ∇Λᵢ(dΛᵢ/dt) = a Λₛ - b,  ∇Λᵣ(dΛᵢ/dt) = 0
    ∇Λₛ(dΛᵣ/dt) = 0,       ∇Λᵢ(dΛᵣ/dt) = b,         ∇Λᵣ(dΛᵣ/dt) = 0

derivative of SIR equations mapped to real space
    ∇Λₛ(dXₛ/dt) = 0,  ∇Λᵢ(dXₛ/dt) = - a,     ∇Λᵣ(dXₛ/dt) = 0
    ∇Λₛ(dXᵢ/dt) = a,  ∇Λᵢ(dXᵢ/dt) = 0,       ∇Λᵣ(dXᵢ/dt) = 0
    ∇Λₛ(dXᵣ/dt) = 0,  ∇Λᵢ(dXᵣ/dt) = b / Λᵣ,  ∇Λᵣ(dXᵣ/dt) = - b Λᵢ / Λᵣ^2
"""
function ∂G1∂λ(Λ::Matrix{T}, θ₊::SIRParams) where {T<:Real}
    U, C = size(Λ)
    @assert C == 3
    @unpack a, b = θ₊
    grads = []
    for c in 1:C, t in 1:U
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(Λ))
        grad .= 0.0
        if c == 1  # ∇ₛ
            # grad[t, 1] = 0
            grad[t, 2] = a
            # grad[t, 3] = 0
        elseif c == 2  # ∇ᵢ
            grad[t, 1] = -a
            # grad[t, 2] = 0
            grad[t, 3] = b / Λ[t, 3]
        else  # c == 3  # ∇ᵣ
            # grad[t, 1] = 0
            # grad[t, 2] = 0
            grad[t, 3] = -b * Λ[t, 2] / Λ[t, 3]^2
        end
        push!(grads, grad)
    end
    return grads  # T × C shaped Matrix contains T × C shaped Matrices as elements
end

function ∂F1∂x(
    X::Matrix{T}, θ₊::SIRParams
) where {T<:Real}

    U⁺, C = size(X)
    @unpack a, b = θ₊
    grads = []
    for c in 1:C, t in 1:U⁺
        grad = Array{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
        grad .= 0.0
        if c == 1  # ∇ₛ
            grad[t, 1] = -a * max(X[t, 2], 0.0)
            grad[t, 2] = a * max(X[t, 2], 0.0)
            # grad[t, 3] = 0
        elseif c == 2  # ∇ᵢ
            grad[t, 1] = -a * max(X[t, 1], 0.0)
            grad[t, 2] = a * max(X[t, 1], 0.0) - b
            grad[t, 3] = b
        elseif c == 3  # ∇ᵣ
        # grad[t, 1] = 0
        # grad[t, 2] = 0
        # grad[t, 3] = 0
        else
            error("Invalid index of class. class must be one of (1, 2, 3)")
        end
        push!(grads, grad)
    end
    return grads  # T × C shaped Matrix contains T × C shaped Matrices as elements
end

"""
SIR equations
    dΛₛ/dt = - a Λₛ Λᵢ
    dΛᵢ/dt = a Λₛ Λᵢ - b Λᵢ
    dΛᵣ/dt = b Λᵢ

SIR equations mapped to real space
    dXₛ/dt = - a Λᵢ
    dXᵢ/dt = a Λₛ - b
    dXᵣ/dt = b Λᵢ / Λᵣ

derivative of SIR equations
    ∇a(dΛₛ/dt) = - Λₛ Λᵢ,  ∇b(dΛₛ/dt) = 0
    ∇a(dΛᵢ/dt) = Λₛ Λᵢ,    ∇b(dΛᵢ/dt) = - Λᵢ
    ∇a(dΛᵣ/dt) = 0,        ∇b(dΛᵣ/dt) = Λᵢ

derivative of SIR equations mapped to real space
    ∇a(dΛₛ/dt) = - Λᵢ,  ∇b(dΛₛ/dt) = 0
    ∇a(dΛᵢ/dt) = Λₛ,    ∇b(dΛᵢ/dt) = - 1
    ∇a(dΛᵣ/dt) = 0,     ∇b(dΛᵣ/dt) = Λᵢ / Λᵣ

"""
function ∂G1∂θ(Λ::Matrix{T}, θ₊::SIRParams, θ::SIRParamsFullSupport, θ_priors::SIRParamPriors) where {T<:Real}

    function _∂G1∂aℝ(Λ::Matrix{T}, θ₊::SIRParams, ∂a∂aℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂a(Λ::Matrix{T}, θ₊::SIRParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b = θ₊
            grads[:, 1] = -Λ[:, 2]  # ∇a(dΛₛ/dt) = - Λᵢ
            grads[:, 2] = Λ[:, 1]    # ∇a(dΛᵢ/dt) = Λₛ
            # grads[:, 3] .= 0       # ∇a(dΛᵣ/dt) = 0
            return grads
        end

        ∂G1∂a = _∂G1∂a(Λ, θ₊)  # [T × C]
        ∂G1∂aℝ = ∂G1∂a .* ∂a∂aℝ      # [T × C]
        return ∂G1∂aℝ
    end

    function _∂G1∂bℝ(Λ::Matrix{T}, θ₊::SIRParams, ∂b∂bℝ::T2) where {T<:Real,T2<:Real}

        function _∂G1∂b(Λ::Matrix{T}, θ₊::SIRParams) where {T<:Real}
            grads = zeros(T, size(Λ))
            # @unpack a, b = θ₊
            # grads[:, 1] .= 0                # ∇b(dΛₛ/dt) = 0
            grads[:, 2] .= -1                # ∇b(dΛᵢ/dt) = - 1
            grads[:, 3] = Λ[:, 2] ./ Λ[:, 3]  # ∇b(dΛᵣ/dt) =  Λᵢ / Λᵣ
            return grads
        end

        ∂G1∂b = _∂G1∂b(Λ, θ₊)  # [T × C]
        ∂G1∂bℝ = ∂G1∂b .* ∂b∂bℝ      # [T × C]
        return ∂G1∂bℝ
    end

    @unpack a_ℝ, b_ℝ = θ
    @unpack a_prior, b_prior = θ_priors
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)  # scalar
    ∂b∂bℝ = ∂θ∂θℝ(b_ℝ, b_prior)  # scalar

    ∂G1∂aℝ = _∂G1∂aℝ(Λ, θ₊, ∂a∂aℝ)  # [T × C]
    ∂G1∂bℝ = _∂G1∂bℝ(Λ, θ₊, ∂b∂bℝ)  # [T × C]

    return [∂G1∂aℝ, ∂G1∂bℝ]  # [T × C] * 2
end

function ∂F1∂θ(
    X::Matrix{T},
    θ₊::SIRParams,
    θ::SIRParamsFullSupport,
    θ_priors::SIRParamPriors
) where {T<:Real}

    function _∂F1∂aℝ(X::Matrix{T}, θ₊::SIRParams, ∂a∂aℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂a(X::Matrix{T}, θ₊::SIRParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b = θ₊
            grads[:, 1] = -max.(X[:, 1], 0.0) .* max.(X[:, 2], 0.0)  # ∇a(dΛₛ/dt) = - Λₛ Λᵢ
            grads[:, 2] = max.(X[:, 1], 0.0) .* max.(X[:, 2], 0.0)    # ∇a(dΛᵢ/dt) = Λₛ Λᵢ
            # grads[:, 3] .= 0                  # ∇a(dΛᵣ/dt) = 0
            return grads
        end

        ∂F1∂a = _∂F1∂a(X, θ₊)  # [T × C]
        ∂F1∂aℝ = ∂F1∂a .* ∂a∂aℝ      # [T × C]
        return ∂F1∂aℝ
    end

    function _∂F1∂bℝ(X::Matrix{T}, θ₊::SIRParams, ∂b∂bℝ::T2) where {T<:Real,T2<:Real}

        function _∂F1∂b(X::Matrix{T}, θ₊::SIRParams) where {T<:Real}
            grads = Matrix{Union{Float64,ForwardDiff.Dual}}(undef, size(X))
            grads .= 0.0
            # @unpack a, b = θ₊
            # grads[:, 1] .= 0       # ∇b(dΛₛ/dt) = 0
            grads[:, 2] = -max.(X[:, 2], 0.0)  # ∇b(dΛᵢ/dt) = - Λᵢ
            grads[:, 3] = max.(X[:, 2], 0.0)    # ∇b(dΛᵣ/dt) = Λᵢ
            return grads
        end

        ∂F1∂b = _∂F1∂b(X, θ₊)  # [T × C]
        ∂F1∂bℝ = ∂F1∂b .* ∂b∂bℝ      # [T × C]
        return ∂F1∂bℝ
    end

    @unpack a_ℝ, b_ℝ = θ
    @unpack a_prior, b_prior = θ_priors
    ∂a∂aℝ = ∂θ∂θℝ(a_ℝ, a_prior)  # scalar
    ∂b∂bℝ = ∂θ∂θℝ(b_ℝ, b_prior)  # scalar

    ∂F1∂aℝ = _∂F1∂aℝ(X, θ₊, ∂a∂aℝ)  # [T × C]
    ∂F1∂bℝ = _∂F1∂bℝ(X, θ₊, ∂b∂bℝ)  # [T × C]

    return [∂F1∂aℝ, ∂F1∂bℝ]  # [T × C] * 2
end  # NOTE: not tested
