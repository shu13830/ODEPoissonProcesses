struct KernelParams
    base_kernel::Union{KernelFunctions.RBFKernel,KernelFunctions.Matern52Kernel}
    α::Float64
    𝓁::Float64
    δ::Float64
end

mutable struct SparseGPCache{T<:Real}
    m::T                                  # constant mean value of GP
    ϕ::KernelParams                       # kernel parameters
    k::KernelFunctions.Kernel             # kernel of GP
    nu::Int                               # number of inducing points
    u::Vector{T}                          # inducing time points
    Kuu::PDMats.PDMat{T,Matrix{T}}        # covariance at u
    Kuu⁻¹::Matrix{T}                      # inverse covariance at u
    Luu::LowerTriangular{T,Matrix{T}}     # Lower Triangle Matrix from Cholesky decomposition of Kuu
    Luu⁻¹::LowerTriangular{T,Matrix{T}}   # inverse of Luu
    Kuu′::Matrix{T}                       # Kuu′. gradient of Kuu (cross covariance)
    Kuu″::Matrix{T}                       # Kuu″. hessian of Kuu (auto covariance)
    # gradient of GP curve
    D::Matrix{T}                          # Kuu′'*Kuu⁻¹. Linear operator to calculate predictive mean of gradient of x as follows: D * x
    A::Matrix{T}                          # Kuu″ - Kuu′'*Kuu⁻¹*Kuu′. covariance of gradient of x
    should_diagonalize_A::Bool            # Whether to diagonalize A
    # sparse approximation
    nt::Int                               # number of observation time points
    t::Vector{T}                          # observation time points
    KtuKuu⁻¹::Matrix{T}                   # Linear operator to calculate predictive mean of y as follows: KtuKuu⁻¹ * x
    K̂::Matrix{T}                          # diagm(diag(Ktt - Kut'*Kuu⁻¹*Kut)): sparse (diagonal) predictive covariance of y
    diagK̂::Vector{T}                      # diag(K̂)

    function SparseGPCache(
        m::T,
        base_kernel::Symbol,
        ascale::T,
        lscale::T,
        δ::T,
        nu::Int,
        u::Vector{T},
        t::Vector{T};
        should_diagonalize_A::Bool=false
    ) where {T<:Real}

        @assert base_kernel ∈ [:RBF, :Matern52]
        if base_kernel == :RBF
            base_kernel = RBFKernel()
        elseif base_kernel == :Matern52
            base_kernel = Matern52Kernel()
        end
        ϕ = KernelParams(base_kernel, ascale, lscale, δ)
        k = ϕ.α^2 * with_lengthscale(ϕ.base_kernel, ϕ.𝓁) + ϕ.δ^2 * WhiteKernel()
        Kuu = k.(u, u') |> PDMat
        Luu = Kuu.chol.L
        Luu⁻¹ = Luu |> inv
        Kuu⁻¹ = Luu⁻¹' * Luu⁻¹
        Kuu′ = dKxy_dx(ascale, lscale, δ, u, u, base_kernel)
        Kuu″ = d2Kxy_dxdy(ascale, lscale, δ, u, u, base_kernel)
        D = calc_gp_gradient_mean_linear_operator(Kuu′, Kuu⁻¹)
        A = calc_gp_gradient_cov(Kuu″, Kuu′, Kuu)
        nt = length(t)
        if u == t
            KtuKuu⁻¹ = diagm(ones(length(t)))
            K̂ = zeros(length(t), length(t))
        else
            KtuKuu⁻¹ = k.(t, u') * Kuu⁻¹
            K̂ = diagm(diag(k.(t, t') - k.(t, u') * Kuu⁻¹ * k.(u, t')))
        end
        diagK̂ = diag(K̂)

        new{T}(
            m, ϕ, k, nu, u, Kuu, Kuu⁻¹, Luu, Luu⁻¹, Kuu′, Kuu″, D, A, should_diagonalize_A, nt, t, KtuKuu⁻¹, K̂, diagK̂
        )
    end
end

function update_kernel_params!(s::SparseGPCache{T}, ϕ::KernelParams) where {T<:Real}
    s.ϕ = ϕ
    @unpack u, t = s
    s.k = s.ϕ.α^2 * with_lengthscale(s.ϕ.base_kernel, s.ϕ.𝓁) + s.ϕ.δ^2 * WhiteKernel()
    s.Kuu = s.k.(u, u') |> PDMat
    s.Luu = s.Kuu.chol.L
    s.Luu⁻¹ = s.Luu |> inv
    s.Kuu⁻¹ = s.Luu⁻¹' * s.Luu⁻¹
    s.Kuu′ = dKxy_dx(s.ϕ.α, s.ϕ.𝓁, s.ϕ.δ, u, u, s.ϕ.base_kernel)
    s.Kuu″ = d2Kxy_dxdy(s.ϕ.α, s.ϕ.𝓁, s.ϕ.δ, u, u, s.ϕ.base_kernel)
    s.D = calc_gp_gradient_mean_linear_operator(s)
    s.A = calc_gp_gradient_cov(s)
    if u == t
        s.KtuKuu⁻¹ = diagm(ones(length(t)))
        s.K̂ = zeros(length(t), length(t))
    else
        s.KtuKuu⁻¹ = s.k.(t, u') * s.Kuu⁻¹
        s.K̂ = diagm(diag(s.k.(t, t') - s.k.(t, u') * s.Kuu⁻¹ * s.k.(u, t')))
    end
    s.diagK̂ = diag(s.K̂)
end

calc_gp_gradient_mean_linear_operator(s::SparseGPCache{T}) where T<:Real = calc_gp_gradient_mean_linear_operator(s.Kuu′, s.Kuu⁻¹)
calc_gp_gradient_mean_linear_operator(Kuu′::Matrix{T}, Kuu⁻¹::Matrix{T}) where T<:Union{Float64,ForwardDiff.Dual} = Kuu′' * Kuu⁻¹

function calc_gp_gradient_cov(
    s::SparseGPCache{T};
    whitening::Float64=1e-6
) where T<:Real

    calc_gp_gradient_cov(s.Kuu″, s.Kuu′, s.Kuu; whitening=whitening)
end

function calc_gp_gradient_cov(
    Kuu″::AbstractMatrix,
    Kuu′::AbstractMatrix,
    Kuu::PDMats.AbstractPDMat;
    whitening::Float64=1e-6
)
    A = (Kuu″ - Kuu′' * (Kuu \ Kuu′)) + whitening * LinearAlgebra.I
    A = A |> Hermitian |> Matrix
    @assert isposdef(A)
    return A    
end

"radial basis function (squared exponential/Gaussian) kernel with additional white noise"
function rbf_kernel(α::T, 𝓁::T, δ::T) where {T<:Real}
    kernel = α^2 * with_lengthscale(RBFKernel(), 𝓁) + δ^2 * WhiteKernel()
    return kernel
end

"Matern52 kernel with additional white noise"
function matern52_kernel(α::T, 𝓁::T, δ::T) where {T<:Real}
    kernel = α^2 * with_lengthscale(Matern52Kernel(), 𝓁) + δ^2 * WhiteKernel()
    return kernel
end