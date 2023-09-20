# export K, K̂, K⁻¹ 
export dkxy_dx, dKxy_dx                          # cross covariance
export d2kxy_dxdy, d2Kxy_dxdy                    # auto covariance
export ∂RBF∂a, ∂RBF∂l, ∂RBF∂w                    # derivative of RBF kernel
export ∂K∂a, ∂K∂l, ∂K∂w                          # derivative of kernel matrix
export ∂K⁻¹∂a, ∂K⁻¹∂l, ∂K⁻¹∂w                    # derivative of inverse kernel matrix
export kᵀK⁻¹k, ∂kᵀK⁻¹k∂a, ∂kᵀK⁻¹k∂l, ∂kᵀK⁻¹k∂w
export ∂K̂∂a, ∂K̂∂l, ∂K̂∂w                          # derivative of predictive kernel matrix
export d2kxy_dxda, d2kxy_dxdl, d2kxy_dxdw        # derivative of cross covariance
export dK′_da, dK′_dl, dK′_dw                    # derivative of cross covariance matrix
export dD_da, dD_dl, dD_dw                       # derivative of GP graident mean calculation linear operator
export d3kxy_dxdyda, d3kxy_dxdydl, d3kxy_dxdydw  # derivative of auto covariance
export dK″_da, dK″_dl, dK″_dw                    # derivative of auto covariance matrix
export dA_da, dA_dl, dA_dw                       # derivative of GP gradient covariance


# cross covariance
function dkxy_dx(
    a::T1, l::T2, w::T3, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real,T3<:Real}
    k = a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    dk_dx = (x, y) -> ForwardDiff.derivative(x -> k(x, y), x)
    dk_dx
end  # TEST

# cross covariance matrix
function dKxy_dx(
    a::T1, l::T2, w::T3, x::Vector{T}, y::Vector{T}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real,T1<:Real,T2<:Real,T3<:Real}
    _dkxy_dx = dkxy_dx(a, l, w, base_kernel)
    [_dkxy_dx(x_i, y_j) for y_j in y, x_i in x]
end  # TEST

# auto covariance
function d2kxy_dxdy(
    a::T1, l::T2, w::T3, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real,T3<:Real}
    k = a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2k_dxdy = (x, y) -> ForwardDiff.derivative(ξ -> ForwardDiff.derivative(η -> k(ξ, η), y), x)
end  # TEST

# auto covariance matrix
function d2Kxy_dxdy(
    a::T1, l::T2, w::T3, x::Vector{T}, y::Vector{T}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real,T1<:Real,T2<:Real,T3<:Real}
    _d2kxy_dxdy = d2kxy_dxdy(a, l, w, base_kernel)
    [_d2kxy_dxdy(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


function ∂RBF∂a(a, l, x, x′)
    Δ = x - x′
    2 * a * exp(-1 / (2 * l^2) * Δ' * Δ)
end  # TEST

function ∂RBF∂l(a, l, x, x′)
    Δ = x - x′
    a^2 * exp(-1 / (2 * l^2) * Δ' * Δ) * (Δ' * Δ / l^3)
end  # TEST

function ∂RBF∂w(w, x, x′)
    δ = x == x′
    2 * w * δ
end  # TEST

function ∂K∂a(a, l, x, x′, ::RBFKernel)
    [∂RBF∂a(a, l, x[ind[1]], x′[ind[2]]) for ind in CartesianIndices((size(x, 1), size(x′, 1)))]
end  # TEST

function ∂K∂l(a, l, x, x′, ::RBFKernel)
    [∂RBF∂l(a, l, x[ind[1]], x′[ind[2]]) for ind in CartesianIndices((size(x, 1), size(x′, 1)))]
end  # TEST

function ∂K∂w(w, x, x′, ::RBFKernel)
    [∂RBF∂w(w, x[ind[1]], x′[ind[2]]) for ind in CartesianIndices((size(x, 1), size(x′, 1)))]
end  # TEST

function K(a, l, w, x, x′, ::RBFKernel)
    k = a^2 * with_lengthscale(RBFKernel(), l) + w^2 * WhiteKernel()
    k.(x, x′')
end  # TEST

function K̂(a, l, w, x̂, x, ::RBFKernel)
    Kxx = K(a, l, w, x, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())
    Kx̂x̂ = K(a, l, w, x̂, x̂, RBFKernel())

    Kx̂x̂ - Kx̂x * (Kxx \ Kx̂x')
end  # TEST

function K⁻¹(a, l, w, x, ::RBFKernel)
    k = a^2 * with_lengthscale(RBFKernel(), l) + w^2 * WhiteKernel()
    K = k.(x, x') |> PDMat
    L⁻¹ = inv(K.chol.L)
    K⁻¹ = L⁻¹' * L⁻¹
end  # TEST

∂K⁻¹∂a(a, l, w, x, K⁻¹, ::RBFKernel) = -K⁻¹ * ∂K∂a(a, l, x, x, RBFKernel()) * K⁻¹  # TEST
∂K⁻¹∂l(a, l, w, x, K⁻¹, ::RBFKernel) = -K⁻¹ * ∂K∂l(a, l, x, x, RBFKernel()) * K⁻¹  # TEST
∂K⁻¹∂w(a, l, w, x, K⁻¹, ::RBFKernel) = -K⁻¹ * ∂K∂w(w, x, x, RBFKernel()) * K⁻¹  # TEST

function ∂K⁻¹∂a(a, l, w, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    ∂K⁻¹∂a(a, l, w, x, _K⁻¹, RBFKernel())
end  # TEST

function ∂K⁻¹∂l(a, l, w, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    ∂K⁻¹∂l(a, l, w, x, _K⁻¹, RBFKernel())
end  # TEST

function ∂K⁻¹∂w(a, l, w, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    ∂K⁻¹∂w(a, l, w, x, _K⁻¹, RBFKernel())
end  # TEST

# Validation
# ∂K⁻¹∂a(a, l, w, u, RBFKernel())
# (K⁻¹(a + 1e-6, l, w, u, RBFKernel()) - K⁻¹(a, l, w, u, RBFKernel())) / 1e-6 
#
# ∂K⁻¹∂l(a, l, w, u, RBFKernel())
# (K⁻¹(a, l + 1e-6, w, u, RBFKernel()) - K⁻¹(a, l, w, u, RBFKernel())) / 1e-6 
#
# ∂K⁻¹∂w(a, l, w, u, RBFKernel())
# (K⁻¹(a, l, w + 1e-6, u, RBFKernel()) - K⁻¹(a, l, w, u, RBFKernel())) / 1e-6 

function kᵀK⁻¹k(a, l, w, x̂, x, ::RBFKernel)
    Kxx = K(a, l, w, x, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    Kx̂x * (Kxx \ Kx̂x')
end  # TEST

function ∂kᵀK⁻¹k∂a(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂kᵀK⁻¹k∂a(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

function ∂kᵀK⁻¹k∂a(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel)
    _∂K⁻¹∂a = ∂K⁻¹∂a(a, l, w, x, RBFKernel())
    _∂Kx̂x∂a = ∂K∂a(a, l, x̂, x, RBFKernel())

    _∂Kx̂x∂a * Kxx⁻¹ * Kx̂x' + Kx̂x * _∂K⁻¹∂a * Kx̂x' + Kx̂x * Kxx⁻¹ * _∂Kx̂x∂a'
end  # TEST

function ∂kᵀK⁻¹k∂l(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂kᵀK⁻¹k∂l(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

function ∂kᵀK⁻¹k∂l(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel)
    _∂K⁻¹∂l = ∂K⁻¹∂l(a, l, w, x, RBFKernel())
    _∂Kx̂x∂l = ∂K∂l(a, l, x̂, x, RBFKernel())

    _∂Kx̂x∂l * Kxx⁻¹ * Kx̂x' + Kx̂x * _∂K⁻¹∂l * Kx̂x' + Kx̂x * Kxx⁻¹ * _∂Kx̂x∂l'
end  # TEST

function ∂kᵀK⁻¹k∂w(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂kᵀK⁻¹k∂w(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

function ∂kᵀK⁻¹k∂w(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel)
    _∂K⁻¹∂w = ∂K⁻¹∂w(a, l, w, x, RBFKernel())
    _∂Kx̂x∂w = ∂K∂w(w, x̂, x, RBFKernel())

    _∂Kx̂x∂w * Kxx⁻¹ * Kx̂x' + Kx̂x * _∂K⁻¹∂w * Kx̂x' + Kx̂x * Kxx⁻¹ * _∂Kx̂x∂w'
end  # TEST

# Validation
# ∂kᵀK⁻¹k∂a(a, l, w, t, u, RBFKernel())
# (kᵀK⁻¹k(a + 1e-6, l, w, t, u, RBFKernel()) - kᵀK⁻¹k(a, l, w, t, u, RBFKernel())) / 1e-6 
#
# ∂kᵀK⁻¹k∂l(a, l, w, t, u, RBFKernel())
# (kᵀK⁻¹k(a, l + 1e-6, w, t, u, RBFKernel()) - kᵀK⁻¹k(a, l, w, t, u, RBFKernel())) / 1e-6 
#
# ∂kᵀK⁻¹k∂w(a, l, w, t, u, RBFKernel())
# (kᵀK⁻¹k(a, l, w + 1e-6, t, u, RBFKernel()) - kᵀK⁻¹k(a, l, w, t, u, RBFKernel())) / 1e-6 

∂K̂∂a(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel) = (
    ∂K∂a(a, l, x̂, x̂, RBFKernel()) - ∂kᵀK⁻¹k∂a(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, RBFKernel())
)  # TEST
∂K̂∂l(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel) = (
    ∂K∂l(a, l, x̂, x̂, RBFKernel()) - ∂kᵀK⁻¹k∂l(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, RBFKernel())
)  # TEST
∂K̂∂w(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, ::RBFKernel) = (
    ∂K∂w(w, x̂, x̂, RBFKernel()) - ∂kᵀK⁻¹k∂w(a, l, w, x̂, x, Kxx⁻¹, Kx̂x, RBFKernel())
)  # TEST

function ∂K̂∂a(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂K̂∂a(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

function ∂K̂∂l(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂K̂∂l(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

function ∂K̂∂w(a, l, w, x̂, x, ::RBFKernel)
    _K⁻¹ = K⁻¹(a, l, w, x, RBFKernel())
    Kx̂x = K(a, l, w, x̂, x, RBFKernel())

    ∂K̂∂w(a, l, w, x̂, x, _K⁻¹, Kx̂x, RBFKernel())
end  # TEST

# Validation
# ∂K̂∂a(a, l, w, t, u, RBFKernel())
# (K̂(a + 1e-6, l, w, t, u, RBFKernel()) - K̂(a, l, w, t, u, RBFKernel())) / 1e-6 
#
# ∂K̂∂l(a, l, w, t, u, RBFKernel())
# (K̂(a, l + 1e-6, w, t, u, RBFKernel()) - K̂(a, l, w, t, u, RBFKernel())) / 1e-6 
#
# ∂K̂∂w(a, l, w, t, u, RBFKernel())
# (K̂(a, l, w + 1e-6, t, u, RBFKernel()) - K̂(a, l, w, t, u, RBFKernel())) / 1e-6 

# ======================================================================
# Derivative of cross covariance with respect to the kernel parameters
# ======================================================================
function d2kxy_dxda(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = a -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxda = (x, y) -> ForwardDiff.derivative(_a -> ForwardDiff.derivative(_x -> k(_a)(_x, y), x), a)
    d2kxy_dxda
end  # TEST


function dK′_da(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d2kxy_dxda = d2kxy_dxda(a, l, w, base_kernel)
    [_d2kxy_dxda(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


function d2kxy_dxdl(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = l -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxdl = (x, y) -> ForwardDiff.derivative(_l -> ForwardDiff.derivative(_x -> k(_l)(_x, y), x), l)
    d2kxy_dxdl
end  # TEST


function dK′_dl(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d2kxy_dxdl = d2kxy_dxdl(a, l, w, base_kernel)
    [_d2kxy_dxdl(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


function d2kxy_dxdw(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = w -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxdw = (x, y) -> ForwardDiff.derivative(_w -> ForwardDiff.derivative(_x -> k(_w)(_x, y), x), w)
    d2kxy_dxdw
end  # TEST


function dK′_dw(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d2kxy_dxdw = d2kxy_dxdw(a, l, w, base_kernel)
    [_d2kxy_dxdw(x_i, y_j) for y_j in y, x_i in x]
end  # TEST

# ======================================================================================================
# Derivative of Linear operator for Calculating GP gradient mean with respect to the kernel parameters
# ======================================================================================================
function dD_da(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹da = ∂K⁻¹∂a(a, l, w, x, K⁻¹, base_kernel)
    _dK′_da = dK′_da(a, l, w, x, y, base_kernel)

    dD_da = _dK′_da' * K⁻¹ + K′' * _dK⁻¹da
    dD_da
end  # TEST

function dD_dl(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹dl = ∂K⁻¹∂l(a, l, w, x, K⁻¹, base_kernel)
    _dK′_dl = dK′_dl(a, l, w, x, y, base_kernel)

    dD_dl = _dK′_dl' * K⁻¹ + K′' * _dK⁻¹dl
    dD_dl
end  # TEST

function dD_dw(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹dw = ∂K⁻¹∂w(a, l, w, x, K⁻¹, base_kernel)
    _dK′_dw = dK′_dw(a, l, w, x, y, base_kernel)

    dD_dw = _dK′_dw' * K⁻¹ + K′' * _dK⁻¹dw
    dD_dw
end  # TEST

# ======================================================================
# Derivative of auto covariance with respect to the kernel parameters
# ======================================================================
function d3kxy_dxdyda(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = a -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxda = (x, y) -> ForwardDiff.derivative(
        _a -> ForwardDiff.derivative(
            _y -> ForwardDiff.derivative(
                _x -> k(_a)(_x, _y), x), y), a)
    d2kxy_dxda
end  # TEST


function dK″_da(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d3kxy_dxdyda = d3kxy_dxdyda(a, l, w, base_kernel)
    [_d3kxy_dxdyda(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


function d3kxy_dxdydl(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = l -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxdl = (x, y) -> ForwardDiff.derivative(
        _l -> ForwardDiff.derivative(
            _y -> ForwardDiff.derivative(
                _x -> k(_l)(_x, _y), x), y), l)
    d2kxy_dxdl
end  # TEST


function dK″_dl(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d3kxy_dxdydl = d3kxy_dxdydl(a, l, w, base_kernel)
    [_d3kxy_dxdydl(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


function d3kxy_dxdydw(
    a::T, l::T, w::T, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T<:Real}
    k = w -> a^2 * with_lengthscale(base_kernel, l) + w^2 * WhiteKernel()
    d2kxy_dxdw = (x, y) -> ForwardDiff.derivative(
        _w -> ForwardDiff.derivative(
            _y -> ForwardDiff.derivative(
                _x -> k(_w)(_x, _y), x), y), w)
    d2kxy_dxdw
end  # TEST


function dK″_dw(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2}, base_kernel::Union{RBFKernel,Matern52Kernel}
) where {T1<:Real,T2<:Real}
    _d3kxy_dxdydw = d3kxy_dxdydw(a, l, w, base_kernel)
    [_d3kxy_dxdydw(x_i, y_j) for y_j in y, x_i in x]
end  # TEST


# ======================================================================================================
# Derivative of GP gradient covariance with respect to the kernel parameters
# ======================================================================================================
function dA_da(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel};
    should_diagonalize_A::Bool=false
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹da = ∂K⁻¹∂a(a, l, w, x, K⁻¹, base_kernel)
    _dK′_da = dK′_da(a, l, w, x, y, base_kernel)
    _dK″_da = dK″_da(a, l, w, x, y, base_kernel)

    dA_da = _dK″_da -
            (
        _dK′_da' * K⁻¹ * K′ +
        K′' * _dK⁻¹da * K′ +
        K′' * K⁻¹ * _dK′_da
    )
    if should_diagonalize_A
        dA_da = dA_da |> diag |> diagm
    end
    dA_da
end  # TEST


function dA_dl(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel};
    should_diagonalize_A::Bool=false
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹dl = ∂K⁻¹∂l(a, l, w, x, K⁻¹, base_kernel)
    _dK′_dl = dK′_dl(a, l, w, x, y, base_kernel)
    _dK″_dl = dK″_dl(a, l, w, x, y, base_kernel)

    dA_dl = _dK″_dl -
            (
        _dK′_dl' * K⁻¹ * K′ +
        K′' * _dK⁻¹dl * K′ +
        K′' * K⁻¹ * _dK′_dl
    )
    if should_diagonalize_A
        dA_dl = dA_dl |> diag |> diagm
    end

    dA_dl
end  # TEST

function dA_dw(
    a::T1, l::T1, w::T1, x::Vector{T2}, y::Vector{T2},
    K′::Matrix{T3}, K⁻¹::Matrix{T3}, base_kernel::Union{RBFKernel,Matern52Kernel};
    should_diagonalize_A::Bool=false
) where {T1<:Real,T2<:Real,T3<:Real}
    _dK⁻¹dw = ∂K⁻¹∂w(a, l, w, x, K⁻¹, base_kernel)
    _dK′_dw = dK′_dw(a, l, w, x, y, base_kernel)
    _dK″_dw = dK″_dw(a, l, w, x, y, base_kernel)

    dA_dw = _dK″_dw -
            (
        _dK′_dw' * K⁻¹ * K′ +
        K′' * _dK⁻¹dw * K′ +
        K′' * K⁻¹ * _dK′_dw
    )
    if should_diagonalize_A
        dA_dw = dA_dw |> diag |> diagm
    end

    dA_dw
end  # TEST
