# Poisson Distribution =============================================================================

@doc raw"""
    gradlogpdf(s::Distributions.Poisson, x::Real)

derivative of Poisson log pmf w.r.t λ

```math
\mathcal l(\lambda) = -\lambda + x \log(\lambda) - \log(x!)
```

```math
\frac{\partial\mathcal l(\lambda)}{\partial \lambda} = -1 + \frac{x}{\lambda}
```
"""
function gradlogpmf_λ(s::Distributions.Poisson, x::T) where {T<:Real}
    @assert x % 1 == 0
    @assert x >= 0
    return -1 + x / s.λ
end

# Scaled Logit Normal ==============================================================================
# ```math
# f_{X}(x ; \mu, \sigma, \text{min}, \text{max}, k, x_0) =
# \frac{1}{\sigma \sqrt{2 \pi}} \frac{1}{x(1-x)}
# e^{-\frac{(\operatorname{logit}(x, \text{min}, \text{max}, k, x_0)-\mu)^{2}}{2 \sigma^{2}}}
# ``` #TODO 多分grawuth rate の分がPDFの値に乗ってくる
@doc raw"""
The continuous univariate probability distribution of a random variable whose scaled logit has a
normal distribution.

```math
\text{logit}(x, \text{min}, \text{max}, k, x_0) = log\left(\frac{p}{1-p}\right)
```
```math
p = k \left(\frac{x - x_0 - \text{min}}{\text{max} - \text{min}} \right)
```
"""
mutable struct ScaledLogitNormal <: Distributions.ContinuousUnivariateDistribution
    normal::Normal
    min::Float64
    max::Float64
    rate::Float64
    mid::Float64

    function ScaledLogitNormal(
        μ::Float64, σ::Float64, min::Float64, max::Float64;
        rate::Float64=1.0, mid::Float64=0.0
    )

        @assert min < max
        return new(Normal(μ, σ), float(min), float(max), float(rate), float(mid))
    end
end

"""
extended sampler for ScaledLogitNormal.
see, https://juliastats.org/Distributions.jl/stable/extends/
"""
function rand(rng::AbstractRNG, s::ScaledLogitNormal)
    x = Distributions.rand(s.normal)
    x = scaled_logistic(x, s.min, s.max, rate=s.rate, mid=s.mid)
    return x
end

function pdf(s::ScaledLogitNormal, x::Float64)
    x = scaled_logit(x, s.min, s.max, rate=s.rate, mid=s.mid)
    return pdf(s.normal, x)
end

function logpdf(s::ScaledLogitNormal, x::Float64)
    x = scaled_logit(x, s.min, s.max, rate=s.rate, mid=s.mid)
    return logpdf(s.normal, x)
end

@doc raw"""
generalized logistic function\
$x_0$, the $x$ value of the sigmoid's midpoint;\
$\text{min}$, the curve's minimum value;\
$\text{max}$, the curve's maximum value;\
$k$, the logistic growth rate or steepness of the curve.

    scaled_logistic(
        x::Float64, min::Float64=0., max::Float64=1.;
        rate::Float64=1., mid::Float64=0.)

```math
\sigma(x) = \dfrac{\text{max} - \text{min}}{1 - e^{-k(x-x_0)}} - \text{min}
```
"""
function scaled_logistic(
    x::Union{Float64,ForwardDiff.Dual}, min::Float64=0.0, max::Float64=1.0;
    rate::Float64=1.0, mid::Float64=0.0
)

    @assert min < max
    @assert rate > 0
    return (max - min) / (1 + exp(-rate * (x - mid))) + min
end

function scaled_logistic(x::T, s::ScaledLogitNormal) where {T<:Real}
    return scaled_logistic(x, s.min, s.max, rate=s.rate, mid=s.mid)
end

f(s::ScaledLogitNormal) = scaled_logistic

"""mapping x ∈ [min, max] to y ∈ ℝ"""
function scaled_logit(
    x::Float64, min::Float64=0.0, max::Float64=1.0;
    rate::Float64=1.0, mid::Float64=0.0
)

    @assert min <= x <= max
    @assert min < max
    @assert rate > 0
    p = ((x - min) / (max - min))
    @assert 0 < p < 1
    return (log(p) - log(1 - p)) / rate + mid
end


# Scaled Logistic Function =========================================================================

@doc raw"""deriavative of scaled logistic
    grad_scaled_logistic_x(
        x::Float64, min::Float64=0., max::Float64=1.;
        rate::Float64=1., mid::Float64=0.)

- rate: growth rate
- mid: mid point
"""
function grad_scaled_logistic_x(
    x::Float64, min::Float64=0.0, max::Float64=1.0;
    rate::Float64=1.0, mid::Float64=0.0
)

    f = scaled_logistic(x, 0.0, 1.0, rate=rate, mid=mid)
    return rate * (max - min) * f .* (1.0 - f)
end

# Scaled Logit Function ============================================================================

@doc raw"""deriavative of scaled logit
    grad_scaled_logit_x(
        x::Float64, min::Float64=0., max::Float64=1.;
        rate::Float64=1., mid::Float64=0.)

- rate: growth rate
- mid: mid point
"""
function grad_scaled_logit_x(
    x::Float64, min::Float64=0.0, max::Float64=1.0;
    rate::Float64=1.0, mid::Float64=0.0
)
    @assert min <= x <= max
    @assert min < max
    @assert rate > 0
    δ = max - min
    p = (x - min) / δ
    @assert 0 < p < 1
    return (1 / p + 1 / (1 - p)) / (rate * δ)
end


# Scaled Logit Normal Distribution =================================================================
function gradlogpdf(s::ScaledLogitNormal, x::Float64)
    d = grad_scaled_logit_x(x, s.min, s.max, rate=s.rate, mid=s.mid)
    x = scaled_logit(x, s.min, s.max, rate=s.rate, mid=s.mid)
    return d * Distributions.gradlogpdf(s.normal, x)
end
