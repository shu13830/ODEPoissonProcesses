mutable struct PredatorPreyParamsFullSupport <: AbstractODEParamsFullSupport
    a_ℝ::Union{Float64,ForwardDiff.Dual}
    b_ℝ::Union{Float64,ForwardDiff.Dual}
    c_ℝ::Union{Float64,ForwardDiff.Dual}
    d_ℝ::Union{Float64,ForwardDiff.Dual}

    PredatorPreyParamsFullSupport() = new()
    function PredatorPreyParamsFullSupport(
        a_ℝ::T, b_ℝ::T, c_ℝ::T, d_ℝ::T
    ) where {T<:Float64}

        return new(a_ℝ, b_ℝ, c_ℝ, d_ℝ)
    end
    function PredatorPreyParamsFullSupport(v::AbstractVector)
        @assert length(v) == 4
        return new(v[1], v[2], v[3], v[4])
    end
end

mutable struct PredatorPreyParams <: AbstractODEParams
    a::Union{Float64,ForwardDiff.Dual}
    b::Union{Float64,ForwardDiff.Dual}
    c::Union{Float64,ForwardDiff.Dual}
    d::Union{Float64,ForwardDiff.Dual}
end

mutable struct PredatorPreyParamPriors <: AbstractODEParamPriors
    a_prior::ScaledLogitNormal
    b_prior::ScaledLogitNormal
    c_prior::ScaledLogitNormal
    d_prior::ScaledLogitNormal
end

function create_PredatorPrey_parameter_setting(
    C::Int,
    a_prior::ScaledLogitNormal,
    b_prior::ScaledLogitNormal,
    c_prior::ScaledLogitNormal,
    d_prior::ScaledLogitNormal
)

    @assert C == 2
    a_ℝ = rand(a_prior.normal)
    b_ℝ = rand(b_prior.normal)
    c_ℝ = rand(c_prior.normal)
    d_ℝ = rand(d_prior.normal)
    return ODEParamSetting(
        4,
        PredatorPreyParamsFullSupport(a_ℝ, b_ℝ, c_ℝ, d_ℝ),
        PredatorPreyParamPriors(a_prior, b_prior, c_prior, d_prior),
        [1, 1, 1, 1]
    )
end


function PredatorPreyPoissonProcess(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0)],
    base_kernel=:RBF,
    ascale=3.0,
    lscale=0.15,
    δ=0.1,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = PredatorPrey_equation
    init_ode_param_setting_fn = create_PredatorPrey_parameter_setting
    observed_ids = [1, 2]

    ODEPoissonProcess(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
end


function PredatorPreyModel(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0)],
    base_kernel::Symbol=:RBF,
    ascale::T1=3.0,
    lscale::T1=0.15,
    δ::T1=0.1,
    opt_hp::Bool=true,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = PredatorPrey_equation
    init_ode_param_setting_fn = create_PredatorPrey_parameter_setting
    observed_ids = [1, 2]

    ODEModel(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, opt_hp, should_diagonalize_A)
end

# ==================================================================================================
# Lotka Volterra Competition Equations
# ==================================================================================================
@doc raw"""
    PredatorPrey_equation(Λ, θ₊, times)

Compute the gradient values of Lotka-Volterra Predator-Prey equations.
```math
\dfrac{d\lambda_c}{dt} = ...
```
Return matrix with same shape of input matrix Λ namely T × C, where T is number of time points and
C (=2) in the number of classes, i.e. predator and prey.

"""
function PredatorPrey_equation(
    Λ::Matrix{T}, θ₊::PredatorPreyParams
) where {T<:Real}

    @unpack a, b, c, d = θ₊
    Λ_prey, Λ_predator = Λ[:, 1], Λ[:, 2]
    Λ_prey = max.(Λ_prey, 0.)
    Λ_predator = max.(Λ_predator, 0.)
    grad_predator = Λ_predator .* (-c .+ d .* Λ_prey)
    grad_prey = Λ_prey .* (a .- b .* Λ_predator)
    return hcat(grad_prey, grad_predator)
end


function get_θ₊(
    θ::PredatorPreyParamsFullSupport,
    θ_priors::PredatorPreyParamPriors)

    @unpack a_ℝ, b_ℝ, c_ℝ, d_ℝ = θ
    @unpack a_prior, b_prior, c_prior, d_prior = θ_priors
    a = scaled_logistic.(a_ℝ, a_prior)
    b = scaled_logistic.(b_ℝ, b_prior)
    c = scaled_logistic.(c_ℝ, c_prior)
    d = scaled_logistic.(d_ℝ, d_prior)
    θ₊ = PredatorPreyParams(a, b, c, d)
    return θ₊
end

function f_predatorprey(dx, x, p, t)
    a = p[1]
    b = p[2]
    c = p[3]
    d = p[4]
    x₊ = max.(x, 0.)
    dx[1] = a * x₊[1] - b * x₊[1] * x₊[2]   # prey
    dx[2] = -c * x₊[2] + d * x₊[1] * x₊[2]  # predator
end

function sim_PredatorPrey_event_data(;
    x0=[0.5; 1.0],
    interval=0.001,
    extrapolation_time_length=0.5,
    θ=(a=20.0, b=20.0, c=15.0, d=10.0),
    seed=125,
    λ0=100.0
)
    # forward solution with Lotka-Volterra Predator-Prey model
    f = ODEPoissonProcesses.f_predatorprey
    time_length = 1 + extrapolation_time_length
    tspan = (0.0, time_length)

    W = Int(1 / interval) + 1
    W⁺ = Int(time_length / interval) + 1

    # ground truth ODE parameters
    p = [θ.a, θ.b, θ.c, θ.d]
    prob = DifferentialEquations.ODEProblem(f, x0, tspan, p)
    sol = DifferentialEquations.solve(prob, saveat=interval)

    # Generate counts of events
    Random.seed!(seed)
    C = length(x0)
    λ0s = fill(float(λ0), C)
    Λ = reduce(hcat, sol.u)
    X = log.(Λ)
    M = Int.(zeros(C, W))
    for c in 1:C, t in 1:W
        M[c, t] += rand(Poisson(λ0s[c] / W * Λ[c, t]))
    end

    # time points
    times = Vector{Float64}[]
    ticks = collect(range(0, stop=1, length=W))
    for c in 1:C
        times_c = Float64[]
        for (i, t) in enumerate(ticks)
            m = M[c, i]
            if m > 0
                for _ in 1:m
                    push!(times_c, t)
                end
            end
        end
        push!(times, times_c)
    end

    # ground truth dataset
    dat = (
        f=f, tspan=tspan, p=p, θ=θ, x0=x0, Λ=Λ, X=X, M=M, λ0s=float.(λ0s),
        times=times, ticks=ticks, W=W, W⁺=W⁺,
        interval=interval, extrapolation_time_length=extrapolation_time_length)
end