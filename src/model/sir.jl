mutable struct SIRParamsFullSupport <: AbstractODEParamsFullSupport
    a_ℝ::Union{Float64,ForwardDiff.Dual}
    b_ℝ::Union{Float64,ForwardDiff.Dual}

    SIRParamsFullSupport() = new()
    SIRParamsFullSupport(a_ℝ::T, b_ℝ::T) where {T<:Float64} = new(a_ℝ, b_ℝ)
    function SIRParamsFullSupport(v::AbstractVector)
        @assert length(v) == 2
        return new(v[1], v[2])
    end
end


mutable struct SIRParams <: AbstractODEParams
    a::Union{Float64,ForwardDiff.Dual}
    b::Union{Float64,ForwardDiff.Dual}
end


mutable struct SIRParamPriors <: AbstractODEParamPriors
    a_prior::ScaledLogitNormal
    b_prior::ScaledLogitNormal
end

function create_SIR_parameter_setting(
    C::Int,
    a_prior::ScaledLogitNormal,
    b_prior::ScaledLogitNormal
)

    @assert C == 3
    a_ℝ = rand(a_prior.normal)
    b_ℝ = rand(b_prior.normal)
    return ODEParamSetting(
        2,
        SIRParamsFullSupport(a_ℝ, b_ℝ),
        SIRParamPriors(a_prior, b_prior),
        [1, 1]
    )
end


function SIRPoissonProcess(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0)],
    base_kernel=:RBF,
    ascale=5.0,
    lscale=0.25,
    δ=0.01,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = SIR_equation
    init_ode_param_setting_fn = create_SIR_parameter_setting
    observed_ids = [1, 2, 3]
    ODEPoissonProcess(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
end


function SIRModel(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0)],
    base_kernel::Symbol=:RBF,
    ascale::T1=5.0,
    lscale::T1=0.25,
    δ::T1=0.01,
    opt_hp::Bool=true,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = SIR_equation
    init_ode_param_setting_fn = create_SIR_parameter_setting
    observed_ids = [1, 2, 3]

    ODEModel(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, opt_hp, should_diagonalize_A)
end




# ==================================================================================================
# SIR Equations
# ==================================================================================================
@doc raw"""
    SIR(Λ, θ₊, times)

Compute the gradient values of SIR equations.
```math
\dfrac{d\lambda_c}{dt} = ...
```
Return matrix with same shape of input matrix Λ namely T × C, where T is number of time points and
C (=3) in the number of classes, i.e. S, I and R.

"""
function SIR_equation(Λ::Matrix{T}, θ₊::SIRParams) where {T<:Real}

    @unpack a, b = θ₊
    Λ_S, Λ_I, Λ_R = Λ[:, 1], Λ[:, 2], Λ[:, 3]
    grad_S = -a .* Λ_S .* Λ_I
    grad_I = a .* Λ_S .* Λ_I .- b .* Λ_I
    grad_R = b .* Λ_I
    return hcat(grad_S, grad_I, grad_R)
end


function get_θ₊(θ::SIRParamsFullSupport, θ_priors::SIRParamPriors)
    @unpack a_ℝ, b_ℝ = θ
    @unpack a_prior, b_prior = θ_priors
    a = scaled_logistic(a_ℝ, a_prior)
    b = scaled_logistic(b_ℝ, b_prior)
    θ₊ = SIRParams(a, b)
    return θ₊
end


function f_sir(dx, x, p, t)
    a = p[1]
    b = p[2]
    x₊ = max.(x, 0.)
    dx[1] = -a * x₊[1] * x₊[2]  # S
    dx[2] = a * x₊[1] * x₊[2] - b * x₊[2]  # I
    dx[3] = b * x₊[2]  # R
end

function sim_SIR_event_data(;
    x0=[10; 0.01; 0.01],
    interval=0.001,
    extrapolation_time_length=0.5,
    θ=(a=2.0, b=2.5),
    seed=125,
    λ0=100.0
)
    # forward solution with SIR model
    f = ODEPoissonProcesses.f_sir
    time_length = 1 + extrapolation_time_length
    tspan = (0.0, time_length)

    W = Int(1 / interval) + 1
    W⁺ = Int(time_length / interval) + 1

    # ground truth ODE parameters
    p = [θ.a, θ.b]
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
    return dat
end