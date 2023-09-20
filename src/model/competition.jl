mutable struct CompetitionParamsFullSupport <: AbstractODEParamsFullSupport
    r_ℝ::Vector{Union{Float64,ForwardDiff.Dual}}
    η_ℝ::Vector{Union{Float64,ForwardDiff.Dual}}
    a_ℝ::Vector{Union{Float64,ForwardDiff.Dual}}

    CompetitionParamsFullSupport() = new()
    function CompetitionParamsFullSupport(
        r_ℝ::Vector{T}, η_ℝ::Vector{T}, a_ℝ::Vector{T}
    ) where {T<:Float64}

        return new(r_ℝ, η_ℝ, a_ℝ)
    end
    function CompetitionParamsFullSupport(v::AbstractVector)
        C = 2
        n_params = length(v)
        while true
            if n_params == C + C^2
                break
            elseif C + C^2 < n_params < (C + 1) + (C + 1)^2
                error("Invalid length of paramm vector.")
            else
                C += 1
            end
        end
        return new(v[1:C], v[C+1:C+C], v[C+C+1:end])
    end
end


mutable struct CompetitionParams <: AbstractODEParams
    r::Vector{Union{Float64,ForwardDiff.Dual}}
    η::Vector{Union{Float64,ForwardDiff.Dual}}
    a::Vector{Union{Float64,ForwardDiff.Dual}}
end


mutable struct CompetitionParamPriors <: AbstractODEParamPriors
    r_prior::ScaledLogitNormal
    η_prior::ScaledLogitNormal
    a_prior::ScaledLogitNormal
end

function create_Competition_parameter_setting(
    C::I,
    r_prior::ScaledLogitNormal,
    η_prior::ScaledLogitNormal,
    a_prior::ScaledLogitNormal
) where {I<:Integer}

    @assert C > 1
    r_ℝ = rand(r_prior.normal, C)
    η_ℝ = rand(η_prior.normal, C)
    a_ℝ = rand(a_prior.normal, C * C - C)
    return ODEParamSetting(
        3,
        CompetitionParamsFullSupport(r_ℝ, η_ℝ, a_ℝ),
        CompetitionParamPriors(r_prior, η_prior, a_prior),
        [C, C, Int(C^2 - C)]
    )
end

function CompetitionPoissonProcess(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 15.0),
        ScaledLogitNormal(0.0, 2.0, 5.0, 15.0),
        ScaledLogitNormal(0.0, 2.0, 0.0, 15.0)],
    base_kernel::Symbol=:RBF,
    ascale::T1=3.0,
    lscale::T1=0.15,
    δ::T1=0.1,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = Competition_equation
    init_ode_param_setting_fn = create_Competition_parameter_setting
    observed_ids = collect(1:data.C)

    ODEPoissonProcess(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
end

function CompetitionModel(
    times::Dict{Int,Vector{T1}},
    classes::Dict{Int,String};
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    λ0::Union{Nothing,T1,Vector{T1}}=nothing, is_λ0_unified::Bool=false,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 15.0),
        ScaledLogitNormal(0.0, 2.0, 5.0, 15.0),
        ScaledLogitNormal(0.0, 2.0, 0.0, 15.0)],
    base_kernel::Symbol=:RBF,
    ascale::T1=3.0,
    lscale::T1=0.15,
    δ::T1=0.1,
    opt_hp::Bool=true,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    data = ModelInput(times, from_to, classes)
    ode = Competition_equation
    init_ode_param_setting_fn = create_Competition_parameter_setting
    observed_ids = collect(1:data.C)

    ODEModel(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, opt_hp, should_diagonalize_A)
end



# ==================================================================================================
# Lotka Volterra Competition Equations
# ==================================================================================================
@doc raw"""
    Lotka_Volterra(Λ, θ₊, times)

Compute the gradient values of Lotka-Volterra Competitive equations.
```math
\dfrac{d\lambda_c}{dt} = r_c \lambda_c \left(1 - \dfrac{\sum_j a_{c, j} \lambda_j}{\eta_c}\right)
```
Return matrix with same shape of input matrix Λ namely T × C, where T is number of time points and
C is the number of classes.

"""
function Competition_equation(
    Λ::Matrix{T}, θ₊::CompetitionParams
) where {T<:Real}

    @unpack r, η, a = θ₊
    C = length(r)
    A = competitive_coef_matrix(a, C)
    return r' .* Λ .* (1 .- A * Λ' ./ η)'
end


@doc raw"""
    competitive_coef_matrix(a, C)

construct competitive coefficient matrix from competitive coefficient vector

```julia
julia> C = 4
julia> a = collect(1:(C^2-C))
julia> get_competitive_mat(a, C)

4×4 Matrix{Float64}:
 1.0  4.0  7.0  10.0
 1.0  1.0  8.0  11.0
 2.0  5.0  1.0  12.0
 3.0  6.0  9.0   1.0
```
"""
function competitive_coef_matrix(a::Vector{T}, C::Int) where {T<:Real}
    A = Array{Union{Float64,ForwardDiff.Dual}}(undef, C, C)
    A .= 1.0
    # indices = [CartesianIndex(j,i) for i in 1:C, j in 1:C if i != j]
    indices = [pair for pair in CartesianIndices((C, C)) if pair[1] != pair[2]]
    A[indices] = a
    return A
end


function get_θ₊(
    θ::CompetitionParamsFullSupport,
    θ_priors::CompetitionParamPriors)

    @unpack r_ℝ, η_ℝ, a_ℝ = θ
    @unpack r_prior, η_prior, a_prior = θ_priors
    r = scaled_logistic.(r_ℝ, r_prior)
    η = scaled_logistic.(η_ℝ, η_prior)
    a = scaled_logistic.(a_ℝ, a_prior)
    θ₊ = CompetitionParams(r, η, a)
    return θ₊
end


function f_competition(dx, x, p, t)
    C = length(x)
    r = p[1:C]
    η = p[C+1:2*C]
    a = p[2*C+1:end]
    A = competitive_coef_matrix(a, C)
    x₊ = max.(x, 0.)
    for i in 1:length(x)
        dx[i] = r[i] * x₊[i] * (1 - sum(A[i, :] .* x₊) / η[i])  # competitor 1
    end
end

function sim_Competition_event_data(;
    x0=[0.1; 0.5; 1.5],
    interval=0.001,
    extrapolation_time_length=0.5,
    θ=(
        r=[7.0, 6.0, 4.0],
        η=[8.0, 10.0, 12.0],
        a=[10.0, 6.0, 0.6, 3.0, 1.2, 0.0]
    ),
    seed=125,
    λ0=100.0
)
    # forward solution with Lotka-Volterra Competition
    f = ODEPoissonProcesses.f_competition
    time_length = 1 + extrapolation_time_length
    tspan = (0.0, time_length)

    W = Int(1 / interval) + 1
    W⁺ = Int(time_length / interval) + 1

    # ground truth ODE parameters
    p = vcat([θ.r, θ.η, θ.a]...)
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