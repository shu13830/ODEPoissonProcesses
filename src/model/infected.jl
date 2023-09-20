function InfectedPoissonProcess(
    times::Vector{T1};
    λ0::T1,
    from_to::Union{Nothing,Tuple{T1,T1}}=nothing,
    T::Int=100, U::Int=25, ex_time::T1=0.0, γ::T1=0.3, m::T1=0.0,
    ode_param_priors::Vector{ScaledLogitNormal}=[
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0),
        ScaledLogitNormal(0.0, 2.0, 1.0, 50.0)],
    base_kernel::Symbol=:RBF,
    ascale::T1=3.0,
    lscale::T1=0.15,
    δ::T1=0.1,
    should_diagonalize_A::Bool=true
) where {T1<:Real}

    classes = Dict(1 => "S", 2 => "I", 3 => "R")
    times = Dict(1 => Float64[], 2 => times, 3 => Float64[])
    data = ModelInput(times, from_to, classes)
    observed_ids = [2]
    is_λ0_unified = true

    ode = SIR_equation
    init_ode_param_setting_fn = create_SIR_parameter_setting

    mod = ODEPoissonProcess(data, ode, observed_ids, init_ode_param_setting_fn, λ0, is_λ0_unified, T, U, ex_time, 
        γ, m, ode_param_priors, base_kernel, ascale, lscale, δ, should_diagonalize_A)
    mod.gm.M[1, 3] = 0
    return mod
end

