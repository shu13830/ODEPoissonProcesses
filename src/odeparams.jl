abstract type AbstractODEParams end
abstract type AbstractODEParamsFullSupport end
abstract type AbstractODEParamPriors end


mutable struct ODEParamSetting
    n_param_type::Int
    params::AbstractODEParamsFullSupport
    priors::AbstractODEParamPriors
    paramlengths::Vector{Int}
end


function θ_prior_vec(θ::ODEParamSetting)
    priors = getfields(θ.priors)
    return vcat([fill(priors[i], l) for (i, l) in enumerate(θ.paramlengths)]...)
end