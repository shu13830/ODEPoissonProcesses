module ODEPoissonProcesses

using AbstractGPs,
  AdvancedHMC,
  DataFrames,
  DataStructures,
  Dates,
  DifferentialEquations,
  Distributions,
  EllipticalSliceSampling,
  ForwardDiff,
  GaussianProcesses,
  KernelFunctions,
  LinearAlgebra,
  MCMCChains,
  Optim,
  PDMats,
  Plots,
  Plots.PlotMeasures,
  Printf,
  ProgressMeter,
  Random,
  StatsBase,
  StatsFuns,
  UnPack

import Base: rand, length, @kwdef
import Distributions: pdf, logpdf
import StatsFuns: logistic

include("dist/dist.jl")
include("odeparams.jl")
include("gp.jl")
include("sampler/lgcpgm.jl")
include("sampler/gpgm.jl")

include("model/struct.jl")
include("model/competition.jl")
include("model/predatorprey.jl")
include("model/sir.jl")
include("model/infected.jl")
include("utils.jl")

include("derivative/kernel.jl")
include("derivative/competition.jl")
include("derivative/predatorprey.jl")
include("derivative/sir.jl")

include("sampler/utils.jl")
include("sampler/sample.jl")
include("sampler/record.jl")
include("posthoc.jl")

export ScaledLogitNormal, scaled_logistic, scaled_logit

# struct /  model
export ModelInput, ODEPoissonProcess, LGCPGM, GPGM, ODEModel, Chain
export optimize_hyperparams!
export CompetitionPoissonProcess, PredatorPreyPoissonProcess, SIRPoissonProcess, InfectedPoissonProcess
export CompetitionModel, PredatorPreyModel, SIRModel
export CompetitionParams, PredatorPreyParams, SIRParams
export Competition_equation, PredatorPrey_equation, SIR_equation
export create_Competition_parameter_setting, create_PredatorPrey_parameter_setting, create_SIR_parameter_setting
export sim_Competition_event_data, sim_PredatorPrey_event_data, sim_SIR_event_data
export competitive_coef_matrix

# sampler
export train!, get_state_vector, get_proposed_states, reflect_states!
export HMCBlock, NUTSBlock, ESSBlock, GESSBlock, tHMCBlock, tNUTSBlock

# posthoc
export get_posterior_logA_matrix,
  get_posterior_θ₊_samples,
  get_posterior_X_samples,
  get_posterior_Y_samples,
  simulate_event, 
  convert_event_to_count,
  eval_nll, 
  eval_extrapolative_pred_perf,
  train_lgcp, 
  get_posterior_plgp_samples,
  get_posterior_lgcpgm_samples,
  plot_posterior_function, 
  get_quantile

end
