# ODE-guided Poisson processes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://shu13830.github.io/ODEPoissonProcesses/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://shu13830.github.io/ODEPoissonProcesses/dev)
[![Build Status](https://github.com/shu13830/ODEPoissonProcesses/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/shu13830/ODEPoissonProcesses/actions/workflows/CI.yml?query=branch%3Amain) -->

This computer program provides a reference implementation for [our paper](https://openreview.net/forum?id=cJgHzw8Qhq).

This repository contains the following files:
- `src/`: Julia implementation of
  - ODE guided Poisson Processes and its inference scheme, Log-Gaussian Cox Process-based Gradient Matching
  - conventional ODE Models and its inference scheme, Gaussin Process-based Gradient Matching
- `experiments/`: codes for replicating experiments in the paper
- `experiments/data/`: The data used for the experiments

## 1. Citation
If you use this method or this code in your paper, then please cite our paper:

```
@article{
  miyazawa2023estimating,
  title={Estimating Differential Equations from Temporal Point Processes},
  author={Shuichi Miyazawa and Daichi Mochihashi},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=cJgHzw8Qhq},
  note={}
}

```

## 2. Installation
- go to the directory where the files for this package are located (this README.md exists)
- launch julia in cmd and open julia REPL
```cmd
% cd <the directory including this README.md>
% julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.7 (2022-07-19)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```
- enter `]` to switch to the julia package mode
- and activate virtual environment of this package
- and instantiate this package to install dependent packages
```julia
(@v1.6) pkg>
(@v1.6) pkg> activate .
(ODEPoissonProcesses) pkg> resolve
(ODEPoissonProcesses) pkg> instantiate
# running installation ... 
```
- enter backsapce to switch to the julai REPL mode
- load this package (with running precompilation, this takes minutes)
- after precompilation you can use the functionalities of this package
```julia
(ODEPoissonProcesses) pkg> 
julia> using ODEPoissonProcesses
# running precompilation ... 
```

### 2.1 [*OPTION*] use this package in jupyter notebooks
- (Prerequisite) jupyter notebook must be installed.
- activate virtual environment of this pakcage
- add IJulia package
- enter backsapce to switch to the julai REPL mode
- load IJulia and install a new kernel for using this package
```julia
(ODEPoissonProcesses) pkg> add IJulia
julia> using IJulia
julia> IJulia.installkernel("ODEPoissonProcesses", "--project=$(Base.active_project())")
# after this you can choose the "LGCPGradientMacthing" kernel on jupyter notebook
```

## 3. Usage

### 3.1 ODE parameter estimation from event data
```julia
using ODEPoissonProcesses  # take minutes for precompilation

# data format
times = Dict(1 => [0.013, 0.015, ...], 2 => [0.003, 0.012, ...])
classes = Dict(1 => "prey", 2 => "predator")

# init model
mod = PredatorPreyPoissonProcess(times, classes; λ0=100.0)

# inference
## init chain
chain = Chain(mod)

## run mcmc for 10000 iterations where the first 1000 iterations are for bunin.
chain = train!(mod, 10000, chain; n_burnin=1000, algo=:HMC)

# posteror analysis (estimated parameters)
θ = posterior_θ(mod, chain)
```


If the event observation time is given in Date format, it can be converted to a numeric vector as follows:
```julia
dates = ["2022/01/01", "2022/06/30", "2022/12/31", "2023/1/1"]
ODEPoissonProcesses.float_times(dates, "YYYY/MM/DD")
```
```
4-element Vector{Float64}:
 2022.0
 2022.4931506849316
 2022.9972602739726
 2023.0
```

### 3.2 Extrapolation
During model initialization, it is possible to construct a model that also estimates the latent function of the extrapolation time range using the `ex_time` argument. The `ex_time` argument specifies how much additional time should be added as the extrapolaton time, assuming the interpolation time range is from 0 to 1. In the example below, the setting `ex_time=0.5` means that time 0~1 is the interpolation time with observations and 1~1.5 is the extrapolation time.
```julia
mod = PredatorPreyPoissonProcess(times, classes; λ0=100.0, ex_time=0.5)
```

### 3.3 Models with other ODEs
```julia
mod = CompetitionPoissonProcess(times, classes; λ0=100.0)
mod = SIRPoissonProcess(times, classes; λ0=100.0)
mod = InfectedPoissonProcess(times, classes; λ0=100.0)
```

### 3.4 Further settings of model parameters
You can set the parameters of the model by giving them as arguments. The following is a list of the parameters that can be set.
- `from_to::Tuple{Float64,Float64}`: 
- `is_λ0_unified::Bool`: 
- `W::Int`: number of window points in the interpolation time range
- `U::Int`: number of inducing points in the interpolation time range
- `γ::Float64`: Default: `0.3`.
- `m::Float64`: constant mean value of Gaussian Process. Default: `0.0`.
- `ode_param_priors::Vector{ScaledLogitNormal}`: 
- `base_kernel::Symbol`: Type of kernel function. Default: `:RBF`. Options: {`:RBF`, `:Matern52`}
- `ascale::Float64`: amplitude scale parameter of the kernel function. Default: `5.0`
- `lscale::Float64`: length scale parameter of the kernel function. Default: `0.3`
- `δ::Float64`: white noise parameter of the kernel function. Default: `0.01`

### 3.5 Further settings of chain parameters
You can set chain parameters when chain is initialized.
```julia
chain = Chain(mod;
    n_burnin=5000,
    blocks=[HMCBlock(:y, L=10), HMCBlock(:x, L=10), HMCBlock(:θ, L=10)],
    n_keep_state_rec=1000,
    n_thinning=10,
    do_gm=true,
    should_weight_γ=false,
    should_diagonalize_A=true
)
```

- `n_burnin::Int`: Number of burnin iterations.
- `n_keep_state_rec::Int`: Defauls value is `1000`.
- `n_thinning::Int`: Defauls value is `10`.
- `do_gm::Bool`: Defauls value is `true`.
- `α::Float64`: Defauls value is `0.0`.
- `target::Vector{Symbol}`: Defauls value is `[:y, :x, :θ]`.

You can reset chain parameters whithin training.
```julia
chain = train!(mod, 10000, chain;
    n_burnin=0,
    burnin=true,
    ϵ=Dict(:y=>0.001, :x=>0.001, :θ=>0.001, :xθ=>0.001, :yx=>0.001, :yxθ=>0.001),
    L=10,
    algo=:HMC,
    n_keep_state_rec=1000,
    n_thinning=10,
    do_gm=true,
    α=0.0,
    target=[:y, :x, :θ]
)
```


### 3.6 Models using GP-based Gradient Matching

```julia
mod = PredatorPreyModel(times, classes; λ0=100.0)
mod = CompetitionModel(times, classes; λ0=100.0)
mod = SIRModel(times, classes; λ0=100.0)
```

## 4. Reproduce Experiments

```julia
import Pkg
Pkg.add("Literate")

using Literate

# generate notebooks
Literate.notebook("experiments/jl/exp-5.2.2-PredatorPrey.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.2-Competition.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.2-SIR.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.2-PosteriorAnalysis.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.3-PredatorPrey.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.3-Competition.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.3-SIR.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.3-PosteriorAnalysis.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.4-Infected.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.2.4-PosteriorAnalysis.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.3.1-Patents.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.3.2-WebFrameworks.jl", "experiments/notebook", execute=false)
Literate.notebook("experiments/jl/exp-5.3.3-MovieTrends.jl", "experiments/notebook", execute=false)
```
