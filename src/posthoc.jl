""
function calc_rmsd(data, mod, chain)
    # ground truth
    gt_g = get_ground_truth_g(data, mod.gm.u⁺)
    gt_θ = get_ground_truth_θ(data)

    # posterior samples
    g_samples = get_posterior_g_samples(mod, chain)
    θ_samples = get_posterior_θ_samples(mod, chain)

    # posterior mean
    g_postmean = mean(g_samples)
    θ_postmean = mean(θ_samples)

    # RMSD
    g_rmsd = rmsd(vec(g_postmean), vec(gt_g))
    θ_rmsd = rmsd(vec(θ_postmean), vec(gt_θ))

    return g_rmsd, θ_rmsd
end

""
function get_ground_truth_g(data, times)
    sol = solve(ODEProblem(data.f, data.x0, data.tspan, data.p), saveat=data.interval)
    func_at_the_time_points = hcat(sol(times).u...)
    # logfunc_at_inducing_points = data.X[:, 1:data.W][:, [any(tick .≈ mod.gm.u) for tick in data.ticks]]
    return func_at_the_time_points' |> Matrix
end

""
function get_ground_truth_θ(data)
    return vcat(collect(data.θ)...)
end

""
function get_posterior_θ_samples(mod::Union{ODEModel,ODEPoissonProcess}, chain)
    θ_records = [typeof(mod.gm.θ.params)(θvec) for θvec in chain.info.history.state_rec.θ]
    θ₊_records = [get_θ₊(θ, mod.gm.θ.priors) for θ in θ_records]
    posterior_parameter_samples = [vcat([getproperty(θ₊, name) for name in fieldnames(typeof(θ₊))]...) for θ₊ in θ₊_records]
    return posterior_parameter_samples
end

""
function get_posterior_g_samples(mod::Union{ODEModel,ODEPoissonProcess}, chain)
    @unpack U, U⁺ = mod.gm
    if mod isa ODEModel
        f_samples = [reshape(x, U⁺, mod.data.C)[1:U, :] for x in chain.info.history.state_rec.x]
    elseif mod isa ODEPoissonProcess
        f_samples = [reshape(exp.(x), U⁺, mod.data.C)[1:U, :] for x in chain.info.history.state_rec.x]
    end
    return f_samples
end

""
function get_posterior_X̂_samples(mod::ODEPoissonProcess, chain)
    @unpack T⁺ = mod.gm
    X̂_samples = [reshape(y, T⁺, mod.data.C) for y in chain.info.history.state_rec.y]
    return X̂_samples
end

""
function sim_ode_with_posterior_θ_samples(data, mod, chain)
    # posterior θ samples
    θ_samples = ODEPoissonProcesses.get_posterior_θ_samples(mod, chain)
    Zs = []

    # simulated ode
    for θ in θ_samples
        prob = ODEProblem(data.f, data.x0, data.tspan, θ)
        sol = solve(prob, saveat=data.interval)
        push!(Zs, hcat(sol.u...))
    end
    Zs = cat(Zs..., dims=3)

    return Zs
end

""
function draw_gt_and_sim_ode(data, Zs; qlow=0.125, qhigh=0.875)
    colors = [
        palette(:seaborn_bright)[1],
        palette(:seaborn_bright)[4],
        palette(:seaborn_bright)[3],
        palette(:seaborn_bright)[2],
        palette(:seaborn_bright)[5]
    ]

    _qlow = mapslices(z -> quantile(z, qlow), Zs, dims=3)
    _median = mapslices(z -> quantile(z, 0.5), Zs, dims=3)
    _qhigh = mapslices(z -> quantile(z, qhigh), Zs, dims=3)

    p = Plots.plot()
    C = length(data.x0)
    len = length(data.ticks)
    for c in 1:C
        _median_c = _median[c, 1:len]
        _qlow_c = _qlow[c, 1:len]
        _qhigh_c = _qhigh[c, 1:len]
        Plots.plot!(data.ticks, _median_c, ribbon=(_median_c - _qlow_c, _qhigh_c - _median_c), c=colors[c], fillalpha=0.3, legend=:none)
    end
    for c in 1:C
        Plots.plot!(data.ticks, data.Λ[c, 1:len], c=colors[c], s=:dash, lw=1)
    end
    return p
end

""
function draw_gt_and_est_ode(data, mod, Zs; qlow=0.125, qhigh=0.875, extrapolation::Bool=true)
    colors = [
        palette(:seaborn_bright)[1],
        palette(:seaborn_bright)[4],
        palette(:seaborn_bright)[3],
        palette(:seaborn_bright)[2],
        palette(:seaborn_bright)[5]
    ]

    _qlow = mapslices(z -> quantile(z, qlow), Zs, dims=3)
    _median = mapslices(z -> quantile(z, 0.5), Zs, dims=3)
    _qhigh = mapslices(z -> quantile(z, qhigh), Zs, dims=3)

    p = Plots.plot()
    C = length(data.x0)
    for c in 1:C
        _median_c = _median[:, c]
        _qlow_c = _qlow[:, c]
        _qhigh_c = _qhigh[:, c]
        Plots.plot!(mod.gm.t⁺, _median_c, ribbon=(_median_c - _qlow_c, _qhigh_c - _median_c), c=colors[c], fillalpha=0.3, legend=:none)
    end
    for c in 1:C
        Plots.plot!(0:data.interval:1.5, data.Λ[c, :], c=colors[c], s=:dash, lw=1)
    end
    if extrapolation
        Plots.vline!([1], ls=:dashdot, c=:black, label=:none, lw=1)
    else
        xlims!(0,1)
    end
    return p
end

""
function eventplot(dat; visible_components::Union{Nothing,Vector{Int}}=nothing, kwargs...)
    colors = [
        palette(:seaborn_bright)[1],
        palette(:seaborn_bright)[4],
        palette(:seaborn_bright)[3],
        palette(:seaborn_bright)[2],
        palette(:seaborn_bright)[5]]
    p = Plots.plot(ticks=:none, axis=false, legend=:none)
    C = length(dat.times)
    for i in 1:C
        if isnothing(visible_components)
            scatter!(dat.times[i], fill(-i, length(dat.times[i])), m=:vline, ms=1.5, c=colors[i]; kwargs...)
        else
            if i in visible_components
                scatter!(dat.times[i], fill(-i, length(dat.times[i])), m=:vline, ms=1.5, c=colors[i]; kwargs...)
            else
                scatter!(dat.times[i], fill(-i, length(dat.times[i])), m=:vline, ms=1.5, c=:white; kwargs...)
            end
        end
    end
    ylims!(-C - 1, 0)
    return p
end

""
function sim_extrapolation_eventcounts(data, mod; n_simulations::Int)
    # prepare ground-truth intensities for extrapolation time 
    obs_times = mod.gm.t⁺
    gt_modulations = ODEPoissonProcesses.get_ground_truth_g(data, obs_times)
    gt_intensities = (mod.gm.λ0 / mod.gm.T)' .* gt_modulations
    gt_intensities_for_extrapolation_time = gt_intensities[mod.gm.T+1:mod.gm.T⁺, :]

    # simulate event counts for extrapolation time from ground-truth intensities
    C = mod.data.C
    n_ext_obs_points = mod.gm.T⁺ - mod.gm.T
    M′_samples = []

    for i in 1:n_simulations
        M′ = zeros(n_ext_obs_points, C)
        for c in 1:C
            for t in 1:n_ext_obs_points
                λ = gt_intensities_for_extrapolation_time[t, c]
                M′[t, c] = rand(Poisson(λ))
            end
        end
        push!(M′_samples, M′)
    end
    return M′_samples
end
