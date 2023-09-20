function calc_lls(mod::ODEPoissonProcess)
    v = get_state_vector(mod, [:y, :x, :θ])
    lps = mod.gm.targdists[:yxθ].lp(mod.gm.targdists[:yxθ], v; should_sum=false)
    ll_m⎸y = lps.lp₍m⎸y₎
    ll_y⎸xσmϕ = lps.lp₍y⎸xσmϕ₎
    ll_x⎸mϕ = lps.lp₍x⎸mϕ₎
    ll_x⎸mϕθγ = lps.lp₍x⎸mϕθγ₎
    ll_θ = lps.lp₍θ₎
    return (
        ll=ll_m⎸y + ll_y⎸xσmϕ + ll_x⎸mϕ + ll_x⎸mϕθγ + ll_θ,
        ll_m⎸y=ll_m⎸y,
        ll_y⎸xσmϕ=ll_y⎸xσmϕ,
        ll_x⎸mϕ=ll_x⎸mϕ,
        ll_x⎸mϕθγ=ll_x⎸mϕθγ,
        ll_θ=ll_θ
    )
end


function calc_lls(mod::ODEModel)
    v = get_state_vector(mod, [:y, :x, :θ])
    lps = mod.gm.targdists[:yxθ].lp(mod.gm.targdists[:yxθ], v; should_sum=false)
    ll_y⎸xσmϕ = lps.lp₍y⎸xσmϕ₎
    ll_x⎸mϕ = lps.lp₍x⎸mϕ₎
    ll_x⎸mϕθγ = lps.lp₍x⎸mϕθγ₎
    ll_θ = lps.lp₍θ₎
    return (
        ll=ll_y⎸xσmϕ + ll_x⎸mϕ + ll_x⎸mϕθγ + ll_θ,
        ll_y⎸xσmϕ=ll_y⎸xσmϕ,
        ll_x⎸mϕ=ll_x⎸mϕ,
        ll_x⎸mϕθγ=ll_x⎸mϕθγ,
        ll_θ=ll_θ
    )
end


function record_lls(chain::MCMCChains.Chains, lls::NamedTuple)
    _lls = [getfields(lls)...]'
    new_lls = [Array(chain); _lls]
    if all(new_lls[1, :] .== 0)
        new_lls = new_lls[2:end, :] # remove the first  row if it is all zeros (dummy row)
    end
    new_chain = Chains(new_lls, names(chain))
    new_chain = setinfo(new_chain, (setting=chain.info.setting, history=chain.info.history))
    return new_chain
end

function record_states!(mod::Union{ODEPoissonProcess,ODEModel}, setting::ChainSetting, history::ChainHistory)
    rec = history.state_rec
    n_keep = setting.n_keep_state_rec

    push!(rec.y, vec(copy(mod.gm.Y)))
    length(rec.y) > n_keep && popfirst!(rec.y)

    push!(rec.x, vec(copy(mod.gm.X)))
    length(rec.x) > n_keep && popfirst!(rec.x)

    push!(rec.θ, vcat(param2tuple(mod.gm.θ.params)...))
    length(rec.θ) > n_keep && popfirst!(rec.θ)

    push!(rec.ϕ, exp.(get_state_vector(mod, [:ϕ])))
    length(rec.ϕ) > n_keep && popfirst!(rec.ϕ)

    if mod isa ODEModel
        push!(rec.σ, copy(mod.gm.σ))
        length(rec.σ) > n_keep && popfirst!(rec.σ)
    end

end


function init_state_rec()
    rec = (
        y=Vector{Float64}[],
        x=Vector{Float64}[],
        θ=Vector{Float64}[],
        ϕ=Vector{Float64}[]
    )
    return rec
end
