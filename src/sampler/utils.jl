function get_state_vector(
    mod::ODEPoissonProcess, targets::Vector{Symbol}
)
    @assert all([t in [:y, :x, :θ, :ϕ] for t in targets])
    @unpack gpcache, Y, X, θ = mod.gm
    v = Vector{Float64}[]
    if :y in targets
        push!(v, get_y_vec(Y))
    end
    if :x in targets
        push!(v, get_x_vec(gpcache, X))
    end
    if :θ in targets
        push!(v, get_θ_vec(θ))
    end
    if :ϕ in targets
        push!(v, get_ϕ_vec(gpcache) .|> log)
    end
    return vcat(v...)
end

function get_state_vector(
    mod::ODEModel, targets::Vector{Symbol}
)
    @assert all([t in [:y, :x, :θ, :σ, :ϕ] for t in targets])
    @unpack gpcache, Y, X, θ, σ = mod.gm
    v = Vector{Float64}[]
    if :y in targets
        push!(v, get_y_vec(Y[mod.gm.T+1:end, :]))
    end
    if :x in targets
        push!(v, get_x_vec(gpcache, X))
    end
    if :θ in targets
        push!(v, get_θ_vec(θ))
    end
    if :σ in targets
        push!(v, σ .|> log)
    end
    if :ϕ in targets
        push!(v, get_ϕ_vec(gpcache) .|> log)
    end
    return vcat(v...)
end

function get_y_vec(Y::Matrix{T}) where {T<:Real}
    return vec(Y)
end

function get_x_vec(gpcache::Vector{SparseGPCache{T1}}, X::Matrix{T2}) where {T1<:Real,T2<:Real}
    U, C = size(X)
    x_vec = []
    for (c, x) in enumerate(eachcol(X))
        x_mapped = gpcache[c].Luu⁻¹ * x
        push!(x_vec, x_mapped)
    end
    x_vec = vcat(x_vec...)
    return x_vec
end

function get_θ_vec(θ::ODEParamSetting)
    θ_vec = vcat(param2tuple(θ.params)...)
    return θ_vec
end

function get_ϕ_vec(gpcache::Vector{SparseGPCache{T1}}) where {T1<:Real}
    vcat([[s.ϕ.α, s.ϕ.𝓁, s.ϕ.δ] for s in gpcache]...)
end

function get_proposed_states(
    v, mod::Union{ODEPoissonProcess,ODEModel}, targets::Vector{Symbol}
)

    @assert all([t in [:y, :x, :θ, :σ, :ϕ] for t in targets])

    @unpack T, T⁺, U⁺, gpcache, θ = mod.gm
    @unpack C = mod.data

    if ~(:y in targets)
        len_y = 0
        new_Y = nothing
    else
        if mod isa ODEPoissonProcess
            len_y = T⁺ * C
            new_Y = reshape(v[1:len_y], T⁺, C)
        else  # mod isa ODEModel
            len_y = (T⁺ - T) * C
            new_Yext = reshape(v[1:len_y], T⁺ - T, C)
            new_Y = vcat(mod.gm.Y[1:T, :], new_Yext)
        end
    end
    if ~(:x in targets)
        len_x = 0
        new_X = nothing
    else
        len_x = Int(U⁺ * C)
        new_x = v[len_y+1:len_y+len_x]
        new_X_mapped = reshape(new_x, U⁺, C)
        new_X = zeros(size(new_X_mapped))
        for (c, x) in enumerate(eachcol(new_X_mapped))
            new_X[:, c] = gpcache[c].Luu * x
        end
    end
    if ~(:θ in targets)
        len_θ = 0
        new_θ = nothing
    else
        len_θ = sum(mod.gm.θ.paramlengths)
        new_θ_vec = v[len_y+len_x+1:len_y+len_x+len_θ]
        new_θ = typeof(θ.params)(new_θ_vec)
    end
    if ~(:σ in targets)
        len_σ = 0
        new_σ = nothing
    else
        len_σ = C
        new_σ = v[len_y+len_x+len_θ+1:len_y+len_x+len_θ+len_σ] .|> exp
    end
    if ~(:ϕ in targets)
        new_ϕ = nothing
    else
        len_ϕ = 3*C
        new_ϕ = v[len_y+len_x+len_θ+len_σ+1:len_y+len_x+len_θ+len_σ+len_ϕ] .|> exp
    end

    proposed_states = (Y=new_Y, X=new_X, θ=new_θ, σ=new_σ, ϕ=new_ϕ)
    return proposed_states
end

function reflect_states!(
    v::Vector{T},
    mod::Union{ODEPoissonProcess,ODEModel},
    targets::Vector{Symbol}
) where {T<:Real}

    @assert all([t in [:y, :x, :θ, :σ, :ϕ] for t in targets])
    proposed = get_proposed_states(v, mod, targets)
    if ~isnothing(proposed.Y)
        mod.gm.Y[:, :] = proposed.Y
    end
    if ~isnothing(proposed.X)
        mod.gm.X[:, :] = proposed.X
    end
    if ~isnothing(proposed.θ)
        mod.gm.θ.params = proposed.θ
    end
    if ~isnothing(proposed.ϕ)
        reflect_ϕ!(proposed.ϕ, mod.data.C, mod.gm.gpcache)
    end
    if mod isa ODEModel
        if ~isnothing(proposed.σ)
            mod.gm.σ[:] = proposed.σ
        end
    end
end