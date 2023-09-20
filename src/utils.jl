getfields(mod) = fieldnames(typeof(mod)) .|> name -> getfield(mod, name)

find_nan(A::AbstractArray) = any(isnan.(A))
function detect_nan(A::AbstractArray)
    @assert ~find_nan(A) "detect NaN"
end

find_non_finite(A::AbstractArray) = ~all(isfinite.(A))
function detect_non_finite(A)
    @assert ~find_non_finite(A) "detect Non-finite value"
end

reverse_dictionary(d::Dict) = Dict(v => k for (k, v) in d)

function nondiagonal_vec(A)
    @assert size(A, 1) == size(A, 2)
    return [A[cids] for cids in CartesianIndices(A) if cids[1] != cids[2]]
end

function reccursive_flatten(nested_arr::AbstractArray)
    cache = nested_arr
    while true
        nested_arr = collect(Base.Iterators.flatten(nested_arr))
        if cache == nested_arr
            break
        else
            cache = nested_arr
        end
    end
    return nested_arr
end

"""
    decimaldate(dt)

Convert datetime to float value.

# Usage
```
julia> decimaldaye(Dates.Date("2000/01/01", "yyyy/mm/dd"))
2000.0

julia> decimaldate(Dates.Date("2021-12-31", "yyyy-mm-dd"))
2021.9972602739726
```
"""
function decimaldate(dt::Union{Date,DateTime})
    function yfrac(dt::Union{Date,DateTime})
        (dayofyear(dt) - 1) / daysinyear(dt)
    end
    return year(dt) + yfrac(dt)
end

strdate2float(datestr::String, datefmt::String) = decimaldate(Dates.Date(datestr, datefmt))

function float_times(strdates::Vector{String}, datefmt::String)
    return Float64[strdate2float(datestr, lowercase(datefmt)) for datestr in strdates]
end

function normalize_times(times::Vector{T}, t_min::T, t_max::T) where {T<:Real}
    return (times .- t_min) ./ (t_max - t_min)
end

"""
    onehotvec(L, i_hot)

Create L length one-hot vector with i th elemnt as one.
"""
function onehotvec(L::T, i_hot::T) where {T<:Integer}
    vec = zeros(Bool, L)
    vec[i_hot] = true
    return vec
end

"""
    diagi(A)

diagonal indices array for n dimensional array.
```
julia> A = zeros(3,3,3)
julia> A[diagi(A)] = [1,2,3]
3×3×3 Array{Float64, 3}:
[:, :, 1] =
 1.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  0.0

[:, :, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  3.0
```
"""
diagi(A::AbstractArray{T,N} where {T}) where {N} = CartesianIndex.(fill(1:minimum(size(A)), N)...)

"""
    join_covs(A, B)

join covariance matrix

# Usage
A = [1. 1.; 1. 1.]
B = [3. 3. 3.; 3. 3. 3.; 3. 3. 3.]
join_covs(A, B)

# 7×7 Array{Float64,2}:
#  1.0  1.0  0.0  0.0  0.0
#  1.0  1.0  0.0  0.0  0.0
#  0.0  0.0  3.0  3.0  3.0
#  0.0  0.0  3.0  3.0  3.0
#  0.0  0.0  3.0  3.0  3.0
"""
function join_covs(covs::Vector{Matrix{T}}) where {T<:Real}
    n_covs = length(covs)
    n_cov_sizes = map(q -> size(q)[1], covs)
    M = sum(n_cov_sizes)
    cov = zeros(M, M)
    slice_start_index = 1
    for i in 1:n_covs
        slice = slice_start_index:(slice_start_index+n_cov_sizes[i]-1)
        cov[slice, slice] = covs[i]
        slice_start_index += n_cov_sizes[i]
    end
    return cov
end

function join_matrices(matrices::Vector{Matrix{T}}) where {T<:Real}
    n_mats = length(matrices)
    row_lengths = map(q -> size(q)[1], matrices)
    col_lengths = map(q -> size(q)[2], matrices)

    joined_mat = zeros(sum(row_lengths), sum(col_lengths))
    r_start = 1
    c_start = 1
    for (i, (nr, nc)) in enumerate(zip(row_lengths, col_lengths))
        r_end = r_start + nr - 1
        c_end = c_start + nc - 1
        joined_mat[r_start:r_end, c_start:c_end] = matrices[i]
        r_start = r_start + nr
        c_start = c_start + nc
    end
    return joined_mat
end

"""
    weave_covs([A, B, C])

weave covariance matrix

# Usage
A = [1. 1.; 1. 1.]
B = [2. 2.; 2. 2.]
C = [3. 3.; 3. 3.]
weave_covs([A, B, C])

# 6×6 Array{Float64,2}:
#  1.0  0.0  0.0  1.0  0.0  0.0
#  0.0  2.0  0.0  0.0  2.0  0.0
#  0.0  0.0  3.0  0.0  0.0  3.0
#  1.0  0.0  0.0  1.0  0.0  0.0
#  0.0  2.0  0.0  0.0  2.0  0.0
#  0.0  0.0  3.0  0.0  0.0  3.0
"""
function weave_covs(covs::Vector{Matrix{T}}) where {T<:Real}
    # size check
    @assert length(Set(size.(covs))) == 1 "Error: The size of covariances must be equall."
    n_covs = length(covs)
    m = size(covs[1])[1]
    M = m * n_covs
    weaved_mat = zeros(M, M)
    for i in 1:M, j in 1:M
        cov_index_i = (i - 1) % n_covs + 1
        cov_index_j = (j - 1) % n_covs + 1
        if cov_index_i == cov_index_j
            row_index = div(i - 1, n_covs) + 1
            col_index = div(j - 1, n_covs) + 1
            cov = covs[cov_index_i]
            weaved_mat[i, j] = cov[row_index, col_index]
        end
    end
    return weaved_mat
end

# combinations_asymmetric(C::Int) = [(i, j) for j in 1:C, i in 1:C if i != j]
function combinations_asymmetric_(C::Int)
    return [pair for pair in vec(CartesianIndices(zeros(C, C))) if pair[1] != pair[2]]
    # NOTE もとの実装（↑でコメントアウトしたやつ）と順番が異なる
end
