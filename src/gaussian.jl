import Optimization: OptimizationFunction, OptimizationProblem, solve, SciMLBase, AutoZygote
import OptimizationOptimJL: BFGS
import Statistics: mean, std
import Random: GLOBAL_RNG, AbstractRNG
import Zygote

"""
Multichannel colored Gaussian noise model.
"""
struct GaussianNoiseModel
  s::Vector{Float64}          # per-channel scaling factor
  α::Array{Float64,3}         # noise mixing parameters
end

function Base.show(io::IO, model::GaussianNoiseModel)
  print(io, "GaussianNoiseModel($(size(model.α,1)) channel$(size(model.α,1)==1 ? "" : "s"))")
end

"""
    fit(GaussianNoiseModel, data::AbstractArray; maxlag=64, maxiters=100)

Fit a multichannel colored Gaussian noise model to `data`. Returns a fitted model
struct. `maxlag` controls the maximum time lag (in samples) to estimate noise
correlation. `maxiters` controls the maximum number of iterations for optimization.
"""
function fit(::Type{GaussianNoiseModel}, data::AbstractMatrix; maxlag=64, maxiters=100, verbose=false)
  # estimate scale factor to make the problem numerically stable
  s = std(data; dims=1)
  s[s .< 1e-6] .= 0
  data = data ./ s
  # estimate covariance matrix
  verbose && @info "Estimating covariance matrix..."
  nchannels = size(data, 2)
  R̄ = zeros(nchannels, nchannels, maxlag+1)
  Threads.@threads for i ∈ 1:nchannels
    for j ∈ 1:nchannels
      @simd for δ ∈ 0:maxlag
        @inbounds R̄[i,j,δ+1] = @views mean(data[1+δ:end,i] .* data[1:end-δ,j])
      end
    end
  end
  # pick a good initial value for noise mixing parameters
  α0 = zeros(size(R̄))
  for i ∈ 1:nchannels
    α0[i,i,1] = 1
  end
  # estimate noise mixing parameters
  verbose && @info "Estimating mixing parameters..."
  @info "Initial loss = $(first(_gaussian_loss(vec(α0), R̄)))"
  optf = OptimizationFunction(_gaussian_loss, AutoZygote())
  prob = OptimizationProblem(optf, vec(α0), R̄)
  sol = solve(prob, BFGS(); maxiters)
  @info "Final loss = $(first(_gaussian_loss(sol.u, R̄)))"
  # create noise model
  GaussianNoiseModel(vec(s), reshape(sol.u, size(α0)))
end

function fit(modeltype::Type{GaussianNoiseModel}, data::AbstractVector)
  fit(modeltype, Base.ReshapedArray(data, (length(data),1), ()))
end

"""
    rand(model::GaussianNoiseModel, n)
    rand(rng::AbstractRNG, model::GaussianNoiseModel, n)

Generate `n` time samples of multichannel colored Gaussian noise as per the
`model`.
"""
function Base.rand(rng::AbstractRNG, model::GaussianNoiseModel, n::Integer)
  nchannels, _, nlags = size(model.α)
  z = randn(nchannels, n + nlags - 1)
  x = zeros(n, nchannels)
  for i ∈ 1:nchannels, k ∈ 1:n
    x[k,i] += sum(model.α[i,:,:] .* z[:,k:k+nlags-1])
  end
  x .* model.s'
end

Base.rand(model::GaussianNoiseModel, n::Integer) = rand(GLOBAL_RNG, model, n)

### helpers

function _gaussian_loss(u, R̄)
  α = reshape(u, size(R̄))
  s = 0.0
  @inbounds for i ∈ 1:size(R̄,1), j ∈ 1:size(R̄,2), δ ∈ 0:size(R̄,3)-1
    R = @views sum(α[i,:,δ+1:end] .* α[j,:,1:end-δ])
    s += abs2(R̄[i,j,δ+1] - R)
  end
  s, nothing
end
