module NoiseModels

import StatsAPI: fit

export GaussianNoiseModel
export fit

include("gaussian.jl")

end # module
