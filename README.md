# NoiseModels.jl
**Generative noise models**

### Installation

```julia
julia> # press ] for pkg mode
pkg> add NoiseModels
```

### Models

Currently the only model implemented is a `GaussianNoiseModel`. This model generates multichannel colored Gaussian noise given a noise sample or a covariance tensor.

### Usage

Given `training_data` as a $T \times N$ matrix of $T$ time samples and $N$ channels,
we train a `GaussianNoiseModel` as follows:

```julia
using NoiseModels

# fit a GaussianNoiseModel to training_data
model = fit(GaussianNoiseModel, training_data)
```

To generate noise samples with the same statistics as the training data:

```julia
# generate 10000 time samples of N channel random noise
generated = rand(model, 10000)
```
