using StatsBase
using Distances
using LinearAlgebra
using StatsPlots
using Random
using Distributions
using EmpiricalDistributions

chain1 = rand(Normal(0, 1), 2000)
chain1_alt = rand(Normal(0, 1), 2000)
chain2 = rand(Normal(3.5, 1.5), 2000)

hist1 = fit(Histogram, chain1; nbins =  20)
hist2 = fit(Histogram, chain2; nbins =  20)
hist1_alt = fit(Histogram, chain1_alt; nbins =  20)

nhist1 = normalize(hist1, mode = :pdf)
nhist2 = normalize(hist2, mode = :pdf)
nhist1_alt = normalize(hist1_alt, mode = :pdf)

hist1_V2 = fit(Histogram, chain1, collect(nhist1.edges...) ∪ collect(nhist2.edges...))
hist2_V2 = fit(Histogram, chain2, collect(nhist1.edges...) ∪ collect(nhist2.edges...))
hist1_alt_V2 = fit(Histogram, chain1_alt, collect(nhist1.edges...) ∪ collect(nhist2.edges...))



nhist1_V2 = normalize(hist1_V2, mode = :pdf)
nhist2_V2 = normalize(hist2_V2, mode = :pdf)
nhist1_alt_V2 = normalize(hist1_alt_V2, mode = :pdf)

nhist1_V2.weights = nhist1_V2.weights .+ 1e-10
nhist2_V2.weights = nhist2_V2.weights .+ 1e-10
nhist1_alt_V2.weights = nhist1_alt_V2.weights .+ 1e-10

# renormalise the weights
nhist1_V2.weights = nhist1_V2.weights ./ sum(nhist1_V2.weights)
nhist2_V2.weights = nhist2_V2.weights ./ sum(nhist2_V2.weights) 
nhist1_alt_V2.weights = nhist1_alt_V2.weights ./ sum(nhist1_alt_V2.weights)


# d1 = UvBinnedDist(nhist1)
# d2 = UvBinnedDist(nhist2)
# d1_alt = UvBinnedDist(nhist1_alt)

# [kldivergence(d1, d2), kldivergence(d1, d1_alt), kldivergence(d1_alt, d2)]

d1_V2 = UvBinnedDist(nhist1_V2)
d2_V2 = UvBinnedDist(nhist2_V2)
d1_alt_V2 = UvBinnedDist(nhist1_alt_V2)

[kldivergence(d1_V2, d2_V2), kldivergence(d1_V2, d1_alt_V2), kldivergence(d1_alt_V2, d2_V2)]

evaluate(Euclidean(), chain1, chain2)
evaluate(Euclidean(), chain1, chain1_alt)
evaluate(Euclidean(), chain1_alt, chain2)

evaluate(Minkowski(20), chain1, chain2)
evaluate(Minkowski(20), chain1, chain1_alt)
evaluate(Minkowski(20), chain1_alt, chain2)