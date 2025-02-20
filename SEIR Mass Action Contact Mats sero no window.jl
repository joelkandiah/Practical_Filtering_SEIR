# Imports 
using OrdinaryDiffEq
using DiffEqCallbacks
using Distributions
using DataInterpolations
using Turing
using LinearAlgebra
using StatsPlots
using Random
using Bijectors
using TOML
using ProgressMeter
using BenchmarkTools
using SparseArrays
using Plots.PlotMeasures
using PracticalFiltering
using DelimitedFiles
# using Revise

@assert length(ARGS) == 8

# Read parameters from command line for the varied submission scripts
# seed_idx = parse(Int, ARGS[1])
# Δ_βt = parse(Int, ARGS[2])

# Data_update_window = parse(Int, ARGS[4])
# n_chains = parse(Int, ARGS[5])
# discard_init = parse(Int, ARGS[6])
# tmax = parse(Float64, ARGS[7])
seed_idx = parse(Int, ARGS[1])
Δ_βt = parse(Int, ARGS[2])
Window_size = parse(Int, ARGS[3])
Data_update_window = parse(Int, ARGS[4])
n_chains = parse(Int, ARGS[5])
discard_init = parse(Int, ARGS[6])
n_warmup_samples = parse(Int, ARGS[7])
tmax = parse(Float64, ARGS[8])

tmax_int = Integer(tmax)

# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
outdir = string("Results/SEIR_MASS_ACTION_SERO/$Δ_βt","_beta/no_window/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")
tmpstore = string("Chains/SEIR_MASS_ACTION_SERO/$Δ_βt","_beta/no_window/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
NA_N= [74103.0, 318183.0, 804260.0, 704025.0, 1634429.0, 1697206.0, 683583.0, 577399.0]
NA = length(NA_N)
N = sum(NA_N)
I0 = [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
u0 = zeros(8,5)
u0[:,1] = NA_N - I0
u0[:,3] = I0
inv_γ = 10  
inv_σ = 3
γ = 1/ inv_γ
σ = 1/ inv_σ
p = [γ, σ, N];

trans_unconstrained_I0 = Bijectors.Logit(1.0/N, NA_N[5] / N)

I0_μ_prior_orig = trans_unconstrained_I0(100 / N)
I0_μ_prior = -9.5

# Set parameters for inference and draw betas from prior
β₀σ = 0.15
β₀μ = 0.14
βσ = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax/ Δ_βt)) + 1)
true_beta[1] = exp(rand(Normal(log(β₀μ), β₀σ)))
for i in 2:(length(true_beta) - 1)
    true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,βσ)))
end 
true_beta[length(true_beta)] = true_beta[length(true_beta)-1]
knots = collect(0.0:Δ_βt:tmax)
knots = knots[end] != tmax ? vcat(knots, tmax) : knots
K = length(knots)

C = readdlm("ContactMats/england_8ag_contact_ldwk1_20221116_stable_household.txt")

# Construct an ODE for the SEIR model
function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
    @inbounds begin
        S = @view u[1,:]
        E = @view u[2,:]
        I = @view u[3,:]
        # R = @view u[:,4]
        # I_tot = @view u[:,5]
    end
    (γ, σ, N) = p_.params_floats
    βt = p_.β_function(t)
    b = βt * sum(p_.C * I * 0.22 ./ N)
    # println(b)
    for a in axes(du,2)
        infection = b * S[a]
        infectious = σ * E[a]
        recovery = γ * I[a]
        @inbounds begin
            du[1,a] = - infection
            du[2,a] = infection - infectious
            du[3,a] = infectious - recovery
            du[4,a] = infection
            du[5,a] = infectious
        end
    end
end;# Construct an ODE for the SEIR model



struct idd_params{T <: Real, T2 <: DataInterpolations.AbstractInterpolation, T3 <: Real}
    params_floats::Vector{T}
    β_function::T2
    N_regions::Int
    C::Matrix{T3}
end

NR = 1

params_test = idd_params(p, ConstantInterpolation(true_beta, knots), NR, C)

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem(sir_tvp_ode!, u0', tspan, params_test);
#? Note the choice of tstops and d_discontinuities to note the changepoints in β
#? Also note the choice of solver to resolve issues with the "stiffness" of the ODE system
sol_ode = solve(prob_ode,
            Tsit5(; thread = OrdinaryDiffEq.False()),
            # callback = cb,
            saveat = 1.0,
            tstops = knots[2:end-1],
            d_discontinuities = knots);

# Optionally StatsPlots.plot the SEIR system
StatsPlots.plot(stack(map(x -> x[3,:], sol_ode.u))',
    xlabel="Time",
    ylabel="Number",
    linewidth = 1)
StatsPlots.plot!(size = (1200,800))
# savefig(string("SEIR_system_wth_CM2_older_infec_$seed_idx.png"))

# Find the cumulative number of cases

I_tot_2 = Array(sol_ode(obstimes))[5,:,:]

# Define utility function for the difference between consecutive arguments in a list f: Array{N} x Array{N} -> Array{N-1}
function rowadjdiff(ary)
    ary1 = copy(ary)
    ary1[:, begin + 1:end] =  (@view ary[:, begin+1:end]) - (@view ary[:,begin:end-1])
    return ary1
end

function adjdiff(ary)
    ary1 = copy(ary)
    ary1[ begin + 1:end] =  (@view ary[begin+1:end]) - (@view ary[begin:end-1])
    return ary1
end



# Number of new infections
# X = adjdiff(I_tot)
# X = rowadjdiff(I_tot)
X = rowadjdiff(I_tot_2)

# Define Gamma distribution by mean and standard deviation
function Gamma_mean_sd_dist(μ, σ)
    α = @. (μ * μ) / (σ * σ)
    θ = @. (σ * σ) / μ
    return Gamma.(α, θ)
end

# Define helpful distributions (arbitrary choice from sample in RTM)
incubation_dist = Gamma_mean_sd_dist(4.0, 1.41)
symp_to_hosp = Gamma_mean_sd_dist(9.0, 8.0666667)

# Define approximate convolution of Gamma distributions
# f: Distributions.Gamma x Distributions.Gamma -> Distributions.Gamma
function approx_convolve_gamma(d1::Gamma, d2::Gamma)
    μ_new = (d1.α * d1.θ) + (d2.α * d2.θ)

    var1 = d1.α * d1.θ * d1.θ
    var2 = d2.α * d2.θ * d2.θ

    σ_new = sqrt(var1 + var2)

    return Gamma_mean_sd_dist(μ_new, σ_new)
end

# Define observation distributions (new infections to reported hospitalisations)
inf_to_hosp = approx_convolve_gamma(incubation_dist,symp_to_hosp)
inf_to_hosp_array_cdf = cdf(inf_to_hosp,1:80)
inf_to_hosp_array_cdf = adjdiff(inf_to_hosp_array_cdf)


using SparseArrays
# Create function to create a matrix to calculate the discrete convolution (multiply convolution matrix by new infections vector to get mean of number of (eligible) hospitalisations per day)
function construct_pmatrix(
    v = inf_to_hosp_array_cdf,
    l = Integer(tmax))
    rev_v = @view v[end:-1:begin]
    len_v = length(rev_v)
    ret_mat = zeros(l, l)
    for i in axes(ret_mat, 1)
        ret_mat[i, max(1, i + 1 - len_v):min(i, l)] .= @view rev_v[max(1, len_v-i+1):end]        
    end
    return sparse(ret_mat)
end

IFR_vec = [0.0000078, 0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]
# [0.00001, 0.0001, 0.0005,......., 0.01, 0.06, 0.1]

# (conv_mat * (IFR_vec .* X)')' ≈ mapreduce(x -> conv_mat * x, hcat, eachrow( IFR_vec .* X))'
# Evaluate mean number of hospitalisations (using proportion of 0.3)
conv_mat = construct_pmatrix(;)  
Y_mu = (conv_mat * (IFR_vec .* X)')'

# Create function to construct Negative binomial with properties matching those in Birrell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

η = rand(Gamma(1,0.2))

# Draw sample of hospitalisations
Y = @. rand(NegativeBinomial3(Y_mu + 1e-3, η));

# StatsPlots.plot mean hospitalisations over hospitalisations
bar(obstimes, Y', legend=true, alpha = 0.3)
StatsPlots.plot!(obstimes, eachrow(Y_mu))
StatsPlots.plot!(size = (1200,800))
# Store ODE system parameters in a dictionary and write to a file
params = Dict(
    "seed" => seeds_list[seed_idx],
    "n_chains" => n_chains,
    "discard_init" => discard_init,
    "tmax" => tmax,
    "N" => N,
    "I0" => I0,
    "γ" => γ,
    "σ" => σ,
    "β" => true_beta,
    "Δ_βt" => Δ_βt,
    "knots" => knots,
    "inital β mean" => β₀μ,
    "initial β sd" => β₀σ,
    "β sd" => βσ,
    "Gamma alpha" => inf_to_hosp.α,
    "Gamma theta" => inf_to_hosp.θ
)

open(string(outdir, "params.toml"), "w") do io
        TOML.print(io, params)
end

# Define Seroprevalence estimates
sero_sens = 0.7659149
# sero_sens = 0.999
sero_spec = 0.9430569
# sero_spec = 0.999

using DelimitedFiles
sample_sizes = readdlm("Serosamples_N/region_1.txt", Int64)

sample_sizes = sample_sizes[1:length(obstimes),:]'

sus_pop = stack(map(x -> x[1,:], sol_ode(obstimes)))
sus_pop_mask = sample_sizes .!= 0
sus_pop_samps = @view (sus_pop ./ NA_N)[sus_pop_mask]
sample_sizes_non_zero = @view sample_sizes[sus_pop_mask]

💉 = @. rand(
    Binomial(
        sample_sizes_non_zero,
        (sero_sens * (1 - (sus_pop_samps))) + ((1-sero_spec) * (sus_pop_samps))
    )
)

obs_exp = zeros(NA, length(obstimes))

obs_exp[sus_pop_mask] = 💉 ./ sample_sizes_non_zero

StatsPlots.scatter(obs_exp', legend = true)

# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp(
    # y,
    K,
    γ = γ,
    σ = σ,
    N = N,
    NA = NA,
    NA_N = NA_N,
    N_regions = NR,
    conv_mat = conv_mat,
    knots = knots,
    C = C,
    sero_sample_sizes = sample_sizes_non_zero,
    sus_pop_mask = sus_pop_mask,
    sero_sens = sero_sens,
    sero_spec = sero_spec,
    obstimes = obstimes,
    I0_μ_prior = I0_μ_prior,
    β₀μ = β₀μ,
    β₀σ = β₀σ,
    βσ = βσ,
    IFR_vec = IFR_vec,
    trans_unconstrained_I0 = trans_unconstrained_I0,
    ::Type{T_β} = Float64,
    ::Type{T_I} = Float64,
    ::Type{T_u0} = Float64,
    ::Type{T_Seir} = Float64;
) where {T_β <: Real, T_I <: Real, T_u0 <: Real, T_Seir <: Real}

    # Set prior for initial infected
    logit_I₀  ~ Normal(I0_μ_prior, 0.2)
    I = Bijectors.inverse(trans_unconstrained_I0)(logit_I₀) * N

    I_list = zero(Vector{T_I}(undef, NA))
    I_list[5] = I
    u0 = zero(Matrix{T_u0}(undef, 5, NA))
    u0[1,:] = NA_N - I_list
    u0[3,:] = I_list

    η ~ truncated(Gamma(1,0.2), upper = maximum(N))

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    β = Vector{T_β}(undef, K)
    log_β = Vector{T_β}(undef, K-2)
    p = [γ, σ, N]
    log_β₀ ~ truncated(Normal(β₀μ, β₀σ);
    #  upper=log(1 / maximum(C * 0.22 / N))
     )
    βₜσ = βσ
    β[1] = exp(log_β₀)
    for i in 2:K-1
        log_β[i-1] ~ Normal(0.0, βₜσ)
        β[i] = exp(log(β[i-1]) + log_β[i-1])
    end
    β[K] = β[K-1]

    # if(I < 1)
    #     # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #         @DynamicPPL.addlogprob! -Inf
    #         return
    #     # end
    # end

    # if(any(β .>  1 / maximum(C * 0.22 / N)) | isnan(η) | any(isnan.(β)))
    #     # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #         @DynamicPPL.addlogprob! -Inf
    #         return
    #     # end
    # end

    params_test = idd_params(p, ConstantInterpolation(β, knots), 1, C) 
    # Run model
    ## Remake with new initial conditions and parameter values
    tspan = (zero(eltype(obstimes)), obstimes[end])
    prob = ODEProblem{true}(sir_tvp_ode!, u0, tspan, params_test)
    
    # display(params_test.β_function)
    ## Solve
    sol = 
        # try
        solve(prob,
            Tsit5(),
            saveat = obstimes,
            # maxiters = 1e7,
            d_discontinuities = knots[2:end-1],
            tstops = knots[2:end-1],
            # abstol = 1e-10,
            # reltol = 1e-7
            )
    # catch e
    #     if e isa InexactError
    #         if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #             @DynamicPPL.addlogprob! -Inf
    #             return
    #         end
    #     else
    #         rethrow(e)
    #     end
    # end

    if any(!SciMLBase.successful_retcode(sol))
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end
    
    ## Calculate new infections per day, X
    sol_X = stack(sol.u)[5,:,:] |>
        rowadjdiff
    # println(sol_X)
    # if (any(sol_X .< -(1e-3)) | any(stack(sol.u)[3,:,:] .< -1e-3))
    #     # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #         @DynamicPPL.addlogprob! -Inf
    #         return
    #     # end
    # end
    # check = minimum(sol_X)
    # println(check)
    y_μ = (conv_mat * (IFR_vec .* sol_X)') |>
        transpose

    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    # if (any(isnan.(y_μ)))
    #     # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #         @DynamicPPL.addlogprob! -Inf
    #         return
    #     # end
    # end
    # y = Array{T_y}(undef, NA, length(obstimes))
    y ~ product_distribution(@. NegativeBinomial3(y_μ + 1e-3, η))


    # Introduce Serological data into the model (for the first region)
    sus_pop = map(x -> x[1,:], sol.u) |>
        stack
    sus_pop_samps = @view (sus_pop ./ NA_N)[sus_pop_mask]
    
    # z = Array{T_z}(undef, length(sero_sample_sizes))
    z ~ product_distribution(@. Binomial(
        sero_sample_sizes,
        (sero_sens * (1 - (sus_pop_samps))) + ((1-sero_spec) * (sus_pop_samps))
    ))

    return (; sol, y, z)
end;

# Define the parameters for the model given the known window size (i.e. vectors that fit within the window) 
knots_window = collect(0:Δ_βt:Window_size)
knots_window = knots_window[end] != Window_size ? vcat(knots_window, Window_size) : knots_window
K_window = length(knots_window)
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, Window_size)
obstimes_window = 1.0:1.0:Window_size

sus_pop_mask_window = sus_pop_mask[:,1:Window_size]
sample_sizes_non_zero_window = @view sample_sizes[:,1:Window_size][sus_pop_mask_window]



# Sample the parameters to construct a "correct order" list of parameters
ode_prior = sample(bayes_sir_tvp(K_window,
        γ,
        σ,
        N,
        NA,
        NA_N,
        NR,
        conv_mat_window,
        knots_window,
        C,
        sample_sizes_non_zero_window,
        sus_pop_mask_window,
        sero_sens,
        sero_spec,
        obstimes_window,
        I0_μ_prior,
        β₀μ,
        β₀σ,
        βσ,
        IFR_vec,
        trans_unconstrained_I0,
    ) | (y = Y[:,1:length(obstimes_window)],
     z = 💉[1:length(sample_sizes_non_zero_window)]
     ), Prior(), 1, discard_initial = 0, thinning = 1);

model_window_unconditioned = bayes_sir_tvp(K_window,
    γ,
    σ,
    N,
    NA,
    NA_N,
    NR,
    conv_mat_window,
    knots_window,
    C,
    sample_sizes_non_zero_window,
    sus_pop_mask_window,
    sero_sens,
    sero_spec,
    obstimes_window,
    I0_μ_prior,
    β₀μ,
    β₀σ,
    βσ,
    IFR_vec,
    trans_unconstrained_I0
    ) 

model_window = model_window_unconditioned| (y = Y[:,1:length(obstimes_window)],
    z = 💉[1:length(sample_sizes_non_zero_window)]
)


# @code_warntype model_window.f(
#     model_window,
#     Turing.VarInfo(model_window),
#     # Turing.SamplingContext(
#     #     Random.default_rng(), Turing.SampleFromPrior(), Turing.DefaultContext()
#     # ),
#     Turing.DefaultContext(),
#     model_window.args...
# )

# ode_test = sample(model_window
#      , Turing.NUTS(1000, 0.65;), MCMCThreads(), discard_init, n_chains, discard_initial = 0, thinning = 1);

# ode_test_2 = sample(model_window
#      , Turing.NUTS(2000, 0.65;), MCMCThreads(), discard_init, n_chains, discard_initial = 0, thinning = 1);

name_map_correct_order = ode_prior.name_map.parameters

using AMGS

# my_sampler = Turing.Sampler(AMGS.RAM_Sampler(0.234, 0.6))
my_sampler_AMGS = Turing.Sampler(AMGS.AMGS_Sampler(0.234, 0.6))
# my_sampler_AMGS_stops = Turing.Sampler(AMGS.AMGS_Stopping_Sampler(0.234, 0.6, 150000, 0.23, 0.24, 3000))

# Perform the chosen inference algorithm
t1_init = time_ns()
# ode_nuts = sample(model_window
    #  , Turing.NUTS(2000, 0.65;), MCMCThreads(), discard_init, n_chains, discard_initial = 0, thinning = 1);
# ode_RAM = sample(model_window,
    # my_sampler,
    # MCMCThreads(),
    # discard_init,
    # n_chains,
    # num_warmup = 10000,
    # discard_initial = 150000,
    # thinning = 200);
ode_amgs = sample(
    model_window,
    my_sampler_AMGS,
    MCMCThreads(),
    200 ,
    n_chains,
    num_warmup = 
        n_warmup_samples,
    discard_initial = 
        discard_init,
    thinning = 200
)
# ode_AMGS_stops = sample(
    # model_window,
    # my_sampler_AMGS_stops,
    # MCMCThreads(),
    # discard_init,
    # n_chains,
    # num_warmup = 10000,
    # discard_initial = 150000,
    # thinning = 200
# )
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)

using JLD2
#save chain_for_densities
# JLD2.write(string(tmpstore, "ode_nuts.jld2"), "ode_nuts", ode_nuts)
# save(string(tmpstore, "ode_nuts.jld2"), "ode_nuts", ode_nuts)
# ode_nuts = load(string(tmpstore, "ode_nuts.jld2"),)["ode_nuts"]
#save test chain
# JLD2.write(string(tmpstore, "ode_test.jld2"), "ode_test", ode_test)
# save(string(tmpstore, "ode_test.jld2"), "ode_test", ode_test)
# ode_test = load(string(tmpstore, "ode_test.jld2"),)["ode_test"]
#save test chain
# JLD2.write(string(tmpstore, "ode_test_2.jld2"), "ode_test_2", ode_test_2)
# save(string(tmpstore, "ode_test_2.jld2"), "ode_test_2", ode_test_2)
# ode_test_2 = load(string(tmpstore, "ode_test_2.jld2"),)["ode_test_2"]

# ode_nuts = ode_test
# Plots to aid in diagnosing issues with convergence
loglikvals = logjoint(model_window, ode_amgs)
# plot(loglikvals)
# plot!(logjoint(model_window, ode_AMGS))
# plot!(logjoint(model_window, ode_AMGS_stops))
# # loglikvalstest = logjoint(model_window, ode_test)
# # loglikvalstest2 = logjoint(model_window, ode_test_2)
# StatsPlots.histogram(loglikvals; normalize = :pdf)
# StatsPlots.density!(loglikvals')

# plot(ode_nuts[450:end,:,:])

# StatsPlots.plot(loglikvals[150:500,1])
# StatsPlots.plot(loglikvalstest[250:500,:])
# savefig(string("loglikvals_try_fix_I0_$seed_idx.png"))

# create list of chains to plot
# listochains = [ode_nuts[150:end,:,i] for i in 1:n_chains ]

# pairplot(ode_nuts[250:end,:,2:end]; legend = false)
# pairplot(ode_test[250:end,:,2:end]; legend = false)
# pairplot(listochains[2:end]...)
# pairplot(ode_nuts[250:end,:,2:end], ode_test[250:end,:,:])

# StatsPlots.plot(ode_nuts[250:end,:,1:end]; legend = false)

# Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)

function generate_confint_infec_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 


    infecs = reduce((v, w) ->cat(v,w, dims = 3), map(y -> stack(map(x -> Array(x.sol)[3,:,:], chnm_res[y,:])), 1:size(chnm_res,1)))
    lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 3)[:,:,1]
    medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 3)[:, :, 1]
    uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 3)[:, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

# Create a function to take in the chains and evaluate the number of recovereds and summarise them (at a specific confidence level)
function generate_confint_recov_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 

    infecs = reduce((v, w) ->cat(v,w, dims = 3), map(y -> stack(map(x -> Array(x.sol)[4,:,:], chnm_res[y,:])), 1:size(chnm_res,1)))
    lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 3)[:,:,1]
    medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 3)[:, :, 1]
    uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 3)[:, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

I_dat = Array(sol_ode(obstimes))[3,:,:] # Population of infecteds at times
R_dat = Array(sol_ode(obstimes))[4,:,:] # Population of recovereds at times

get_beta_quantiles = function(chn, K; ci = 0.95)
    # Get the beta values and calculate the estimated confidence interval and median
        betas = Array(chn)
        beta_idx = [collect(3:K+1); K+1]
    
        betas[:,3:end] .= exp.(cumsum(betas[:,3:end], dims = 2))

        beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
        betas_lci = [quantile(betas[:,i], (1 - ci) / 2) for i in beta_idx]
        betas_uci = [quantile(betas[:,i], 1 - ((1-ci) / 2)) for i in beta_idx]
        return (beta_μ, betas_lci, betas_uci)
end

beta_μ, betas_lci, betas_uci = get_beta_quantiles(ode_amgs[:,:,:], K_window)

betat_no_win = ConstantInterpolation(true_beta, knots)

StatsPlots.plot(obstimes_window,
    ConstantInterpolation(beta_μ, knots_window)(obstimes_window),
    ribbon = (ConstantInterpolation(beta_μ, knots_window)(obstimes_window) - ConstantInterpolation(betas_lci, knots_window)(obstimes_window), ConstantInterpolation(betas_uci, knots_window)(obstimes_window) - ConstantInterpolation(beta_μ, knots_window)(obstimes_window)),
    xlabel = "Time",
    ylabel = "β",
    label="Using the NUTS algorithm",
    title="\nEstimates of β",
    color=:blue,
    lw = 2,
    titlefontsize=18,
    guidefontsize=18,
    tickfontsize=16,
    legendfontsize=12,
    fillalpha = 0.4,
    legendposition = :outerbottom,
    margin = 10mm,
    bottom_margin = 0mm)
StatsPlots.plot!(obstimes_window,
    betat_no_win(obstimes_window),
    color=:red,
    label="True β",
    lw = 2)
StatsPlots.plot!(size = (1200,800))


savefig(string(outdir,"nuts_betas_window_1_$seed_idx.png"))


# Plot the infecteds
confint = generate_confint_infec_init(ode_amgs[:,:,:], model_window)
StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf') , legend = false)
StatsPlots.plot!(I_dat'[1:Window_size,:], linewidth = 2, color = :red)
StatsPlots.plot!(size = (1200,800))

savefig(string(outdir,"infections_nuts_window_1_$seed_idx.png"))

# Plot the recovereds
confint = generate_confint_recov_init(ode_amgs[:,:,:],model_window)
StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf')  , legend = false)
StatsPlots.plot!(R_dat'[1:Window_size,:], linewidth = 2, color = :red)
StatsPlots.plot!(size = (1200,800))

savefig(string(outdir,"recoveries_nuts_window_1_$seed_idx.png"))

# Write a funtion to return the ci for the hospitalisations
function generate_confint_hosps_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 

    hosps = stack(map(x -> x.y, chnm_res)[1:end])
    lowci_hosps = mapslices(x -> quantile(x,(1-cri) / 2), hosps, dims = 3)[:,:,1]
    medci_hosps = mapslices(x -> quantile(x, 0.5), hosps, dims = 3)[:,:,1]
    uppci_hosps = mapslices(x -> quantile(x, cri + (1-cri) / 2), hosps, dims = 3)[:,:,1]
    return (; lowci_hosps, medci_hosps, uppci_hosps)
end

# Plot the hospitalisations
confint = generate_confint_hosps_init(ode_amgs[:,:,:], model_window_unconditioned)

plot_obj_vec = Vector{Plots.Plot}(undef, NA)
for i in eachindex(plot_obj_vec)
    plot_part = StatsPlots.scatter(obstimes_window, confint.medci_hosps[i,:], yerr = (confint.medci_hosps[i,:] .- confint.lowci_hosps[i,:], confint.uppci_hosps[i,:] .- confint.medci_hosps[i,:]), legend = false)
    plot_obj_vec[i] = StatsPlots.scatter!(obstimes_window, Y[i,1:Window_size], color = :red, legend = false)
end
StatsPlots.plot(plot_obj_vec..., layout = (4,2), size = (1200,800))

# Save plot
savefig(string(outdir,"hospitalisations_nuts_window_1_$seed_idx.png"))

# Write a function to return the ci for the Serological data
function generate_confint_sero_init(chn, model, bitmatrix_nonzero, denoms; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn)

    sero = stack(map(function anon(x)
            res = zeros(Float64, bitmatrix_nonzero.dims)
            res[bitmatrix_nonzero] .= x.z ./ denoms
            return res
        end,
    chnm_res)[1:end])
    # display(sero)
    lowci_sero = mapslices(x -> quantile(x,(1-cri) / 2), sero, dims = 3)[:,:,1]
    medci_sero = mapslices(x -> quantile(x, 0.5), sero, dims = 3)[:,:,1]
    uppci_sero = mapslices(x -> quantile(x, cri + (1-cri) / 2), sero, dims = 3)[:,:,1]

    lowci_sero[.!bitmatrix_nonzero] .= NaN
    medci_sero[.!bitmatrix_nonzero] .= NaN
    uppci_sero[.!bitmatrix_nonzero] .= NaN  
    return (; lowci_sero, medci_sero, uppci_sero)
end


plot_obj_vec = Vector{Plots.Plot}(undef, NA)
confint = generate_confint_sero_init(ode_amgs[:,:,:], model_window_unconditioned, sus_pop_mask_window, sample_sizes_non_zero_window)
obs_exp_window = obs_exp[:,1:Window_size]
obs_exp_window[.!sus_pop_mask_window] .= NaN
# plot the Serological data
for i in eachindex(plot_obj_vec)
    plot_part = StatsPlots.scatter(obstimes_window, confint.medci_sero[i,:], yerr = (confint.medci_sero[i,:] .- confint.lowci_sero[i,:], confint.uppci_sero[i,:] .- confint.medci_sero[i,:]), legend = false)
    plot_obj_vec[i] = StatsPlots.scatter!(plot_part, obstimes_window, obs_exp_window[i,:], color = :red, legend = false)
end
StatsPlots.plot(plot_obj_vec..., layout = (4,2), size = (1200,800))

# Save plot
savefig(string(outdir,"sero_nuts_window_1_$seed_idx.png"))


# Create an array to store the window ends to know when to run the model
each_end_time = collect(Window_size:Data_update_window:tmax)
each_end_time = each_end_time[end] ≈ tmax ? each_end_time : vcat(each_end_time, tmax)
each_end_time = Integer.(each_end_time)

# Store the runtimes of each window period
algorithm_times = Vector{Float64}(undef, length(each_end_time))
algorithm_times[1] = runtime_init * 60

list_chains = Vector{Chains}(undef, length(each_end_time))
list_chains[1] = ode_amgs


for idx_time_off_by_1 in eachindex(each_end_time[2:end])
    idx_time = idx_time_off_by_1 + 1
    
    # Determine which betas to fix and which ones to sample
    curr_t = each_end_time[idx_time]
    n_old_betas = Int(floor((curr_t - Window_size) / Δ_βt))
    window_betas = Int(ceil(curr_t / Δ_βt) - n_old_betas)

    # Set the values for the current window: knots, K, convolution matrix...
    local knots_window = collect(0:Δ_βt:curr_t)
    local knots_window = knots_window[end] != curr_t ? vcat(knots_window, curr_t) : knots_window
    local K_window = length(knots_window)
    local conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, curr_t)
    local obstimes_window = 1.0:1.0:curr_t

    local sus_pop_mask_window = sus_pop_mask[:,1:length(obstimes_window)]
    local sample_sizes_non_zero_window = @view sample_sizes[:,1:length(obstimes_window)][sus_pop_mask_window]

    y_data_window = Y[:,1:length(obstimes_window)]
    z_data_window = 💉[1:length(sample_sizes_non_zero_window)]

    local model_window_unconditioned = bayes_sir_tvp(K_window,
        γ,
        σ,
        N,
        NA,
        NA_N,
        NR,
        conv_mat_window,
        knots_window,
        C,
        sample_sizes_non_zero_window,
        sus_pop_mask_window,
        sero_sens,
        sero_spec,
        obstimes_window,
        I0_μ_prior,
        β₀μ,
        β₀σ,
        βσ,
        IFR_vec,
        trans_unconstrained_I0
    )

    local model_window = model_window_unconditioned| (y = y_data_window,
        z = z_data_window,
    )

    t1 = time_ns()
    # list_chains[idx_time] = sample(
    #     model_window,  
    #     PracticalFiltering.PracticalFilter(
    #         fixed_param_names,
    #         window_param_names,
    #         list_chains[idx_time - 1],
    #         NUTS(2000,0.65)
    #     ),
    #     MCMCThreads(),
    #     n_chains;
    #     discard_initial = discard_init
    # )
    list_chains[idx_time] = sample(
        model_window,
        my_sampler_AMGS,
        MCMCThreads(),
        200,
        n_chains,
        num_warmup = n_warmup_samples,
        discard_initial = discard_init,
        thinning = 200
    )
    t2 = time_ns()
    algorithm_times[idx_time] = convert(Int64, t2 - t1)

    # Plot betas
    beta_win_μ, betas_win_lci, betas_win_uci = get_beta_quantiles(list_chains[idx_time], K_window)

    StatsPlots.plot(obstimes_window,
    ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window),
    ribbon = (ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window) - ConstantInterpolation(betas_win_lci, knots_window)(obstimes_window), ConstantInterpolation(betas_win_uci, knots_window)(obstimes_window) - ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window)),
        xlabel = "Time",
        ylabel = "β",
        label="Window $idx_time",
    )
    StatsPlots.plot!(obstimes_window,
        betat_no_win(obstimes_window);
        color=:red,
        label="True β")
    StatsPlots.plot!(size = (1200,800))

    savefig(string(outdir,"β_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # logjoint(model_window, list_chains[idx_time-1])
    logjoint(model_window, list_chains[idx_time])

    # Plot infections
    local confint = generate_confint_infec_init(list_chains[idx_time], model_window)

    StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf') , legend = false)
    StatsPlots.plot!(I_dat[:,1:curr_t]', linewidth = 2, color = :red)
    StatsPlots.plot!(size = (1200,800))

    # Save the plot
    savefig(string(outdir,"infections_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot recoveries
    local confint = generate_confint_recov_init(list_chains[idx_time], model_window)

    StatsPlots.plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf') , legend = false)
    StatsPlots.plot!(R_dat[:,1:curr_t]', linewidth = 2, color = :red)
    StatsPlots.plot!(size = (1200,800))

    # Save the plot 
    savefig(string(outdir,"recoveries_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot hospitalisations
    confint = generate_confint_hosps_init(list_chains[idx_time], model_window_unconditioned)

    local plot_obj_vec = Vector{Plots.Plot}(undef, NA)

    for i in eachindex(plot_obj_vec)
        plot_part = StatsPlots.scatter(1:curr_t, confint.medci_hosps[i,:], yerr = (confint.medci_hosps[i,:] .- confint.lowci_hosps[i,:], confint.uppci_hosps[i,:] .- confint.medci_hosps[i,:]), legend = false)
        plot_obj_vec[i] = StatsPlots.scatter!(plot_part,1:curr_t, Y[i,1:curr_t], color = :red, legend = false)
    end
    StatsPlots.plot(plot_obj_vec..., layout = (4,2), size = (1200,800))

    # Save the plot
    savefig(string(outdir,"hospitalisations_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot serological data

    local plot_obj_vec = Vector{Plots.Plot}(undef, NA)
    local confint = generate_confint_sero_init(list_chains[idx_time], model_window_unconditioned, sus_pop_mask_window, sample_sizes_non_zero_window)
    local obs_exp_window = obs_exp[:,1:curr_t]
    local obs_exp_window[.!sus_pop_mask_window] .= NaN

    for i in eachindex(plot_obj_vec)
        plot_part = StatsPlots.scatter(1:curr_t, confint.medci_sero[i,:], yerr = (confint.medci_sero[i,:] .- confint.lowci_sero[i,:], confint.uppci_sero[i,:] .- confint.medci_sero[i,:]), legend = false)
        plot_obj_vec[i] = StatsPlots.scatter!(plot_part, 1:curr_t, obs_exp_window[i,:], color = :red, legend = false)
    end
    StatsPlots.plot(plot_obj_vec..., layout = (4,2), size = (1200,800))

    # Save the plot
    savefig(string(outdir,"serological_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Check logjoint
    local loglikvals = logjoint(model_window, list_chains[idx_time])
    StatsPlots.histogram(loglikvals; normalize = :pdf)
    savefig(string(outdir,"loglikvals_iter_","$idx_time","_$seed_idx.png"))


end


# Save the chains   
JLD2.jldsave(string(outdir, "chains.jld2"), chains = list_chains)


params = Dict(
    "algorithm_times" => algorithm_times,
    # "algorithm_times_each_sample" => algorithm_times_each_sample
    )

open(string(outdir, "timings.toml"), "w") do io
        TOML.print(io, params)
end
