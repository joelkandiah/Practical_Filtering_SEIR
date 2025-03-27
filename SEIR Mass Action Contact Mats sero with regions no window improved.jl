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

@assert length(ARGS) == 2

# Read parameters from command line for the varied submission scripts
seed_idx = parse(Int, ARGS[1])
opts_file = TOML.parsefile(ARGS[2])["opts"]


# Set the parameters for the model
Δ_βt = Int(opts_file["delta_betat"])
Window_size = Int(opts_file["Window_size"])
Data_update_window = Int(opts_file["Data_update_window"])
n_chains = Int(opts_file["n_chains"])
discard_init = Int.(opts_file["discard_init"])
n_warmup_samples = Int(opts_file["n_warmup_samples"])
tmax = Int(opts_file["tmax"])

# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
outdir = string("Results/reg/$Δ_βt","_beta/nowin/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")
tmpstore = string("Chains/reg/$Δ_βt","_beta/nowin/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
NA_N = Matrix(transpose([74103.0 318183.0 804260.0 704025.0 1634429.0 1697206.0 683583.0 577399.0;
          122401.0 493480.0 1123981.0 1028009.0 3063113.0 2017884.0 575433.0 483780.0;
          118454.0 505611.0 1284743.0 1308343.0 2631847.0 2708355.0 1085138.0 895188.0;
          92626.0 396683.0 1014827.0 1056588.0 2115517.0 2253914.0 904953.0 731817.0;
          79977.0 340962.0 851539.0 845215.0 1786666.0 1820128.0 710782.0 577678.0;
          94823.0 412569.0 1085466.0 1022927.0 2175624.0 2338669.0 921243.0 801040.0;
          55450.0 241405.0 633169.0 644122.0 1304392.0 1496240.0 666261.0 564958.0;
]))
NA = size(NA_N)[1]
NR = size(NA_N)[2]
N_pop_R = sum(NA_N, dims = 1)[1,:]
I0_μ_prior = -9.5
trans_unconstrained_I0 = Bijectors.Logit.(1.0 ./N_pop_R, NA_N[5,:]./ N_pop_R)
I0_per_reg = (rand(Normal(I0_μ_prior, 0.2), NR)) .|> Bijectors.Inverse.(trans_unconstrained_I0) |> x -> x .* N_pop_R
I0 = zeros(NA, NR)
I0[5,:] = I0_per_reg
u0 = zeros(5,NA,NR)
u0[1,:,:] = NA_N - I0
u0[3,:,:] = I0
inv_γ = 10  
inv_σ = 3
γ = 1/ inv_γ
σ = 1/ inv_σ
p = [γ, σ ];


# Set parameters for inference and draw betas from prior
β₀σ = 0.15
β₀μ = 0.14
βσ = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax/ Δ_βt)) + 1, NR)
true_beta[1,:] = exp.(rand(Normal(log(β₀μ), β₀σ), NR))
for i in 2:(size(true_beta)[1] - 1)
    true_beta[i,:] = exp.(log.(true_beta[i-1,:]) .+ rand(Normal(0.0,βσ),NR))
end 
true_beta[size(true_beta)[1],:] = true_beta[size(true_beta)[1]-1,:]
knots = collect(0.0:Δ_βt:tmax)
knots = knots[end] != tmax ? vcat(knots, tmax) : knots
K = length(knots)

C = readdlm("ContactMats/england_8ag_contact_ldwk1_20221116_stable_household.txt")

using Tullio

# Construct an ODE for the SEIR model
function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
    # Grab values needed for calculation
    @inbounds @views begin
        S = u[1,:,:]
        E = u[2,:,:]
        I = u[3,:,:]
    end
    (γ, σ) = p_.params_floats

    (infection, infectious, recovery) = p_.cache_SIRderivs
    (βt, b) = p_.cache_beta
    
    # Calculate the force of infection
    map!(x-> x(t), βt, p_.β_functions)
    # βt = map((f) -> f(t), p_.β_functions)
    @tullio grad = false threads = false b[j] = p_.C[i,k] * I[k,j] * 0.22 / p_.NR_pops[j] |> (_) * βt[j]

    infection .= (S .* transpose(b))
    infectious .= σ .* E
    recovery .= γ .* I
    @inbounds begin
        du[1,:,:] .= .- infection
        du[2,:,:] .= infection .- infectious
        du[3,:,:] .= infectious .- recovery
        du[4,:,:] .= infection
        du[5,:,:] .= infectious
    end
end;

struct idd_params{T <: Real, T2 <: Real, T3 <: DataInterpolations.AbstractInterpolation, T4 <: Real, T5 <: Real, T6 <: Real}
    params_floats::Vector{T}
    NR_pops::Vector{T2}
    β_functions::Vector{T3}
    C::Matrix{T4}
    cache_SIRderivs::Vector{Matrix{T5}}
    cache_beta::Vector{Vector{T6}}
end

my_infection = zeros(NA, NR)
my_infectious = zeros(NA, NR)
my_recovery = zeros(NA, NR)
my_βt = zeros(NR)
my_b = zeros(NR)

params_test = idd_params(p, N_pop_R, map(x -> ConstantInterpolation(x, knots), eachcol(true_beta)), C, [my_infection, my_infectious, my_recovery], [my_βt, my_b])

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem{true}(sir_tvp_ode!, u0, tspan, params_test);
#? Note the choice of tstops and d_discontinuities to note the changepoints in β
#? Also note the choice of solver to resolve issues with the "stiffness" of the ODE system
sol_ode = solve(prob_ode,
            Tsit5(),
            saveat = obstimes,
            tstops = knots,
            d_discontinuities = knots,
            );

# Optionally plot the SEIR system
a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a[i] = StatsPlots.plot( stack(map(x -> x[3,:,i], sol_ode.u))',
        xlabel="Time",
        ylabel="Number",
        linewidth = 1,
        legend = false)
end
StatsPlots.plot(a..., layout = (4,2), size = (1200,1800))

# Find the cumulative number of cases
I_dat = Array(sol_ode(obstimes))[3,:,:,:]
R_dat = Array(sol_ode(obstimes))[4,:,:,:]
I_tot_2 = Array(sol_ode(obstimes))[5,:,:,:]

# Define utility function for the difference between consecutive arguments in a list f: Array{N} x Array{N} -> Array{N-1}
function rowadjdiff(ary)
    ary1 = copy(ary)
    ary1[:, begin + 1:end] =  (@view ary[:, begin+1:end]) - (@view ary[:,begin:end-1])
    return ary1
end

function rowadjdiff3d(ary)
    ary1 = copy(ary)
    ary1[:,:, begin + 1:end] =  (@view ary[:,:, begin+1:end]) - (@view ary[:,:,begin:end-1])
    return ary1
end

function adjdiff(ary)
    ary1 = copy(ary)
    ary1[ begin + 1:end] =  (@view ary[begin+1:end]) - (@view ary[begin:end-1])
    return ary1
end

# Number of new infections
X = rowadjdiff3d(I_tot_2)

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
    l = length(obstimes))
    rev_v = @view v[end:-1:begin]
    len_v = length(rev_v)
    ret_mat = zeros(l, l)
    for i in axes(ret_mat, 1)
        ret_mat[i, max(1, i + 1 - len_v):min(i, l)] .= @view rev_v[max(1, len_v-i+1):end]        
    end
    return sparse(ret_mat)
end

IFR_vec = [0.0000078, 0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]

# Evaluate mean number of hospitalisations (using proportion of 0.3)
conv_mat = construct_pmatrix(;)  
Y_mu = stack(transpose(map(x -> conv_mat * transpose(x), eachslice(IFR_vec .* X, dims = 2)))[:,:,1], dims = 2)

# Create function to construct Negative binomial with properties matching those in Birrell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

η = rand(Gamma(1,0.2))

# Draw sample of hospitalisations
Y = @. rand(NegativeBinomial3(Y_mu + 1e-3, η));

# Plot mean hospitalisations over hospitalisations
a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(obstimes, Y[:,i,:]', legend = false, alpha = 0.3)
    a[i] = StatsPlots.plot!(a[i],obstimes, eachrow(Y_mu[:,i,:]))
end
StatsPlots.plot(a..., layout = (4,2), size = (1200,1800))


# Store ODE system parameters in a dictionary and write to a file
params = Dict(
    "seed" => seeds_list[seed_idx],
    "Window_size" => Window_size,
    "Data_update_window" => Data_update_window,
    "n_chains" => n_chains,
    "discard_init" => discard_init,
    "tmax" => tmax,
    "N" => N_pop_R,
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


# Define Seroprevalence estimates
sero_sens = 0.7659149
# sero_sens = 0.999
sero_spec = 0.9430569
# sero_spec = 0.999

using DelimitedFiles
sample_sizes = map(x->readdlm("Serosamples_N/region_$x.txt", Int64), 1:NR)
sample_sizes = stack(sample_sizes)

sample_sizes = sample_sizes[1:length(obstimes),:,:]
sample_sizes = permutedims(sample_sizes, (2,3,1))

sus_pop = stack(map(x -> x[1,:,:], sol_ode(obstimes)))
sus_pop_mask = sample_sizes .!= 0
sus_pop_samps = @view (sus_pop ./ NA_N)[sus_pop_mask]
sample_sizes_non_zero = @view sample_sizes[sus_pop_mask]

💉 = @. rand(
    Binomial(
        sample_sizes_non_zero,
        (sero_sens * (1 - (sus_pop_samps))) + ((1-sero_spec) * (sus_pop_samps))
    )
)

obs_exp = zeros(NA, NR, length(obstimes))

obs_exp[sus_pop_mask] = 💉 ./ sample_sizes_non_zero

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(1:length(obstimes), obs_exp[:,i,:]', legend = false, alpha = 0.3)
end
StatsPlots.plot(a..., layout = (4,2), size = (1200,1800))

# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp(
    # y,
    K,
    γ = γ,
    σ = σ,
    N = N_pop_R,
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
    prob_ode = prob_ode,
    ::Type{T_β} = Float64,
    ::Type{T_I} = Float64,
    ::Type{T_u0} = Float64,
    ::Type{T_Seir} = Float64;

) where {T_β <: Real, T_I <: Real, T_u0 <: Real, T_Seir <: Real}

    logit_I₀_vec = Vector{T_I}(undef, N_regions)

    logit_I₀_EE ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_LDN ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_MID ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_NEY ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_NW ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_SE ~ Normal(I0_μ_prior, 0.2)
    logit_I₀_SW ~ Normal(I0_μ_prior, 0.2)

    logit_I₀_vec[1] = logit_I₀_EE
    logit_I₀_vec[2] = logit_I₀_LDN
    logit_I₀_vec[3] = logit_I₀_MID
    logit_I₀_vec[4] = logit_I₀_NEY
    logit_I₀_vec[5] = logit_I₀_NW
    logit_I₀_vec[6] = logit_I₀_SE
    logit_I₀_vec[7] = logit_I₀_SW
    
    # I = Vector{T_I}(undef, N_regions)
    I = (logit_I₀_vec) .|> Bijectors.Inverse.(trans_unconstrained_I0) |> (x -> x .* N)

    I_list = zero(Matrix{T_I}(undef, NA, N_regions))
    I_list[5,:] .= I
    u0 = zero(Array{T_u0}(undef, 5, NA, N_regions))
    u0[1,:,:] .= NA_N .- I_list
    u0[3,:,:] .= I_list

    η ~ truncated(Gamma(1,0.2), upper = minimum(N))

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    β = Matrix{T_β}(undef, K, N_regions)
    # log_β = Vector{Vector{T_β}}(undef, N_regions)
    # truncate???
    log_β₀_EE ~ Normal(β₀μ, β₀σ)
    log_β₀_LDN ~ Normal(β₀μ, β₀σ)
    log_β₀_MID ~ Normal(β₀μ, β₀σ)
    log_β₀_NEY ~ Normal(β₀μ, β₀σ)
    log_β₀_NW ~ Normal(β₀μ, β₀σ)
    log_β₀_SE ~ Normal(β₀μ, β₀σ)
    log_β₀_SW ~ Normal(β₀μ, β₀σ)

    log_β_EE = Vector{T_β}(undef, K-2) 
    log_β_LDN = Vector{T_β}(undef, K-2) 
    log_β_MID = Vector{T_β}(undef, K-2) 
    log_β_NEY = Vector{T_β}(undef, K-2) 
    log_β_NW = Vector{T_β}(undef, K-2) 
    log_β_SE = Vector{T_β}(undef, K-2) 
    log_β_SW = Vector{T_β}(undef, K-2) 

    p = [γ, σ]

    βₜσ = βσ
    β[1,1] = exp(log_β₀_EE)
    β[1,2] = exp(log_β₀_LDN)
    β[1,3] = exp(log_β₀_MID)
    β[1,4] = exp(log_β₀_NEY)
    β[1,5] = exp(log_β₀_NW)
    β[1,6] = exp(log_β₀_SE)
    β[1,7] = exp(log_β₀_SW)

    # for i in 1:N_regions
        for j in 2:K-1
            log_β_EE[j-1] ~ Normal(0.0, βₜσ)
            β[j,1] = exp(log(β[j-1,1]) + log_β_EE[j-1])

            log_β_LDN[j-1] ~ Normal(0.0, βₜσ)
            β[j,2] = exp(log(β[j-1,2]) + log_β_LDN[j-1])

            log_β_MID[j-1] ~ Normal(0.0, βₜσ)
            β[j,3] = exp(log(β[j-1,3]) + log_β_MID[j-1])

            log_β_NEY[j-1] ~ Normal(0.0, βₜσ)
            β[j,4] = exp(log(β[j-1,4]) + log_β_NEY[j-1])

            log_β_NW[j-1] ~ Normal(0.0, βₜσ)
            β[j,5] = exp(log(β[j-1,5]) + log_β_NW[j-1])

            log_β_SE[j-1] ~ Normal(0.0, βₜσ)
            β[j,6] = exp(log(β[j-1,6]) + log_β_SE[j-1])

            log_β_SW[j-1] ~ Normal(0.0, βₜσ)
            β[j,7] = exp(log(β[j-1,7]) + log_β_SW[j-1])
        end

    β[K,:] .= @view β[K-1,:]

    β_functions = map(x -> ConstantInterpolation(x, knots), eachcol(β))

    infections = zeros(NA, N_regions)
    infectious = zeros(NA, N_regions)
    recovery = zeros(NA, N_regions)
    βt = zeros(N_regions)
    b = zeros(N_regions)
    cache_SIRderivs = [infections, infectious, recovery]
    cache_beta = [βt, b]
    params_test = idd_params(p, N, β_functions, C, cache_SIRderivs, cache_beta)

    prob = remake(prob_ode, u0 = u0, p = params_test)
    ## Solve
    sol = 
        # try
        solve(prob,
            Tsit5(),
            saveat = obstimes,
            d_discontinuities = knots[2:end-1],
            tstops = knots
            )

    if any(!SciMLBase.successful_retcode(sol))
            @DynamicPPL.addlogprob! -Inf
            return
    end
    
    ## Calculate new infections per day, X
    sol_X = stack(sol.u)[5,:,:,:] |>
        rowadjdiff3d

    y_μ = stack(transpose(map(x -> conv_mat * transpose(x), eachslice(IFR_vec .* sol_X, dims = 2)))[:,:,1], dims = 2)

    # Keep this code in case of loss of type inference, but remember to initialise container
    # for i in 1:N_regions
    #     y_μ[:,i,:] .= (conv_mat * (IFR_vec .* sol_X[:,i,:])') |>
    #         transpose
    # end

    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, η))

    # Introduce Serological data into the model (for the first region)
    sus_pop = map(x -> x[1,:,:], sol.u) |>
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
obstimes_window = 1.0:1.0:Window_size
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window))

sus_pop_mask_window = sus_pop_mask[:,:,1:length(obstimes_window)]
sample_sizes_non_zero_window = @view sample_sizes[:,:,1:length(obstimes_window)][sus_pop_mask_window]

tspan_window = (0.0, Window_size)
prob_ode = ODEProblem{true}(sir_tvp_ode!, u0, tspan_window, params_test);

model_window_unconditioned = bayes_sir_tvp(K_window,
    γ,
    σ,
    N_pop_R,
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
    prob_ode
) 

model_window = model_window_unconditioned | (y = Y[:,:,1:length(obstimes_window)],
    z = 💉[1:length(sample_sizes_non_zero_window)]
) 

# @code_warntype model_window.f(
#     model_window,
#     Turing.VarInfo(model_window),
#     Turing.DefaultContext(),
#     model_window.args...
# )

# name_map_correct_order = ode_prior.name_map.parameters
using AMGS

myamgsEE = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsLDN = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsMID = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsNEY = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsNW = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsSE = AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsSW = AMGS_Sampler(0.234, 0.6, true, 1e-5)

myamgs_η = AMGS_Sampler(0.234, 0.6)


global_varnames = (@varname(η),)

EE_varnames = (@varname(logit_I₀_EE), @varname(log_β₀_EE), @varname(log_β_EE))
LDN_varnames = (@varname(logit_I₀_LDN), @varname(log_β₀_LDN), @varname(log_β_LDN))
MID_varnames = (@varname(logit_I₀_MID), @varname(log_β₀_MID), @varname(log_β_MID))
NEY_varnames = (@varname(logit_I₀_NEY), @varname(log_β₀_NEY), @varname(log_β_NEY))
NW_varnames = (@varname(logit_I₀_NW), @varname(log_β₀_NW), @varname(log_β_NW))
SE_varnames = (@varname(logit_I₀_SE), @varname(log_β₀_SE), @varname(log_β_SE))
SW_varnames = (@varname(logit_I₀_SW), @varname(log_β₀_SW), @varname(log_β_SW))

global_regional_gibbs_mine = Gibbs(global_varnames => myamgs_η,
    EE_varnames => myamgsEE,
    LDN_varnames => myamgsLDN,
    MID_varnames => myamgsMID,
    NEY_varnames => myamgsNEY,
    NW_varnames => myamgsNW,
    SE_varnames => myamgsSE,
    SW_varnames => myamgsSW)

using JLD2
if(isfile(string(outdir, "chains1.jld2")))
    chain_file = JLD2.load(string(outdir, "chains1.jld2"))
    ode_nuts = chain_file["chains"]
    runtime_init_precomp = chain_file["runtime_init_precomp"]
    runtime_init = chain_file["runtime_init"]
else
    t1_init_precomp = time_ns()
    sample(
        model_window,
        global_regional_gibbs_mine,
        MCMCThreads(),
        3,
        n_chains,
        num_warmup = 2,
        thinning = 1,
        discard_initial = 0
    );
    t2_init_precomp = time_ns()
    runtime_init_precomp = convert(Int64, t2_init_precomp-t1_init_precomp)
        
    t1_init = time_ns()
    ode_nuts = sample(
        model_window,
        global_regional_gibbs_mine,
        MCMCThreads(),
        1,
        n_chains,
        # num_warmup = 1,
        num_warmup = n_warmup_samples,
        # thinning = 20,
        # discard_initial = 377900,
        discard_initial = discard_init[1],
        # discard_initial = 0
    ) 
    t2_init = time_ns()
    runtime_init = convert(Int64, t2_init-t1_init)
    #35 minutes (after ~5-10min compile) for 25000 + 4000 samples
    JLD2.save(string(outdir, "chains1.jld2"), "chains", ode_nuts, "runtime_init_precomp", runtime_init_precomp, "runtime_init", runtime_init)
end


# Save the chains   
# rhat_evals = map(i -> maximum(rhat((ode_nuts[i:i+199,:,:])).nt.rhat), 1:1800)

# Check for first rhat < 1.05
# findlast(rhat_evals .>= 1.05)
# Gibbs 1745 (thin 20) warmup 1000

# Plots to aid in diagnosing issues with convergence
# loglikvals = logjoint(model_window, ode_nuts[240,:,:])
# histogram(loglikvals; normalize = :pdf)
# density!(loglikvals)

res = generated_quantities(
    model_window,
    ode_nuts[end,:,:]
)

I_tot_win = Array(res[1].sol(obstimes_window))[5,:,:,:]
X_win = rowadjdiff3d(I_tot_win)

Y_win_mu = mapslices(x->(conv_mat_window * (IFR_vec .* x)')', X_win, dims = (1,3))

a = Vector{Plots.Plot}(undef, 2 * NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(obstimes_window, Y[:,i,1:Window_size]', legend = false, alpha = 0.3)
    a[i+NR] = StatsPlots.plot(a[i],obstimes_window, eachrow(Y_win_mu[:,i,:]))
    a[i] = StatsPlots.plot(a[i],obstimes_window, eachrow(Y_mu[:,i,1:Window_size]))
end
StatsPlots.plot(a..., layout = (2,NR), size = (2800,1800))

# Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)

using Statistics

function generate_confint_infec_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 


    infecs = stack(map(x -> Array(x.sol)[3,:,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> Statistics.quantile(x,(1-cri) / 2), infecs, dims = 4)[:,:,:,1]
    medci_inf = mapslices(x -> Statistics.quantile(x, 0.5), infecs, dims = 4)[:, :, :, 1]
    uppci_inf = mapslices(x -> Statistics.quantile(x, cri + (1-cri) / 2), infecs, dims = 4)[:, :, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

# Create a function to take in the chains and evaluate the number of recovereds and summarise them (at a specific confidence level)
function generate_confint_recov_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 

    infecs = stack(map(x -> Array(x.sol)[4,:,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> Statistics.quantile(x,(1-cri) / 2), infecs, dims = 4)[:,:,:,1]
    medci_inf = mapslices(x -> Statistics.quantile(x, 0.5), infecs, dims = 4)[:,:,:,1]
    uppci_inf = mapslices(x -> Statistics.quantile(x, cri + (1-cri) / 2), infecs, dims = 4)[:,:,:,1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

I_dat = Array(sol_ode(obstimes))[3,:,:,:] # Population of infecteds at times
R_dat = Array(sol_ode(obstimes))[4,:,:,:] # Population of recovereds at times


# convert the above code into a function which evaluates the different beta get_beta_quantiles
# TODO Rewrite this function with new β names
function get_beta_quantiles(chn, K_window; quantile = 0.95, NR = NR)
    reg_names = ["EE", "LDN", "MID", "NEY", "NW", "SE", "SW"]
    β₀_syms = [Symbol("log_β₀_EE"),
               Symbol("log_β₀_LDN"),
               Symbol("log_β₀_MID"),
               Symbol("log_β₀_NEY"),
               Symbol("log_β₀_NW"),
               Symbol("log_β₀_SE"),
               Symbol("log_β₀_SW")]
    β_syms = [[Symbol("log_β_$reg_name[$j]") for reg_name in reg_names] for j in 1:K_window - 2]

    tst_arr = Array{Float64}(undef, K_window-1, NR, n_chains)

    for i in 1:NR
        tst_arr[1,i,:] = chn[:,β₀_syms[i],:]
        for j in 1:(K_window - 2)
            tst_arr[j+1,i,:] = chn[:,β_syms[j][i],:]
        end
    end

    tst_arr = exp.(cumsum(tst_arr, dims = 1))

    beta_μ = Array{Float64}(undef, K_window, NR)
    beta_lci = Array{Float64}(undef, K_window, NR)
    beta_uci = Array{Float64}(undef, K_window, NR)

    myindices =  [collect(1:K_window-1); K_window-1]

    for i in 1:NR
       for j in eachindex(myindices)
            k = myindices[j]
            beta_μ[j,i] = Statistics.quantile(tst_arr[k,i,:], 0.5)
            beta_lci[j,i] = Statistics.quantile(tst_arr[k,i,:], (1 - quantile) / 2)
            beta_uci[j,i] = Statistics.quantile(tst_arr[k,i,:], quantile + (1 - quantile) / 2)
        end
    end

   
    return (; beta_μ, beta_lci, beta_uci)
end

beta_μ, beta_lci, beta_uci = get_beta_quantiles(ode_nuts[end,:,:], K_window)

beta_func_init = (y,z) -> map(x -> ConstantInterpolation(x,z), eachcol(y))

beta_μ_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(beta_μ, knots_window)
beta_lci_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(beta_lci, knots_window)
beta_uci_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(beta_uci, knots_window)

beta_true_win = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(true_beta, knots)

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a_tmp = plot(obstimes_window,
        beta_μ_plot[i,:],
        ribbon = ((beta_μ_plot[i,:] - beta_lci_plot[i,:]), (beta_uci_plot[i,:] - beta_μ_plot[i,:])),
        xlabel = "Time",
        ylabel = "β",
        label="Using the NUTS algorithm",
        color=:blue,
        lw = 2,
        titlefontsize=18,
        guidefontsize=18,
        tickfontsize=16,
        legendfontsize=12,
        fillalpha = 0.4,
        legendposition = :none,
        margin = 4mm,
        bottom_margin = 4mm)
    a[i] = plot!(a_tmp, obstimes_window,
        beta_true_win[i,:],
        color=:red,
        label="True β",
        lw = 2)
end
plot(a..., layout = (2, Int(ceil(NR / 2))), plot_title = "Estimates of β",size = (1600,800), margin = 10mm, bottom_margin = 10mm)
savefig(string(outdir,"nuts_betas_window_1_$seed_idx.png"))

# Plot the infecteds
confint = generate_confint_infec_init(ode_nuts[end,:,:], model_window)

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    tmp_a = plot(confint.medci_inf[:,i,:]', ribbon = (confint.medci_inf[:,i,:]' - confint.lowci_inf[:,i,:]', confint.uppci_inf[:,i,:]' - confint.medci_inf[:,i,:]') , legend = false)
    a[i] = plot!(tmp_a, I_dat[:,i,1:Window_size]', linewidth = 2, color = :red)
end
plot(a..., plot_title = "Infectious (I) population",size = (1600,800), margin = 10mm, bottom_margin = 10mm)
savefig(string(outdir,"infections_nuts_window_1_$seed_idx.png"))

# Plot the recovereds
confint = generate_confint_recov_init(ode_nuts[end,:,:],model_window)
a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    tmp_a = plot(confint.medci_inf[:,i,:]', ribbon = (confint.medci_inf[:,i,:]' - confint.lowci_inf[:,i,:]', confint.uppci_inf[:,i,:]' - confint.medci_inf[:,i,:]'), legend = false)
    a[i] = plot!(R_dat[:,i,1:Window_size]', linewidth = 2, color = :red)
end
plot(a..., plot_title = "Infectious (R) population",size = (1600,800), margin = 10mm, bottom_margin = 10mm)
savefig(string(outdir,"recoveries_nuts_window_1_$seed_idx.png"))

# Write a funtion to return the ci for the hospitalisations
function generate_confint_hosps_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 

    hosps = stack(map(x -> x.y, chnm_res)[1:end])
    lowci_hosps = mapslices(x -> quantile(x,(1-cri) / 2), hosps, dims = 4)[:,:,:,1]
    medci_hosps = mapslices(x -> quantile(x, 0.5), hosps, dims = 4)[:,:,:,1]
    uppci_hosps = mapslices(x -> quantile(x, cri + (1-cri) / 2), hosps, dims = 4)[:,:,:,1]
    return (; lowci_hosps, medci_hosps, uppci_hosps)
end

# Plot the hospitalisations
confint = generate_confint_hosps_init(ode_nuts[end,:,:], model_window_unconditioned)

plot_obj_vec = Vector{Plots.Plot}(undef, NA * NR)
for i in 1:NA
    for j in 1:NR
        plot_part = scatter(1:Window_size, confint.medci_hosps[i,j,:], yerr = (confint.medci_hosps[i,j,:] .- confint.lowci_hosps[i,j,:], confint.uppci_hosps[i,j,:] .- confint.medci_hosps[i,j,:]), title = "NA: $i, NR: $j", legend = false)
        plot_obj_vec[(i-1) * NR + j] = scatter!(plot_part,1:Window_size, Y[i,j,1:Window_size], color = :red, legend = false)
    end
end
plot(plot_obj_vec..., layout = (NR * 2, 4), size = (2800,3400))

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
    lowci_sero = mapslices(x -> quantile(x,(1-cri) / 2), sero, dims = 4)[:,:,:,1]
    medci_sero = mapslices(x -> quantile(x, 0.5), sero, dims = 4)[:,:,:,1]
    uppci_sero = mapslices(x -> quantile(x, cri + (1-cri) / 2), sero, dims = 4)[:,:,:,1]
    # display(uppci_sero)
    lowci_sero[.!bitmatrix_nonzero] .= NaN
    medci_sero[.!bitmatrix_nonzero] .= NaN
    uppci_sero[.!bitmatrix_nonzero] .= NaN  
    return (; lowci_sero, medci_sero, uppci_sero)
end


plot_obj_vec = Vector{Plots.Plot}(undef, NA * NR)
confint = generate_confint_sero_init(ode_nuts[end,:,:], model_window_unconditioned, sus_pop_mask_window, sample_sizes_non_zero_window)
obs_exp_window = obs_exp[:,:,1:Window_size]
obs_exp_window[.!sus_pop_mask_window] .= NaN
# plot the Serological data
for i in 1:NA
    for j in 1:NR
        plot_part = scatter(obstimes_window, confint.medci_sero[i,j,:], yerr = (confint.medci_sero[i,j,:] .- confint.lowci_sero[i,j,:], confint.uppci_sero[i,j,:] .- confint.medci_sero[i,j,:]), title = "NA: $i, NR: $j", legend = false, ylims = (0,1))
        plot_obj_vec[(i-1) * NR + j] = scatter!(plot_part, obstimes_window, obs_exp_window[i,j,:], color = :red, legend = false)
    end
end
plot(plot_obj_vec..., layout = (NR * 2, 4), size = (2800,3400))


# Convert the samples to an array
# ode_nuts_arr = Array(ode_nuts)

# Create an array to store the window ends to know when to run the model
each_end_time = collect(Window_size:Data_update_window:tmax)
each_end_time = each_end_time[end] ≈ tmax ? each_end_time : vcat(each_end_time, tmax)
each_end_time = Integer.(each_end_time)

# Store the runtimes of each window period
algorithm_times = Vector{Float64}(undef, length(each_end_time))
algorithm_times[1] = runtime_init / 1e9

runtimes_precomp = Vector{Float64}(undef, length(each_end_time))
runtimes_precomp[1] = runtime_init_precomp /1e9

# Construct store for the resulting chains
list_chains = Vector{Chains}(undef, length(each_end_time))
list_chains[1] = ode_nuts

# Define function to determine the names of the parameters in the model for some number of betas

for idx_time_off_by_1 in eachindex(each_end_time[2:end])
    idx_time = idx_time_off_by_1 + 1
    
    # Determine which betas to fix and which ones to sample
    curr_t = each_end_time[idx_time]

    # Set the values for the current window: knots, K, convolution matrix...
    local knots_window = collect(0:Δ_βt:curr_t)
    local knots_window = knots_window[end] != curr_t ? vcat(knots_window, curr_t) : knots_window
    local K_window = length(knots_window)
    local conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, curr_t)
    local obstimes_window = 1.0:1.0:curr_t

    local sus_pop_mask_window = sus_pop_mask[:,:,1:curr_t]
    local sample_sizes_non_zero_window = @view sample_sizes[:,:,1:curr_t][sus_pop_mask_window]


    y_data_window = Y[:,:,1:curr_t]
    z_data_window = 💉[1:length(sample_sizes_non_zero_window)]

    # Construct the ODE Problem to be used 
    local tspan_window = (0.0, curr_t)
    local prob_ode_window = ODEProblem{true}(sir_tvp_ode!, u0, tspan_window, params_test);

    local model_window_unconditioned = bayes_sir_tvp(K_window,
        γ,
        σ,
        N_pop_R,
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
        prob_ode_window
    )

    local model_window = 
        model_window_unconditioned |
            (y = y_data_window,
                z = z_data_window,
            )

    # mydramgs = DRAMGS_Sampler(0.234, 0.6; noise_stabiliser = true, noise_stabiliser_scaling = 1e-5, max_delayed_rejection_steps = 2, rejection_step_sizes = [0.6, 0.1])
    if(isfile(string(outdir, "chains$idx_time.jld2")))
        local chain_file = JLD2.load(string(outdir, "chains$idx_time.jld2"))
        list_chains[idx_time] = chain_file["chains"]
        runtimes_precomp[idx_time] = chain_file["runtime_init_precomp"]
        algorithm_times[idx_time] = chain_file["runtime_init"]
    else
      # Get the time to compile the Sampler
      t1_precomp = time_ns()
      sample(
          model_window,
          global_regional_gibbs_mine,
          MCMCThreads(),
          3,
          n_chains,
          num_warmup = 2,
          thinning = 1,
          discard_initial = 0
      )
      t2_precomp = time_ns()
      runtimes_precomp[idx_time] = convert(Int64, t2_precomp - t1_precomp)

      t1 = time_ns()
      list_chains[idx_time] = sample(
          model_window,
          global_regional_gibbs_mine,
          MCMCThreads(),
          200,
          n_chains;
          discard_initial = discard_init[idx_time],
          # discard_initial = 0,
          # num_warmup = 1000,
          num_warmup = n_warmup_samples,
          thinning = 25
      )
      t2 = time_ns()
      algorithm_times[idx_time] = convert(Int64, t2 - t1)
       JLD2.save(string(outdir, "chains$idx_time.jld2"), "chains", list_chains[idx_time], "runtime_init_precomp", runtimes_precomp[idx_time], "runtime_init", algorithm_times[idx_time])
    end
    # est_lj = logjoint(model_window, list_chains[idx_time][:,:,:])

    # Correct complicated indices
    # l0 = list_chains[idx_time - 1].info.varname_to_symbol
    # l = list_chains[idx_time].info.varname_to_symbol

    # l_inv = OrderedDict(value => key for (key, value) in l)

    # for (k,v) in l0
    #     if haskey(l_inv,v)
    #         l_inv[v] = k
    #     end
    # end

    # list_chains[idx_time].info.varname_to_symbol = OrderedDict(value => key for (key, value) in l_inv)

    # Plot betas
    beta_win_μ, betas_win_lci, betas_win_uci = get_beta_quantiles(list_chains[idx_time][end,:,:], K_window)
    beta_win_μ_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(beta_win_μ, knots_window)
    beta_win_lci_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(betas_win_lci, knots_window)
    beta_win_uci_plot = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(betas_win_uci, knots_window)
    
    local beta_true_win = stack(repeat([collect(obstimes_window)], NR))' .|> beta_func_init(true_beta, knots)
    
    local a = Vector{Plots.Plot}(undef, NR)
    for i in 1:NR
        a_tmp = plot(obstimes_window,
            beta_win_μ_plot[i,:],
            ribbon = ((beta_win_μ_plot[i,:] - beta_win_lci_plot[i,:]), (beta_win_uci_plot[i,:] - beta_win_μ_plot[i,:])),
            xlabel = "Time",
            ylabel = "β",
            color=:blue,
            lw = 2,
            titlefontsize=18,
            guidefontsize=18,
            tickfontsize=16,
            legendfontsize=12,
            fillalpha = 0.4,
            legendposition = :none,
            margin = 4mm,
            bottom_margin = 4mm)
        a[i] = plot!(a_tmp, obstimes_window,
            beta_true_win[i,:],
            color=:red,
            label="True β",
            lw = 2)
    end
    plot(a..., layout = (2, Int(ceil(NR / 2))), plot_title = "Estimates of β",size = (1600,800), margin = 10mm, bottom_margin = 10mm)
    savefig(string(outdir,"β_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot infections

    # generated_quantities(model_window, list_chains[idx_time])
    local confint = generate_confint_infec_init(list_chains[idx_time][end,:,:], model_window)

    # a = Vector{Plots.Plot}(undef, NR)
    for i in 1:NR
        tmp_a = plot(confint.medci_inf[:,i,:]', ribbon = (confint.medci_inf[:,i,:]' - confint.lowci_inf[:,i,:]', confint.uppci_inf[:,i,:]' - confint.medci_inf[:,i,:]') , legend = false)
        a[i] = plot!(tmp_a, I_dat[:,i,1:curr_t]', linewidth = 2, color = :red)
    end
    plot(a..., plot_title = "Infectious (I) population",size = (1600,800), margin = 10mm, bottom_margin = 10mm)

    # Save the plot
    savefig(string(outdir,"infections_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot recoveries
    confint = generate_confint_recov_init(list_chains[idx_time], model_window)
    # a = Vector{Plots.Plot}(undef, NR)
    for i in 1:NR
        tmp_a = plot(confint.medci_inf[:,i,:]', ribbon = (confint.medci_inf[:,i,:]' - confint.lowci_inf[:,i,:]', confint.uppci_inf[:,i,:]' - confint.medci_inf[:,i,:]'), legend = false)
        a[i] = plot!(tmp_a, R_dat[:,i,1:curr_t]', linewidth = 2, color = :red)
    end
    plot(a..., plot_title = "Infectious (R) population",size = (1600,800), margin = 10mm, bottom_margin = 10mm)

    # Save the plot 
    savefig(string(outdir,"recoveries_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot hospitalisations
    confint = generate_confint_hosps_init(list_chains[idx_time], model_window_unconditioned)

    local plot_obj_vec = Vector{Plots.Plot}(undef, NA * NR)
    for i in 1:NA
        for j in 1:NR
            plot_part = scatter(1:curr_t, confint.medci_hosps[i,j,:], yerr = (confint.medci_hosps[i,j,:] .- confint.lowci_hosps[i,j,:], confint.uppci_hosps[i,j,:] .- confint.medci_hosps[i,j,:]), title = "NA: $i, NR: $j", legend = false)
            plot_obj_vec[(i-1) * NR + j] = scatter!(plot_part,1:Window_size, Y[i,j,1:curr_t], color = :red, legend = false)
        end
    end
    plot(plot_obj_vec..., layout = (NR * 2, 4), size = (2800,3400))

    # Save the plot
    savefig(string(outdir,"hospitalisations_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Plot serological data
    confint = generate_confint_sero_init(ode_nuts[end,:,:], model_window_unconditioned, sus_pop_mask_window, sample_sizes_non_zero_window)
    local obs_exp_window = obs_exp[:,:,1:curr_t]
    obs_exp_window[.!sus_pop_mask_window] .= NaN
    # plot the Serological data
    for i in 1:NA
        for j in 1:NR
            plot_part = scatter(obstimes_window, confint.medci_sero[i,j,:], yerr = (confint.medci_sero[i,j,:] .- confint.lowci_sero[i,j,:], confint.uppci_sero[i,j,:] .- confint.medci_sero[i,j,:]), title = "NA: $i, NR: $j", legend = false, ylims = (0,1))
            plot_obj_vec[(i-1) * NR + j] = scatter!(plot_part, obstimes_window, obs_exp_window[i,j,:], color = :red, legend = false)
        end
    end
    plot(plot_obj_vec..., layout = (NR * 2, 4), size = (2800,3400))

    # Save the plot
    savefig(string(outdir,"serological_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # Check logjoint
    loglikvals = logjoint(model_window, list_chains[idx_time])
    histogram(loglikvals; normalize = :pdf)


end



# knots_init = collect(0:Δ_βt:each_end_time[1])
# knots_init = knots_init[end] != each_end_time[1] ? vcat(knots_init, each_end_time[1]) : knots_init
# beta_μ, betas_lci, betas_uci = get_beta_quantiles(list_chains[1], length(knots_init))

# Sequentially create plots of beta estimates, overlapping previous windows
# for my_idx in 1:length(each_end_time)
#     plot(obstimes_window[1:Window_size],
#         beta_func(beta_μ, knots_window)(obstimes_window[1:Window_size]),
#         ribbon = (beta_func(beta_μ, knots_window)(obstimes_window[1:Window_size]) - beta_func(betas_lci, knots_window)(obstimes_window[1:Window_size]), beta_func(betas_uci, knots_window)(obstimes_window[1:Window_size]) - beta_func(beta_μ, knots_window)(obstimes_window[1:Window_size])),
#         xlabel = "Time",
#         ylabel = "β",
#         label="Window 1",
#         title="\nEstimates of β",
#         color=:blue,
#         xlimits=(0, each_end_time[end]),
#         lw = 2,
#         titlefontsize=18,
#         guidefontsize=18,
#         tickfontsize=16,
#         legendfontsize=12,
#         fillalpha = 0.4,
#         legendposition = :outerbottom,
#         # legendtitleposition = :left,
#         margin = 10mm,
#         bottom_margin = 0mm,
#         legend_column = min(my_idx + 1, 4))
#     if(my_idx > 1)
#         for idx_time in 2:my_idx
#             knots_plot = collect(0:Δ_βt:each_end_time[idx_time])
#             knots_plot = knots_plot[end] != each_end_time[idx_time] ? vcat(knots_plot, each_end_time[idx_time]) : knots_plot
#             beta_win_μ, betas_win_lci, betas_win_uci = get_beta_quantiles(list_chains[idx_time], length(knots_plot))
#             plot!(obstimes[1:each_end_time[idx_time]],
#                 beta_func(beta_win_μ, knots_plot)(obstimes[1:each_end_time[idx_time]]),
#                 ribbon = (beta_func(beta_win_μ, knots_plot)(obstimes[1:each_end_time[idx_time]]) - beta_func(betas_win_lci, knots_plot)(obstimes[1:each_end_time[idx_time]]), beta_func(betas_win_uci, knots_plot)(obstimes[1:each_end_time[idx_time]]) - beta_func(beta_win_μ, knots_plot)(obstimes[1:each_end_time[idx_time]])),
#                 # ylabel = "β",
#                 label="Window $idx_time",
#                 lw=2
#                 )
#         end
#     end 
#     plot!(obstimes,
#         betat_no_win(obstimes),
#         color=:red,
#         label="True β", lw = 2)
#     plot!(size = (1200,800))
#     plot!(collect(each_end_time[1:my_idx].-Window_size), seriestype="vline", color = :black, label = missing, linestyle = :dash, lw = 4)


#     savefig(string(outdir,"recoveries_nuts_window_combined","$my_idx","_$seed_idx","_95.png"))
# end

params = Dict(
    "algorithm_times" => algorithm_times,
    "compilation_times" => runtimes_precomp,
    # "algorithm_times_each_sample" => algorithm_times_each_sample
    )

open(string(outdir, "timings.toml"), "w") do io
        TOML.print(io, params)
end


# Save all chains
jldsave(string(outdir, "chains.jld2"), chains =  list_chains)
