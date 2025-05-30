using ProgressMeter: Distributed
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
using DynamicPPL
using Tullio
using SparseArrays
using AMGS
using JLD2
using Distributed

@assert length(ARGS) == 4 "Please provide the path to the TOML file as an argument."

# Load the TOML file
config_path = ARGS[1]
config = TOML.parsefile(config_path)

# Extract parameters from the TOML file
# Note: The TOML file should contain the necessary parameters for the SEIR model.
const seeds_list = config["seeds_list"]
const seeds_array_list = config["seeds_array_list"]
const Δ_βt = config["Delta_beta_t"]
const Window_size = config["Window_size"]
const n_chains = config["n_chains"]
const discard_init = config["discard_initial"]
const n_samples = config["n_samples"]
const n_warmup_samples = config["n_warmup"]
const tmax = config["tmax"]

# Get the data seed index from the command line arguments
const seed_idx  = parse(Int, ARGS[2])

# Get the seed for the current portion of the run
const seed_array_idx = parse(Int, ARGS[3])

# Set the random seed for reproducibility (of data)
Random.seed!(seeds_list[seed_idx])

const n_threads = Threads.nthreads()

const output_dir = string("Results/Bayescomp/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/seed $seed_idx/part $seed_array_idx/")

if !isdir(output_dir)
    mkpath(output_dir)
end

# Read the previous chain
const prev_chain_path = ARGS[4]
if isfile(string(output_dir,"$prev_chain_path"))
    prev_chain_dat = JLD2.jldopen(string(output_dir,"$prev_chain_path"), "r") do file
        (file["initial_sample"], file["Y"], file["💉"], file["d_I"], file["obstimes"])
    end
else
    prev_chain_dat = nothing
    @error "Previous chain file does not exist: $prev_chain_path"
end

prev_chain = prev_chain_dat[1]
Y = prev_chain_dat[2]
💉 = prev_chain_dat[3]
d_I = prev_chain_dat[4]
prev_max_obstime = maximum(prev_chain_dat[5])

const NA_N = [74103.0, 318183.0, 804260.0, 704025.0, 1634429.0, 1697206.0, 683583.0, 577399.0]
const NA = length(NA_N)
const N_pop = sum(NA_N)

const I0_μ_prior = -9.5
const trans_unconstrained_I0 = Bijectors.Logit(1.0 / N_pop, NA_N[5] / N_pop)

# Initialise the model parameters
const inv_γ = 10
const γ = 1 / inv_γ

const βσ = 0.15

const C = readdlm("ContactMats/england_8ag_contact_ldwk1_20221116_stable_household.txt") # Load the contact matrix

# Construct an ODE for the SEIR model
function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
    # Grab values needed for calculation
    @inbounds @views begin
        S = u[1,:]
        E1 = u[2,:]
        E2 = u[3,:]
        I1 = u[4,:]
        I2 = u[5,:]
    end
    (γ, σ, N_pop, a0) = p_.params_floats

    (infection, infectious_1, infectious_2, recovery_1, recovery_2, b) = p_.cache_SIRderivs
    βt = p_.cache_beta

    # Calculate the force of infection
    βt = p_.β_function(t) # This is a function that returns the β value at time t

    # Calculate the infection rate (using  β, a0
    @tullio grad = false threads = false  b[i] = (I1[j] + I2[j]) * log(1 - (a0 * 0.1 * βt *  p_.sus_M[i,j] * p_.C[j,i])) |> (1 - exp((_))) 

    infection .= S .* b
    infectious_1 .= 0.5 .* σ .* E1
    infectious_2  .= 0.5 .* σ .* E2
    recovery_1 .= 0.5 .* γ .* I1
    recovery_2 .= 0.5 .* γ .* I2
    @inbounds begin
        du[1,:] .= .- infection
        du[2,:] .= infection .- infectious_1
        du[3,:] .= infectious_1 .- infectious_2
        du[4,:] .= infectious_2 .- recovery_1
        du[5,:] .= recovery_1 .- recovery_2
        du[6,:] .= recovery_2
        du[7,:] .= infection
    end
end;

struct idd_params_struct{T <: Real, T3 <: DataInterpolations.AbstractInterpolation, T4 <: Real, T5 <: Real,T6 <: Real, T7 <: Real}
    params_floats::Vector{T}
    β_function::T3
    C::Matrix{T4}
    sus_M::Matrix{T5}
    cache_SIRderivs::Vector{Vector{T7}}
    cache_beta::T6
end

function adjdiff(ary::AbstractArray{T, 1}) where T
    ary1 = copy(ary)
    ary1[2:end] .-= ary1[1:end-1]
    return ary1
end

# Define a function to calculate the row-adjacent differences
function rowadjdiff(ary::AbstractArray{T, 2}) where T
    ary1 = copy(ary) 
    ary1[:, 2:end] .-= ary1[:,1:end-1]
    return ary1
end

# Function to construct Negative binomial with properties matching those in Birrell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1/(1+ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

# Define Gamma distribution by mean and variance
function Gamma_mean_sd_dist(μ, σ)
    α = (μ * μ) / (σ * σ)
    θ =  (σ * σ) / μ
    return Gamma(α, θ)
end

# Define helpful distributions (choice from sample in RTM)
const incubation_dist = Gamma_mean_sd_dist(4.0, 1.41)
const symp_to_hosp = Gamma_mean_sd_dist(9.0, 8.0666667)

# Define approximate convolution of Gamma distributions
# f: Distributions.Gamma x Distributions.Gamma -> Distributions.Gamma
function approx_convolve_gamma(d1::Gamma, d2::Gamma)
    μ_new = (d1.α * d1.θ) + (d2.α * d2.θ)
    σ_new = sqrt((d1.α * d1.θ * d1.θ) + (d2.α * d2.θ * d2.θ))
    return Gamma_mean_sd_dist(μ_new, σ_new)
end

const inf_to_hosp = approx_convolve_gamma(incubation_dist, symp_to_hosp)
inf_to_hosp_array_cdf = cdf(inf_to_hosp, 1:80)
inf_to_hosp_array_cdf = adjdiff(inf_to_hosp_array_cdf)

# Function which creates a matrix to calculate the discrete convolution (multiply convolution matrix by new infections vector to get the mean number of (eligible) new hospitalisations)

function construct_pmatrix(
    v::Vector{T},
    l) where T

    rev_v = @view v[end:-1:1]
    len_v = length(rev_v)
    ret_mat = zeros(T, l, l)

    for i in axes(ret_mat, 1)
        ret_mat[i, max(1, i - len_v + 1):min(i,l)] .= rev_v[max(1, len_v-i+1):end]
    end
    
    return sparse(ret_mat)
end

const IFR_vec = [0.0000078, 0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]

sample_sizes = readdlm("Serosamples_N/region_1.txt", Int64)
sample_sizes = permutedims(sample_sizes, (2, 1))

@model function bayescomp_model(
    K,
    γ,
    N_pop,
    NA,
    NA_N,
    conv_mat,
    knots,
    C,
    sero_sample_sizes,
    sero_pop_mask,
    obstimes,
    I0_μ_prior,
    βσ,
    IFR_vec,
    trans_unconstrained_I0,
    ODEProblem,
    ::Type{T_β} = Float64,
    ::Type{T_I} = Float64,
    ::Type{T_sus_M} = Float64,
    ::Type{T_SEIR} = Float64,
    ::Type{T_u0} = Float64
    ) where {T_β <: Real, T_I <: Real, T_sus_M <: Real, T_SEIR <: Real, T_u0 <: Real}
    # Prior I0
    logit_I₀ ~ Normal(I0_μ_prior, 0.2)

    #I0
    I = Bijectors.Inverse(trans_unconstrained_I0)(logit_I₀) * N_pop
    I_vec = zero(Vector{T_I}(undef, NA))
    I_vec[5] = I

    # Initial conditions
    u0 = zero(Array{T_u0}(undef, 7, NA))
    u0[1, :] = NA_N .- I_vec
    u0[4, :] = I_vec

    # overdispersion parameter η for the hospitalisations
    η ~ truncated(Gamma(1.0,0.2), upper = minimum(NA_N))

    # Initial growth rate parameter
    ψ ~ Gamma(20, 0.001)
    
    # Random walk for β (transmissibility)
    β = Vector{T_β}(undef, K)
    log_β = Vector{T_β}(undef, K-2)
    log_β₀ = zero(T_β)
    β[1] = exp(log_β₀)

    for i in 2:K-1
        log_β[i-1] ~ Normal(0.0, βσ)
        β[i] = exp(log(β[i-1]) + log_β[i-1])
    end
    β[K] = β[K-1] # Repeat the last value to match the length of knots

    # Specificity and sensitivity of seroprevalence tests
    sero_sens ~ Beta(71.5, 29.5)
    sero_spec ~ Beta(777.5, 9.5)

    #  Length of the infectious period
    d_I ~ Gamma(1.43, 0.549)
    inv_σ = d_I + 2
    σ = 1 / inv_σ
    
    inv_γ = 1 / γ

    # Susceptibility matrix
    sus_M = Matrix{T_sus_M}(undef, NA, NA)
    m_1 ~ truncated(Gamma(4,4), upper = 3.0)
    m_2 ~ truncated(Gamma(4,4), upper = 3.0)

    sus_M[1:3, :] .= m_1
    sus_M[4:7, :] .= 1.0
    sus_M[8, :] .= m_2

    if(any(β .> minimum(NA_N)))
        @DynamicPPL.addlogprob! -Inf
        return  # Return -Inf if the β values are too high
    end

    # Create the β function
    β_function = ConstantInterpolation(β, knots)

    # Create a0
    R0_num = 1 + (ψ * inv_σ * 0.5)
    R0_denom = 1 - (1 / ((1 + (ψ * inv_γ * 0.5))^2))
    a0_num = ψ * inv_γ * (R0_num^2) / R0_denom
    a0_denom = β_function(0.0) * 0.1 * inv_γ * eigmax(diagm(u0[1,:]) * (sus_M .* C))
    a0 = a0_num / a0_denom
    

    # Create the parameters struct
    infections_cache = zero(Vector{T_SEIR}(undef, NA))
    infectious_1_cache = zero(Vector{T_SEIR}(undef, NA))
    infectious_2_cache = zero(Vector{T_SEIR}(undef, NA))
    recovery_1_cache = zero(Vector{T_SEIR}(undef, NA))
    recovery_2_cache = zero(Vector{T_SEIR}(undef, NA))
    b_cache = zero(Vector{T_SEIR}(undef, NA))

    βt_cache = zero(T_SEIR)

    params_data = idd_params_struct{T_SEIR, DataInterpolations.AbstractInterpolation, eltype(C), T_sus_M, T_SEIR, T_β}(
        [γ, σ, N_pop, a0],
        β_function,
        C,
        sus_M,
        [infections_cache, infectious_1_cache, infectious_2_cache, recovery_1_cache, recovery_2_cache, b_cache],
        βt_cache
    )
    # Format print each parameter
    # Println("Parameters:")
    # Println("logit_I₀ = ", logit_I₀)
    # Println("I_vec = ", I_vec)
    # Println("η = ", η)
    # Println("ψ = ", ψ)
    # Println("m_1 = ", m_1)
    # Println("m_2 = ", m_2)
    # Println("d_I = ", d_I)
    # Println("inv_σ = ", inv_σ)
    # Println("inv_γ = ", inv_γ)
    # Println("sus_M = ", sus_M)
    # Println("β = ", β)
    # Println("log_β = ", log_β)
    # Println("sero_sens = ", sero_sens)
    # Println("sero_spec = ", sero_spec)
    # Println("a0 = ", a0)
    # Println("β_function = ", β_function)

    # Define the ODE problem
    #tspan = (zero(eltype(obstimes), obstimes[end])
    prob = remake(ODEProblem,
        u0 = u0,
        # tspan = tspan,
        p = params_data)

    sol = solve(prob,
        Tsit5(),
        saveat = obstimes,
        tstops = knots,
        d_discontinuities = knots)

    # Check if the solution is valid
    if any(!SciMLBase.successful_retcode(sol))
        @DynamicPPL.addlogprob! -Inf
        return  # Return -Inf if the solution is not valid
    end

    # Calculate the new infections and hospitalisations
    sol_X = stack(sol.u)[7,:,:] |> rowadjdiff

    if(any(sol_X .<= -1e-3) | any(stack(sol.u)[4,:,:] .<= -1e-3) | any(stack(sol.u)[5,:,:] .<= -1e-3))
        @DynamicPPL.addlogprob! -Inf
        return  # Return -Inf if any new infections are negative
    end

    y_μ = Array{T_SEIR}(undef, NA, length(obstimes))
    y_μ .= transpose(conv_mat * transpose(IFR_vec .* sol_X))

    # Check if the mean hospitalisations are valid
    if any(isnan.(y_μ))
        @DynamicPPL.addlogprob! -Inf
        return  # Return -Inf if the mean hospitalisations are negative
    end

    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, η))

    sus_pop = stack(map(x -> x[1,:], sol.u))
    sus_pop_samps = @view (sus_pop ./ NA_N)[sero_pop_mask]

    z ~ product_distribution(
        Binomial.(sero_sample_sizes,
            (sero_sens * (1 .- sus_pop_samps)) .+ ((1 .- sero_spec) .* sus_pop_samps))
    )

    return(; sol, y, z)

end

end_time = (prev_max_obstime + Δ_βt) > tmax ? tmax : (prev_max_obstime + Δ_βt)

knots_window = collect(0.0:Δ_βt:end_time)
knots_window = (knots_window[end] == end_time) ? knots_window : vcat(knots_window, end_time)
K_window = length(knots_window)
obstimes_window = 1.0:1.0:end_time
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window))
const sus_pop_mask = sample_sizes .!= 0
sus_pop_mask_window = sus_pop_mask[:, 1:length(obstimes_window)]
sample_sizes_non_zero_window = @view sample_sizes[:,1:length(obstimes_window)][sus_pop_mask_window]

sus_M = Matrix{Float64}(undef, NA, NA)
p = [zero(Float64), zero(Float64), N_pop, zero(Float64)]
β_func = ConstantInterpolation(zeros(Float64, K_window), knots_window)

my_infection = zeros(NA)
my_infectious_1 = zeros(NA)
my_infectious_2 = zeros(NA)
my_recovery_1 = zeros(NA)
my_recovery_2 = zeros(NA)
my_b = zeros(NA)

my_βt = zero(Float64)

params_data = idd_params_struct{Float64, DataInterpolations.AbstractInterpolation, Float64, Float64, Float64, Float64}(
    p,
    β_func,
    C,
    sus_M,
    [my_infection, my_infectious_1, my_infectious_2, my_recovery_1, my_recovery_2, my_b],
    my_βt
)

u0 = zeros(Float64, 7, NA)

# Initialise the ODE problem for the window
prob_ode_window = ODEProblem{true}(sir_tvp_ode!, u0, (0.0, end_time), params_data);

# Define the model
model_window_unconditioned = bayescomp_model(
    K_window,
    γ,
    N_pop,
    NA,
    NA_N,
    conv_mat_window,
    knots_window,
    C,
    sample_sizes_non_zero_window,
    sus_pop_mask_window,
    obstimes_window,
    I0_μ_prior,
    βσ,
    IFR_vec,
    trans_unconstrained_I0,
    prob_ode_window)

model_window = model_window_unconditioned | (y = Y[:, 1:length(obstimes_window)],
z = 💉[1:length(sample_sizes_non_zero_window)],)

# @code_warntype model_window.f(
#     model_window,
#     Turing.VarInfo(model_window),
#     Turing.DefaultContext(),
#     model_window.args...)
#

const myamgs_global = AMGS_Sampler(0.234, 0.6, true, 1e-5)
const myamgs_region = AMGS_Sampler(0.234, 0.6, true, 1e-5)

const global_varnames = (@varname(η), @varname(m_1), @varname(m_2), @varname(sero_sens), @varname(sero_spec)) 
const region_varnames = (@varname(logit_I₀), @varname(ψ), @varname(log_β))

const my_gibbs = Gibbs(
    global_varnames => myamgs_global,
    region_varnames => myamgs_region)

model_window_fixed = DynamicPPL.fix(model_window, (d_I = d_I,))

initial_sample = sample(model_window_fixed, my_gibbs, MCMCThreads(), 100, 1; num_warmup = 10, discard_initial = 0, progress = true, model_check = true)

# Write the timing callback
Base.@kwdef struct Timings{A}
    times::A = Vector{UInt64}()
end

function (callback::Timings)(
    rng,
    model,
    sampler,
    sample,
    state,
    iteration;
    kwargs...,
    )
    time = time_ns()  # Get the current time in nanoseconds
    if iteration == 1
        # Initialize the times vector on the first iteration
        empty!(callback.times)
    end
    
    # Store the time for this iteration
    push!(callback.times, time)
    return nothing
end

callback = Timings([])

# write the parallel loop
@assert  Threads.nthreads() >= n_chains
n_chunks = n_chains
interval = 1:n_chunks
rngs = [deepcopy(Random.default_rng()) for _ in interval]
models = [deepcopy(model_window_fixed) for _ in interval]
samplers = [deepcopy(my_gibbs) for _ in interval]
callbacks = [deepcopy(callback) for _ in interval]

all_varnames = collect(keys(initial_sample.info.varname_to_symbol))
prev_chain_varnames = collect(keys(prev_chain.info.varname_to_symbol))

fixed_dict = Dict{Turing.VarName, Any}()

prev_ests_params = map(i -> 
            merge(
                Dict(Symbol(name) in Symbol.(prev_chain_varnames) ?
                            (name, prev_chain[i,:,end][Symbol(name)][1]) :
                            (name, missing)
                    for name ∈ all_varnames
                ),
               fixed_dict 
            ),
        1:n_chains)

chains = Vector{Any}(undef,n_chains)
Random.seed!(seeds_array_list[seed_array_idx])
seeds = rand(Random.default_rng(), UInt, n_chains)

init_time = time_ns()
Distributed.@sync begin
    Distributed.@async begin
        Distributed.@sync for (i, _rng, _model, _sampler, _callback) in
            zip(1:n_chunks, rngs, models, samplers, callbacks)
            Threads.@spawn begin
                Random.seed!(_rng, seeds[i])
                local init_vals = sample(
                            DynamicPPL.fix(_model, prev_ests_params[i]),
                            Prior(),
                            1;
                            progress = false
                        )
                local init_params_dict = merge(
                    Dict((name, prev_chain[i,:,end][Symbol(name)][1]) for name ∈ prev_chain_varnames),
                        Dict(name => init_vals[Symbol(name)][1] for name ∈ keys(init_vals.info.varname_to_symbol))
                )
                local use_init_params = [
                    init_params_dict[name] for name in all_varnames
                ]
                @inbounds chains[i] = sample(
                    _rng,
                    _model,
                    _sampler,
                    MCMCSerial(),
                    n_samples,
                    1;
                    num_warmup = n_warmup_samples,
                    discard_initial = discard_init,
                    progress = false,
                    initial_params = [use_init_params,],
                    callback = _callback
                )
            end
        end
    end
end

using AbstractMCMC
combined_chains = AbstractMCMC.chainsstack(AbstractMCMC.tighten_eltype(chains))


# logliks = logjoint(model_window_fixed, initial_sample)

JLD2.jldsave(string(output_dir, "chain_no_sw_$(Integer(tmax))_$(Integer(maximum(obstimes_window))).jld2"), initial_sample = combined_chains, Y = Y, 💉 = 💉, d_I = d_I, timings = callbacks, init_time = init_time, obstimes = obstimes_window)


# To do
# Split the code into two
#  Section 1: Runs model with 4 chains and samples every n iterations
#  Section 2: Runs model with 25 chains (depends on the array of seeds index)
#
#  Note also need to change code to make use of the Timings object we were interested in testing
#  Note make sure works on the HPC
#
#  Note (maybe just write the practical filtering code manually... use the code from the sampler to help write this)
#  Worth testing convergence????
#
#  Note need to do this for multiple example chains
#  Also remember to use starting points for both sets of chains
