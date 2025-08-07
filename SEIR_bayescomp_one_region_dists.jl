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


# Load the TOML file
config_path = "my_model.toml"
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

seed_idx = 1

# Set the random seed for reproducibility (of data)
Random.seed!(seeds_list[seed_idx])

const n_threads = Threads.nthreads()

output_dir = string("Results/Bayescomp_cc/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/seed $seed_idx/")

# Initialise the model parameters
const tspan = (0.0, tmax)
const obstimes = 1.0:1.0:tmax
const NA_N = [74103.0, 318183.0, 804260.0, 704025.0, 1634429.0, 1697206.0, 683583.0, 577399.0]
const NA = length(NA_N)
const N_pop = sum(NA_N)

const I0_μ_prior = -9.5
const trans_unconstrained_I0 = Bijectors.Logit(1.0 / N_pop, NA_N[5] / N_pop)

const I0_sv = rand(Normal(I0_μ_prior, 0.5)) |> Bijectors.Inverse(trans_unconstrained_I0) |> x -> x * N_pop

I0 = zeros(NA)
I0[5] = I0_sv
u0 = zeros(7, NA)
u0[1, :] = NA_N .- I0
u0[4,:] = I0

const inv_γ = 10
const inv_σ = 3

const γ = 1 / inv_γ
const σ = 1 / inv_σ

const ψ_dist = Gamma(20, 0.001)
const ψ = rand(ψ_dist)

const m_base = 1.0
const m_1 = 0.6
const m_2 = 0.4

sus_M = Matrix{Float64}(undef, NA, NA)

@assert NA == 8 "The number of age groups (NA) must be 8."
sus_M[1:3, :] .= m_1
sus_M[4:7, :] .= m_base
sus_M[8, :] .= m_2

# Set parameters for inference and draw betas from the prior
const βσ = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax / Δ_βt))+1)
true_beta[1] = 1.0

for i in 2:(length(true_beta)-1)
    true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0, βσ)))
end
true_beta[end] = true_beta[end-1] # Remove the last value to match the length of knots

knots = collect(0.0:Δ_βt:tmax)
knots = (knots[end] == tmax) ? knots : vcat(knots, tmax)

const K = length(knots)
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

# Define the initial R0
R0_num = 1 + (ψ * inv_σ * 0.5)
R0_denom = 1 - ( 1 / ((1 + (ψ * inv_γ * 0.5))^2))
a0_num = ψ * inv_γ * (R0_num^2) / R0_denom

β_func = ConstantInterpolation(true_beta, knots)

a0_denom = β_func(0.0) * 0.1 * inv_γ * eigmax(diagm(u0[1,:]) * (sus_M .* C))

a0 = a0_num / a0_denom

p = [γ, σ, N_pop, a0];


struct idd_params_struct{T <: Real, T3 <: DataInterpolations.AbstractInterpolation, T4 <: Real, T5 <: Real,T6 <: Real, T7 <: Real}
    params_floats::Vector{T}
    β_function::T3
    C::Matrix{T4}
    sus_M::Matrix{T5}
    cache_SIRderivs::Vector{Vector{T7}}
    cache_beta::T6
end

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

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem{true}(sir_tvp_ode!, u0, tspan, params_data);

# Define the ODE solver and solve the problem
const sol = solve(prob_ode, Tsit5(), saveat=obstimes, tstops = knots, d_discontinuities = knots);

# Plot the infection curve
plot(stack(map(x -> x[4,:] + x[5,:], sol.u))')

# Find the cumulative number of infections
const I_dat = Array(sol(obstimes))[4,:,:] + Array(sol(obstimes))[5,:,:]
const R_dat = Array(sol(obstimes))[6,:,:]
const I_dat_2 = Array(sol(obstimes))[7,:,:]

# Define a function to calculate the adjacent differences
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

const X = rowadjdiff(I_dat_2)

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
    v::Vector{T} = inf_to_hosp_array_cdf,
    l = length(obstimes)) where T

    rev_v = @view v[end:-1:1]
    len_v = length(rev_v)
    ret_mat = zeros(T, l, l)

    for i in axes(ret_mat, 1)
        ret_mat[i, max(1, i - len_v + 1):min(i,l)] .= rev_v[max(1, len_v-i+1):end]
    end
    
    return sparse(ret_mat)
end

const IFR_vec = [0.0000078, 0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]

# Evaluate the mean number of hospitalisations
const conv_mat = construct_pmatrix(;)
const Y_mu = (conv_mat * (IFR_vec .* X)')'

# Function to construct Negative binomial with properties matching those in Birrell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1/(1+ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

const η = rand(Gamma(1.0, 0.2)) # Randomly sample η from a Gamma distribution

# Draw sample of hospitalisations from the Negative Binomial distribution
const Y = @. rand(NegativeBinomial3(Y_mu + 1e-3, η))

# Plot mean hospitalisations and sampled hospitalisations
StatsPlots.scatter(obstimes, Y', label="Sampled Hospitalisations", markersize=2, legend = false, alpha = 0.3)
StatsPlots.plot!(obstimes, Y_mu', label="Mean Hospitalisations", linewidth=2, color=:red)
plot!(size = (1200, 800))

const sero_sens = 0.7659149
const sero_spec = 0.9430569

sample_sizes = readdlm("Serosamples_N/region_1.txt", Int64)

sample_sizes = sample_sizes[1:length(obstimes), :]
sample_sizes = permutedims(sample_sizes, (2, 1))

const sus_pop = stack(map(x -> x[1,:], sol(obstimes)))
const sus_pop_mask = sample_sizes .!= 0
const sus_pop_samps = @view (sus_pop ./ NA_N)[sus_pop_mask]
const sample_sizes_non_zero = @view sample_sizes[sus_pop_mask]

const 💉 = @. rand(
    Binomial(
        sample_sizes_non_zero,
        (sero_sens * (1 - (sus_pop_samps))) + ((1-sero_spec) * (sus_pop_samps))
    )
)

obs_exp = zeros(NA, length(obstimes))

obs_exp[sus_pop_mask] = 💉 ./ sample_sizes_non_zero

StatsPlots.scatter(1:length(obstimes), obs_exp', legend = false, alpha = 0.3)
plot!(size = (1200, 800), label="Seroprevalence", markersize=2, color=:blue)

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
    # println("Parameters:")
    # println("logit_I₀ = ", logit_I₀)
    # println("I_vec = ", I_vec)
    # println("η = ", η)
    # println("ψ = ", ψ)
    # println("m_1 = ", m_1)
    # println("m_2 = ", m_2)
    # println("d_I = ", d_I)
    # println("inv_σ = ", inv_σ)
    # println("inv_γ = ", inv_γ)
    # println("sus_M = ", sus_M)
    # println("β = ", β)
    # println("log_β = ", log_β)
    # println("sero_sens = ", sero_sens)
    # println("sero_spec = ", sero_spec)
    # println("a0 = ", a0)
    # println("β_function = ", β_function)

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

knots_window = collect(0.0:Δ_βt:Window_size)
knots_window = (knots_window[end] == Window_size) ? knots_window : vcat(knots_window, Window_size)
K_window = length(knots_window)
obstimes_window = 1.0:1.0:Window_size
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window))

sus_pop_mask_window = sus_pop_mask[:, 1:length(obstimes_window)]
sample_sizes_non_zero_window = @view sample_sizes[:,1:length(obstimes_window)][sus_pop_mask_window]

# Initialise the ODE problem for the window
prob_ode_window = ODEProblem{true}(sir_tvp_ode!, u0, (0.0, Window_size), params_data);

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

model_window_fixed = DynamicPPL.fix(model_window, (d_I = inv_σ - 2,))

# Write the timing callback
Base.@kwdef struct Timings{A}
    times::A = Vector{UInt64}()
end

nopf = JLD2.load("chain_no_sw_400_260.jld2")

obstimes_window = nopf["obstimes"]
end_time = obstimes_window[end]

knots_window = collect(0.0:Δ_βt:end_time)
knots_window = (knots_window[end] == end_time) ? knots_window : vcat(knots_window, end_time)
K_window = length(knots_window)
obstimes_window = 1.0:1.0:end_time
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window))

sus_pop_mask_window = sus_pop_mask[:, 1:length(obstimes_window)]
sample_sizes_non_zero_window = @view sample_sizes[:,1:length(obstimes_window)][sus_pop_mask_window]

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

model_window_fixed = DynamicPPL.fix(model_window, (d_I = inv_σ - 2,))


nopf_times_data =Int64.(mapreduce(x -> x.times, hcat, nopf["timings"]) .- nopf["init_time"]) / 1e9   # Load a subset

base_chain = nopf["initial_sample"]

res_ld = logjoint(model_window_fixed, base_chain[1:10:10000,:,:])

alt_init_data = jldopen("chain_no_sw_400_260.jld2", "r") do f
    f["initial_sample"][1:1:700,:,:]  # Load a subset
end;

input_dir = "Results/Bayescomp_cc/40_beta/window_220/chains_10/n_threads_20/seed 1/"  # Change this if needed

# List and filter part directories like "part 1", "part 2", ...
all_dirs = filter(entry -> isdir(joinpath(input_dir, entry)) &&
                            occursin(r"^part \d+$", entry),
                  readdir(input_dir))

# Sort by number after "part "
sorted_dirs = sort(all_dirs, by = dir -> parse(Int, match(r"^part (\d+)$", dir).captures[1]))

# Preload one to get type of `initial_samples`
first_path = joinpath(input_dir, sorted_dirs[1], "chain_sw_400_260.jld2")
first_data = jldopen(first_path, "r") do f
    f["initial_sample"][1:10:10000,:,:]  # Load a subset
end;

T = typeof(first_data)
data_array = Vector{T}(undef, length(sorted_dirs))
data_array[1] = first_data

# Load the rest
for (i, dir) in enumerate(sorted_dirs[2:end])
    file_path = joinpath(input_dir, dir, "chain_sw_400_260.jld2")
    data_array[i+1] = jldopen(file_path, "r") do f
        f["initial_sample"][1:10:10000,:,:]  # Load a subset 
    end;
end

first_times_data = jldopen(first_path, "r") do f
    Int64.(mapreduce(x -> x.times, hcat, f["timings"]) .- f["init_time"]) / 1e9   # Load a subset
end;

T_times = typeof(first_times_data)
times_array = Vector{T_times}(undef, length(sorted_dirs))
times_array[1] = first_times_data

# Load the rest
for (i, dir) in enumerate(sorted_dirs[2:end])
    file_path = joinpath(input_dir, dir, "chain_sw_400_260.jld2")
    times_array[i+1] = jldopen(file_path, "r") do f
        Int64.(mapreduce(x -> x.times, hcat, f["timings"]) .- f["init_time"]) / 1e9   # Load a subset   
    end;
end


using AbstractMCMC

combined_samples = AbstractMCMC.chainsstack(AbstractMCMC.tighten_eltype(data_array))
ests_ld = logjoint(model_window_fixed, combined_samples)

# Create the first plot without a legend
p1 = plot(1:10:10000,res_ld, label = "", legend = false)

# Create the second plot, also without a legend
p2 = plot(1:10:10000,ests_ld, label = "", legend = false)

# Synchronize axes by setting the same xlims and ylims
ylims = (min(minimum(ests_ld[100:end,:]), minimum(res_ld[100:end,:])), max(maximum(ests_ld[100:end,:]), maximum(res_ld[100:end,:])))

plot!(p2, ylim = ylims, title = "Sliding Window (sample) Chains", alpha = 0.2, lw = 2)
plot!(p1, ylim = ylims, title = "Joint MCMC (Convergence) Chains", alpha = 0.3, lw = 2)

# Display side-by-side
plot(p1, p2, layout = (1,2))
savefig("out/Convergence_plots.png")


default(size=(1200, 800), legendfontsize=10, guidefontsize=12, tickfontsize=10)

# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld[end, :],
    color = :blue,
    lw = 3,  # thicker line
    label = "Sliding Window Method (10000 iterations)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(res_ld[400:end, :]))
    density!(p, row,
        color = :red,
        lw = 2,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (Convergence)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box)

savefig("out/End_Sliding_Window_plots.png")

init_ests_ld = logjoint(model_window_fixed, alt_init_data)


# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld[70, :],
    color = :blue,
    lw = 3,  # thicker line
    label = "Sliding Window Method (700 iterations)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(init_ests_ld[501:end, :]))
    density!(p, row,
        color = :red,
        lw = 2,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (700 iterations)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box)

savefig("out/Early_Non_Sliding_Window_plots.png")

# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld[70, :],
    color = :blue,
    lw = 3,  # thicker line
    label = "Sliding Window Method (700 iterations = 51.5 seconds)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(init_ests_ld[447:647, :]))
    density!(p, row,
        color = :red,
        lw = 2,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (647 iterations = 51.5 seconds)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box)

savefig("out/Early_Non_Sliding_Window_plots_same_time.png")



function get_beta_quantiles_sw(chn, knots, obstimes, n_betas, iteration)
    @assert n_betas > 1 "There must be at least two β values to calculate quantiles using this function."
    # Extract the β values from the chain
    β_samples = Array{Float64}(undef, n_betas, size(chn)[3])
    for i in 2:n_betas
        β_samples[i, :] = chn[Symbol("log_β[$(i-1)]")][iteration,:]
    end
    # Hopefully this is the right size
    β_samples[1, :] .= 0

    cumsum_β = cumsum(β_samples, dims=1)
    βs = exp.(cumsum_β)
    # Calculate the number of time points
    n_time_points = length(obstimes)

    # Initialize an array to hold the quantiles
    quantiles = zeros(n_betas+1, 3)  # 3 quantiles: 0.025, 0.5, 0.975

    # Calculate the quantiles for each time point
    for i in 2:n_betas
        quantiles[i, :] = quantile(βs[i, :], [0.025, 0.5, 0.975])
    end

    quantiles[1, :] .= 1
    quantiles[n_betas+1, :] .= quantiles[n_betas, :]
    

    # Write the beta quantiles into ConstantInterpolation functions using obstimes
    return (quantiles,
        ConstantInterpolation(quantiles[:, 2], knots),
        ConstantInterpolation(quantiles[:, 1], knots),
        ConstantInterpolation(quantiles[:, 3], knots)
    )
end


function get_beta_quantiles_no_sw(chn, knots, obstimes, n_betas, iterations)
    @assert n_betas > 1 "There must be at least two β values to calculate quantiles using this function."
    # Extract the β values from the chain
    β_samples = Array{Float64}(undef, n_betas, length(iterations))
    for i in 2:n_betas
        β_samples[i, :] = chn[Symbol("log_β[$(i-1)]")][iterations,1]
    end
    # Hopefully this is the right size
    β_samples[1, :] .= 0

    cumsum_β = cumsum(β_samples, dims=1)
    βs = exp.(cumsum_β)
    # Calculate the number of time points
    n_time_points = length(obstimes)

    # Initialize an array to hold the quantiles
    quantiles = zeros(n_betas+1, 3)  # 3 quantiles: 0.025, 0.5, 0.975

    # Calculate the quantiles for each time point
    for i in 2:n_betas
        quantiles[i, :] = quantile(βs[i, :], [0.025, 0.5, 0.975])
    end

    quantiles[1, :] .= 1
    quantiles[n_betas+1, :] .= quantiles[n_betas, :]
    

    # Write the beta quantiles into ConstantInterpolation functions using obstimes
    return (quantiles,
        ConstantInterpolation(quantiles[:, 2], knots),
        ConstantInterpolation(quantiles[:, 1], knots),
        ConstantInterpolation(quantiles[:, 3], knots)
    )
end



res_betas = get_beta_quantiles_sw(combined_samples, knots_window, obstimes_window, K_window-1, 70)
res_betas_no_sw = get_beta_quantiles_no_sw(base_chain[:,:,1], knots_window, obstimes_window, K_window-1, 98000:100000)

β_func = ConstantInterpolation(true_beta, knots)

# Plot the true β values and the quantiles
plot(obstimes_window, res_betas[2].(obstimes_window),ribbon = (res_betas[2].(obstimes_window)-res_betas[3].(obstimes_window),res_betas[4].(obstimes_window)-res_betas[2].(obstimes_window)), label = "Sliding Window β (700 iterations)", color = :blue, linewidth = 2)

# plot!(obstimes_window, res_betas_no_sw[2].(obstimes_window),ribbon = (res_betas_no_sw[2].(obstimes_window)-res_betas_no_sw[3].(obstimes_window),res_betas_no_sw[4].(obstimes_window)-res_betas_no_sw[2].(obstimes_window)), label = "Adaptive MH β", color = :green, linewidth = 2, alpha = 0.5)

plot!(obstimes_window, β_func.(obstimes_window), label = "True β", color = :red, linewidth = 2)
savefig("out/Sliding_Window_beta_quantiles.png")

nopf2 = JLD2.load(string(input_dir,"part 1/chain_no_sw_400_300.jld2"))

obstimes_window2 = nopf2["obstimes"]
end_time2 = obstimes_window2[end]

knots_window2 = collect(0.0:Δ_βt:end_time2)
knots_window2 = (knots_window[end] == end_time2) ? knots_window2 : vcat(knots_window2, end_time2)
K_window2 = length(knots_window2)
obstimes_window2 = 1.0:1.0:end_time2
conv_mat_window2 = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window2))

sus_pop_mask_window2 = sus_pop_mask[:, 1:length(obstimes_window2)]
sample_sizes_non_zero_window2 = @view sample_sizes[:,1:length(obstimes_window2)][sus_pop_mask_window2]

# Initialise the ODE problem for the window
prob_ode_window2 = ODEProblem{true}(sir_tvp_ode!, u0, (0.0, end_time2), params_data);

# Define the model
model_window_unconditioned2 = bayescomp_model(
    K_window2,
    γ,
    N_pop,
    NA,
    NA_N,
    conv_mat_window2,
    knots_window2,
    C,
    sample_sizes_non_zero_window2,
    sus_pop_mask_window2,
    obstimes_window2,
    I0_μ_prior,
    βσ,
    IFR_vec,
    trans_unconstrained_I0,
    prob_ode_window2)

model_window2 = model_window_unconditioned2 | (y = Y[:, 1:length(obstimes_window2)],
z = 💉[1:length(sample_sizes_non_zero_window2)],)

model_window_fixed2 = DynamicPPL.fix(model_window2, (d_I = inv_σ - 2,))


nopf_times_data2 =Int64.(mapreduce(x -> x.times, hcat, nopf2["timings"]) .- nopf2["init_time"]) / 1e9   # Load a subset

base_chain2 = nopf2["initial_sample"]

res_ld2 = logjoint(model_window_fixed2, base_chain2[1:10:10000,:,:])
alt_res_ld2 = logjoint(model_window_fixed2, base_chain2[14000:10:20000,:,:])

alt_init_data2 = jldopen(string(input_dir,"part 1/chain_no_sw_400_300.jld2"), "r") do f
    f["initial_sample"][1200:1:1600,:,:]  # Load a subset
end;

# Preload one to get type of `initial_samples`
first_path2 = joinpath(input_dir, sorted_dirs[1], "chain_sw_400_300.jld2")
first_data2 = jldopen(first_path2, "r") do f
    f["initial_sample"][1:10:10000,:,:]  # Load a subset
end;

T2 = typeof(first_data2)
data_array2 = Vector{T}(undef, length(sorted_dirs))
data_array2[1] = first_data2

# Load the rest
for (i, dir) in enumerate(sorted_dirs[2:end])
    file_path = joinpath(input_dir, dir, "chain_sw_400_300.jld2")
    data_array2[i+1] = jldopen(file_path, "r") do f
        f["initial_sample"][1:10:10000,:,:]  # Load a subset 
    end;
end

first_times_data2 = jldopen(first_path2, "r") do f
    Int64.(mapreduce(x -> x.times, hcat, f["timings"]) .- f["init_time"]) / 1e9   # Load a subset
end;

T_times2 = typeof(first_times_data2)
times_array2 = Vector{T_times}(undef, length(sorted_dirs))
times_array2[1] = first_times_data2

# Load the rest
for (i, dir) in enumerate(sorted_dirs[2:end])
    file_path = joinpath(input_dir, dir, "chain_sw_400_300.jld2")
    times_array2[i+1] = jldopen(file_path, "r") do f
        Int64.(mapreduce(x -> x.times, hcat, f["timings"]) .- f["init_time"]) / 1e9   # Load a subset   
    end;
end

combined_samples2 = AbstractMCMC.chainsstack(AbstractMCMC.tighten_eltype(data_array2))
ests_ld2 = logjoint(model_window_fixed2, combined_samples2)

# Create the first plot without a legend
p1 = plot(1:10:10000,res_ld2, label = "",seriestype = :line, legend = false, ylabel = "Log Posterior Density",linewidth = 3.0, alpha = 0.5)

# Create the second plot, also without a legend
p2 = plot(1:10:10000,ests_ld2, label = "", legend = false, yticks = ([],[]), linewidth = 3.0, alpha = 0.3)

# Synchronize axes by setting the same xlims and ylims
ylims = (min(minimum(ests_ld2[100:end,:]), minimum(res_ld2[150:end,:])), max(maximum(ests_ld2[100:end,:]), maximum(res_ld2[150:end,:])))

plot!(p2, ylim = ylims, title = "Sliding Window Chains", alpha = 0.2, linewidth = 10.5)
plot!(p1, ylim = ylims, title = "Joint MCMC Chains", alpha = 0.2, linewidth = 10.5)

# Display side-by-side
plot(p1, p2, layout = (1,2))
plot!(right_margin = 10mm, left_margin = 5mm,)
plot!(titlefontsize = 24)
plot!( xlabel = "Iteration", framestyle = :box)
plot!(markersize = 4)
plot!(size = (1200,1000))
savefig("out/Convergence_plots_2.png")

default(size=(1200, 1000))
default(legendfontsize=10, guidefontsize=12, tickfontsize=10)

# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld2[70, :],
    color = :blue,
    lw = 4,  # thicker line
    label = "Sliding Window (700 iterations)",
    alpha = 1.0
)

density!(p, ests_ld2[500, :],
    color = "#80c000",
    lw = 4,  # thicker line
    label = "Sliding Window (5,000 iterations)",
    alpha = 1.0
)

density!(p, ests_ld2[end, :],
    color = "#800080",
    lw = 4,  # thicker line
    label = "Sliding Window (10,000 iterations)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(alt_res_ld2))
    density!(p, row,
        color = :red,
        lw = 3.5,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (Convergence)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box, ylabel = "Sample Density", xlabel = "Log Posterior Density")
plot!(left_margin = 5mm, right_margin = 12mm, bottom_margin = -15mm, legend_columns = 2, legendfontsize = 18)

savefig("out/alt_Early_Sliding_Window_plots_2.png")

init_ests_ld2 = logjoint(model_window_fixed2, alt_init_data2)

alt_alt_init_data2 = jldopen(string(input_dir,"part 1/chain_no_sw_400_300.jld2"), "r") do f
    f["initial_sample"][500:1:700,:,:]  # Load a subset
end;

alt_init_ests_ld2 = logjoint(model_window_fixed2, alt_alt_init_data2)

# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld2[70, :],
    color = :blue,
    lw = 4,  # thicker line
    label = "Sliding Window Method (700 iterations)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(alt_init_ests_ld2))
    density!(p, row,
        color = :red,
        lw = 3.5,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (700 iterations)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box)

savefig("out/Early_Non_Sliding_Window_plots2.png")

# Create the figure
p = plot()

# Plot the single density curve (Sliding Window)
density!(p, ests_ld2[70, :],
    color = :blue,
    lw = 4,  # thicker line
    label = "Sliding Window (700 iterations = 157.15 seconds)",
    alpha = 1.0
)

# Plot multiple red density curves from res_ld, but only label the first one
for (i, row) in enumerate(eachcol(init_ests_ld2))
    density!(p, row,
        color = :red,
        lw = 3.5,
        alpha = 0.3,
        label = i == 1 ? "Joint MCMC (1600 iterations = 157.15 seconds)" : ""
    )
end

# Final tweaks
plot!(p, legend = :outerbottom, framestyle = :box, ylabel = "Sample Density", xlabel = "Log Posterior Density")
plot!(left_margin = 5mm, bottom_margin = -15mm)

savefig("out/Early_Non_Sliding_Window_plots_same_time2.png")

res_betas2 = get_beta_quantiles_sw(combined_samples2, knots_window2, obstimes_window2, K_window2-1, 70)
res_betas_no_sw2 = get_beta_quantiles_no_sw(base_chain2[:,:,1], knots_window2, obstimes_window2, K_window2-1, 98000:100000)

β_func = ConstantInterpolation(true_beta, knots)

# Plot the true β values and the quantiles
plot(obstimes_window2, res_betas2[2].(obstimes_window2),ribbon = (res_betas2[2].(obstimes_window2)-res_betas2[3].(obstimes_window2),res_betas2[4].(obstimes_window2)-res_betas2[2].(obstimes_window2)), label = "Sliding Window β (700 iterations)", color = :blue, linewidth = 2)

# plot!(obstimes_window, res_betas_no_sw[2].(obstimes_window),ribbon = (res_betas_no_sw[2].(obstimes_window)-res_betas_no_sw[3].(obstimes_window),res_betas_no_sw[4].(obstimes_window)-res_betas_no_sw[2].(obstimes_window)), label = "Adaptive MH β", color = :green, linewidth = 2, alpha = 0.5)

plot!(obstimes_window2, β_func.(obstimes_window2), label = "True β", color = :red, linewidth = 2)
savefig("out/Sliding_Window_beta_quantiles2.png")


# Preload one to get type of `initial_samples`
first_data_final = jldopen(first_path, "r") do f
    f["initial_sample"][end-1:end,:,:]  # Load a subset
end;

T_final = typeof(first_data_final)
data_array_final = Vector{T}(undef, length(sorted_dirs))
data_array_final[1] = first_data_final

# Load the rest
for (i, dir) in enumerate(sorted_dirs[2:end])
    file_path = joinpath(input_dir, dir, "chain_sw_400_260.jld2")
    data_array_final[i+1] = jldopen(file_path, "r") do f
f["initial_sample"][end-1:end,:,:]  # Load a subset 
    end;
end

combined_samples_final = AbstractMCMC.chainsstack(AbstractMCMC.tighten_eltype(data_array_final))

res_betas_final = get_beta_quantiles_sw(combined_samples_final, knots_window, obstimes_window, K_window-1, 2)

default(size=(1200, 800), legendfontsize=20, guidefontsize=24, tickfontsize=20)

plot(obstimes_window, res_betas_final[2].(obstimes_window),ribbon = (res_betas_final[2].(obstimes_window)-res_betas_final[3].(obstimes_window),res_betas_final[4].(obstimes_window)-res_betas_final[2].(obstimes_window)), label = "Sliding Window β (Window 1)", color = "#E67300", linewidth = 3.5, alpha =0.4)

plot!(obstimes_window2, res_betas2[2].(obstimes_window2),ribbon = (res_betas2[2].(obstimes_window2)-res_betas2[3].(obstimes_window2),res_betas2[4].(obstimes_window2)-res_betas2[2].(obstimes_window2)), label = "Sliding Window β (Window 2)", color = "#008080", linewidth = 3.5, alpha = 0.4)
# plot!(obstimes_window, res_betas_no_sw[2].(obstimes_window),ribbon = (res_betas_no_sw[2].(obstimes_window)-res_betas_no_sw[3].(obstimes_window),res_betas_no_sw[4].(obstimes_window)-res_betas_no_sw[2].(obstimes_window)), label = "Adaptive MH β", color = :green, linewidth = 2, alpha = 0.5)
plot!([80, 120], seriestype="vline", color = :black, label = "", linestyle = :dash, linewidth = 3)
plot!(obstimes_window2, β_func.(obstimes_window2), label = "True β", color = :red, linewidth = 3.5)
plot!(legend = :outerbottom, xlabel = "Time (days)", ylabel = "β", title = "Sliding Window β Quantiles", framestyle = :box, legend_columns = 2, titlefontsize = 24, left_margin=5mm, right_margin = 5mm, bottom_margin = -10mm)
plot!(tickwidth = 3, borderlinewidth = 5)
plot!(size = (1200,1000))
savefig("out/Sliding_Windows_combined_beta_quantiles_final.png")

using StatsBase: Histogram
using Distances
using EmpiricalDistributions

list_hists = Vector{Histogram{Float64}}(undef, 13)
for i in 1:10
    list_hists[i] = fit(Histogram{Float64}, alt_res_ld2[:,i]; nbins = 20)
end
list_hists[11] = fit(Histogram{Float64}, ests_ld2[70,:]; nbins = 20)
list_hists[12] = fit(Histogram{Float64}, ests_ld2[500,:]; nbins = 20)
list_hists[13] = fit(Histogram{Float64}, ests_ld2[end,:]; nbins = 20)

lower_bound = minimum([first(list_hists[j].edges[1]) for j in eachindex(list_hists)])

lower_bound = first(minimum(map(x -> x.edges[1], list_hists)))
upper_bound = last(maximum(map(x -> x.edges[1], list_hists)))
step_min = minimum(map(x -> step(x.edges[1]), list_hists))

same_bins_list_hists = Vector{Histogram{Float64}}(undef, length(list_hists))
for i in 1:10
    same_bins_list_hists[i] = fit(Histogram{Float64}, alt_res_ld2[:,i], lower_bound:step_min:upper_bound)
end
same_bins_list_hists[11] = fit(Histogram{Float64}, ests_ld2[70,:], lower_bound:step_min:upper_bound)
same_bins_list_hists[12] = fit(Histogram{Float64}, ests_ld2[500,:], lower_bound:step_min:upper_bound)
same_bins_list_hists[13] = fit(Histogram{Float64}, ests_ld2[end,:], lower_bound:step_min:upper_bound)

norm_list_hists = normalize.(same_bins_list_hists, mode = :pdf)
for j in eachindex(norm_list_hists)
        norm_list_hists[j].weights = norm_list_hists[j].weights .+ 1e-11
end
norm_list_hists = normalize.(norm_list_hists, mode = :pdf)


kldivs_within =[
        kldivergence(
            UvBinnedDist(norm_list_hists[j]),
            UvBinnedDist(norm_list_hists[k]))
            for j in 1:10
            for k in j+1:10
    ] 


kldivs_between = [
        kldivergence(
            UvBinnedDist(norm_list_hists[j]),
            UvBinnedDist(norm_list_hists[k]))
            for j in 1:10
            for k in 11:13
    ]

# Perform the same for the euclidean distance (wasserstein distance)
euclidean_between = [
    evaluate(Euclidean(), alt_res_ld2[1:200,j], ests_ld2[k,:]) 
        for j in 1:10
        for k in [70,500,1000]
    ] 

euclidean_within = [
    evaluate(Euclidean(), alt_res_ld2[1:200,j], alt_res_ld2[1:200,k])
        for j in 1:10
        for k in j+1:10
]

# Write me a function which plots densities of the the euclidean within vs between results and likewise for the kldivs
function plot_distances(distances_within, distances_between, title_str)
    p = plot()
    density!(p, distances_within, label = "Within", color = :blue, alpha = 0.5, linewidth = 2)
    density!(p, distances_between, label = "Between", color = :red, alpha = 0.5, linewidth = 2)
    plot!(p, title = title_str, xlabel = "Distance", ylabel = "Density", framestyle = :box)
    return p
end

plot_distances(kldivs_within, kldivs_between, "KLDivergence Within vs Between")

