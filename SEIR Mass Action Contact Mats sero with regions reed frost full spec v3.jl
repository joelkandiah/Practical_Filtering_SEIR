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
using BenchmarkTools
using SparseArrays
using Plots.PlotMeasures
using PracticalFiltering
using DelimitedFiles
using DynamicPPL
using AMGS
using Tullio
using JLD2
using Distributed
using CategoricalArrays

# Read in the arguments from the command line (File path, seed idx, seed_array_idx)
@assert length(ARGS) == 4

config_path = ARGS[1]
config = TOML.parsefile(config_path)

const seeds_list = config["seeds_list"]
const seeds_array_list = config["seeds_array_list"]
const Δ_βt = config["Delta_beta_t"]
const Window_size = config["Window_size"]
const n_chains = config["n_chains"]
const discard_init = config["discard_initial"]
const n_samples = config["n_samples"]
const n_warmup_samples = config["n_warmup"]
const tmax = config["tmax"]

# Get the data seed
const seed_idx = parse(Int, ARGS[2])

# Get the seed for the current portion of the run
const seed_array_idx = parse(Int, ARGS[])
 
# Set seed
Random.seed!(seeds_list[seed_idx])

const n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
const outdir = string("Results/10bp pop_reg_alt/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")
const tmpstore = string("Chains/10bp pop_reg_alt/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
const tspan = (0.0, tmax)
const obstimes = 1.0:1.0:tmax
# NA_N= [74103.0, 318183.0, 804260.0, 704025.0, 1634429.0, 1697206.0, 683583.0, 577399.0]
const NA_N = Matrix(transpose([74103.0 318183.0 804260.0 704025.0 1634429.0 1697206.0 683583.0 577399.0;
          122401.0 493480.0 1123981.0 1028009.0 3063113.0 2017884.0 575433.0 483780.0;
          118454.0 505611.0 1284743.0 1308343.0 2631847.0 2708355.0 1085138.0 895188.0;
          92626.0 396683.0 1014827.0 1056588.0 2115517.0 2253914.0 904953.0 731817.0;
          79977.0 340962.0 851539.0 845215.0 1786666.0 1820128.0 710782.0 577678.0;
          94823.0 412569.0 1085466.0 1022927.0 2175624.0 2338669.0 921243.0 801040.0;
          55450.0 241405.0 633169.0 644122.0 1304392.0 1496240.0 666261.0 564958.0;
]))
const NA = size(NA_N)[1]
const NR = size(NA_N)[2]
const N_pop_R = sum(NA_N, dims = 1)[1,:]
const N_pop_A = sum(NA_N, dims = 2)[:,1]

# Read all of the contact matrices
directory_path = "ContactMats"
txt_files = String[]
for file_name in readdir(directory_path)
   if endswith(file_name, ".txt")
         full_path = joinpath(directory_path, file_name)
         push!(txt_files, full_path)
    end
end

files_with_ldwk = filter(file_path -> occursin(r"ldwk", file_path), txt_files)

sort!(files_with_ldwk, by = file_path -> begin
    file_name = basename(file_path)
    # Use Regex to find numbers after "ldwk" or "ldwk_"
    m = match(r"ldwk_?(\d+)", file_name)
    if m !== nothing
        return parse(Int, m.captures[1]) # Convert the captured number string to an integer
    else
        return typemax(Int) # For files that match 'ldwk' but not 'ldwk#' (shouldn't happen if regex is precise)
    end
end)

using DelimitedFiles

C_base = readdlm(txt_files[1], Float64) ./ N_pop_A .* 1e6
C_ldwks = stack([readdlm(file_path, Float64) ./ N_pop_A .* 1e6 for file_path in files_with_ldwk], dims = 1)


# Construct an ODE for the SEIR model
function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
    # Grab values needed for calculation
    @inbounds @views begin
        S = u[1,:,:]
        E1 = u[2,:,:]
        E2 = u[3,:,:]
        I1 = u[4,:,:]
        I2 = u[5,:,:]
    end
    (γ, σ, ld_start, Δt_C) = p_.params_floats
    a0 = p_.a0

    (infection, infectious_1, infectious_2, recovery_1, recovery_2, b) = p_.cache_SIRderivs
    (βt,) = p_.cache_beta
    C = similar(p_.cache_C)
    # sus_M_tmp = t >= ld_start ? p_.sus_M[2,:,:,:] : p_.sus_M[1,:,:,:]
    # todo: fix contact matrix issues!!!
    C .= t <= ld_start ? p_.C_base : p_.C[Int(ceil(t / Δt_C)) + 2,:,:,:]
    # display("t:$t, C:$(C[1]), sus:($sus_M_tmp[1])")
    # Calculate the force of infection
    map!(x-> x(t), βt, p_.β_functions)
    # @tullio grad = false threads = false b[i,k] = (I1[j,k] + I2[j,k]) * log(1 - (a0[k] * βt[k] * sus_M_tmp[k,i,j] *  C[i,j])) |> (1 - exp((_)))
    @tullio grad = false threads = false b[i,k] = (I1[j,k] + I2[j,k]) * log(1 - (a0[k] * βt[k] * C[k,i,j])) |> (1 - exp((_)))
    
    infection .= S .* b
    infectious_1 .= 0.5 .* σ .* E1
    infectious_2  .= 0.5 .* σ .* E2
    recovery_1 .= 0.5 .* γ .* I1
    recovery_2 .= 0.5 .* γ .* I2
    @inbounds begin
        du[1,:,:] .= .- infection
        du[2,:,:] .= infection .- infectious_1
        du[3,:,:] .= infectious_1 .- infectious_2
        du[4,:,:] .= infectious_2 .- recovery_1
        du[5,:,:] .= recovery_1 .- recovery_2
        du[6,:,:] .= recovery_2
        du[7,:,:] .= infection
    end
end;

# Construct an ODE for the SEIR model
# function sir_tvp_ode!(du::Array{T1}, u::Array{T2}, p_, t) where {T1 <: Real, T2 <: Real}
#     # Grab values needed for calculation
#     @inbounds @views begin
#         S = u[1,:,:]
#         E1 = u[2,:,:]
#         E2 = u[3,:,:]
#         I1 = u[4,:,:]
#         I2 = u[5,:,:]
#     end
#     (γ, σ, ld_start, Δt_C) = p_.params_floats
#     a0 = p_.a0
# 
#     # (infection, infectious_1, infectious_2, recovery_1, recovery_2, b) = p_.cache_SIRderivs
#     # (βt,) = p_.cache_beta
#     # C = p_.cache_C
#     # sus_M_tmp = t >= ld_start ? p_.sus_M[2,:,:,:] : p_.sus_M[1,:,:,:]
#     # todo: fix contact matrix issues!!!
#     p_.cache_C .= t <= ld_start ? p_.C_base : p_.C[Int(ceil(t / Δt_C)) + 2,:,:,:]
#     # display("t:$t, C:$(C[1]), sus:($sus_M_tmp[1])")
#     # Calculate the force of infection
#     map!(x-> x(t), p_.cache_beta[1], p_.β_functions)
#     # @tullio grad = false threads = false b[i,k] = (I1[j,k] + I2[j,k]) * log(1 - (a0[k] * βt[k] * sus_M_tmp[k,i,j] *  C[i,j])) |> (1 - exp((_)))
#     @tullio grad = false threads = false p_.cache_SIRderivs[6][i,k] = (I1[j,k] + I2[j,k]) * log(1 - (a0[k] * p_.cache_beta[1][k] * p_.cache_C[k,i,j])) |> (1 - exp((_)))
# 
#     p_.cache_SIRderivs[1] .= S .* p_.cache_SIRderivs[6]
#     p_.cache_SIRderivs[2] .= 0.5 .* σ .* E1
#     p_.cache_SIRderivs[3] .= 0.5 .* σ .* E2
#     p_.cache_SIRderivs[4] .= 0.5 .* γ .* I1
#     p_.cache_SIRderivs[5] .= 0.5 .* γ .* I2
#     @inbounds begin
#         du[1,:,:] .= .- p_.cache_SIRderivs[1]
#         du[2,:,:] .= p_.cache_SIRderivs[1] .- p_.cache_SIRderivs[2]
#         du[3,:,:] .= p_.cache_SIRderivs[2] .- p_.cache_SIRderivs[3]
#         du[4,:,:] .= p_.cache_SIRderivs[3] .- p_.cache_SIRderivs[4]
#         du[5,:,:] .= p_.cache_SIRderivs[4] .- p_.cache_SIRderivs[5]
#         du[6,:,:] .= p_.cache_SIRderivs[5]
#         du[7,:,:] .= p_.cache_SIRderivs[1]
#     end
# end;

struct idd_params{T <: Real, T2 <: Real, T3 <: DataInterpolations.AbstractInterpolation, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}
    params_floats::Vector{T}
    NR_pops::Vector{T2}
    β_functions::Vector{T3}
    C_base::Array{T4}
    C::Array{T4}
    # sus_M::Array{T5}
    a0::Vector{T6}
    cache_SIRderivs::Vector{Matrix{T7}}
    cache_beta::Vector{Vector{T8}}
    cache_C::Array{T5}
end

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

# Define Gamma distribution by mean and standard deviation
function Gamma_mean_sd_dist(μ, σ)
    α = @. (μ * μ) / (σ * σ)
    θ = @. (σ * σ) / μ
    return Gamma.(α, θ)
end

# Define helpful distributions (arbitrary choice from sample in RTM)
const incubation_dist = Gamma_mean_sd_dist(4.0, 1.41)
const symp_to_hosp = Gamma_mean_sd_dist(15.0, 12.1)

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
IFR_sds = [
    1.28185, 1.281734, 1.279731, 1.283084, 1.284837, 1.295454, 0.346383
]
IFR_means = [
    -11.61184, -10.634830, -9.148093, -7.586303, -5.368160, -3.999493, -2.516182
]
IFR_bijector = Bijectors.Logit(0, 1)
const conv_mat = construct_pmatrix(;)  


# Create function to construct Negative binomial with properties matching those in Birrell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

# Define Seroprevalence estimates

using DelimitedFiles
sample_sizes = map(x->readdlm("Serosamples_N/region_$x.txt", Int64), 1:NR)
sample_sizes = stack(sample_sizes)

sample_sizes = sample_sizes[1:length(obstimes),:,:]
sample_sizes = permutedims(sample_sizes, (2,3,1))

const sus_pop_mask = sample_sizes .!= 0
const sample_sizes_non_zero = @view sample_sizes[sus_pop_mask]

# normalize by max_eigenvalue
function normalize_by_max_eigenvalue(A::AbstractMatrix)
    eigvals_A = eigvals(A)
    max_eigval = maximum(abs.(eigvals_A))
    return A ./ max_eigval
end

function break_matrix_list(breaks::T1, matrices::Vector{<:AbstractMatrix}, const_after_maxtime::Bool) where {T1 <: Real}
    return function(t::Real)
        idx = searchsortedlast(breaks, t)
        if idx != length(breaks)
            return matrices[idx]
        end
        if const_after_maxtime
            return matrices[end]
        end     
        error("t = $t is outside the break range.")
    end
end


# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp(
    ode_prob,
    K,
    σ = σ,
    N = N_pop_R,
    NA = NA,
    NA_N = NA_N,
    N_regions = NR,
    conv_mat = conv_mat,
    knots = knots,
    C_base = C_base,
    C = C_ldwks,
    sero_sample_sizes = sample_sizes_non_zero,
    sus_pop_mask = sus_pop_mask,
    obstimes = obstimes,
    ld_start = 35,
    Δt_C = 7,
    IFR_means::Vector{Float64} = IFR_means,
    IFR_sds::Vector{Float64} = IFR_sds,
    trans_unconstrained_IFR = inverse(IFR_bijector),
    ::Type{T_β} = Float64,
    ::Type{T_ψ} = Float64,
    ::Type{T_IFR} = Float64,
    ::Type{T_sus_M}  = Float64,
    ::Type{T_u0} = Float64,
    ::Type{T_Seir} = Float64,
    ::Type{T_λ} = Float64
) where {T_β <: Real, T_ψ <: Real, T_IFR <: Real, T_sus_M <: Real, T_u0 <: Real, T_Seir <: Real, T_λ <: Real}

    η ~ truncated(Gamma(1,1/0.2), upper = 150)

    # Set priors for ψ
    ψ = Vector{T_ψ}(undef, N_regions)
    ψ_EE  ~ truncated(Gamma(31.36,1/224); upper = 0.5) 
    ψ_LDN ~ truncated(Gamma(31.36,1/224); upper = 0.5) 
    ψ_MID ~ truncated(Gamma(31.36,1/224); upper = 0.5)
    ψ_NEY ~ truncated(Gamma(31.36,1/224); upper = 0.5) 
    ψ_NW  ~ truncated(Gamma(31.36,1/224); upper = 0.5) 
    ψ_SE  ~ truncated(Gamma(31.36,1/224); upper = 0.5)
    ψ_SW  ~ truncated(Gamma(31.36,1/224); upper = 0.5) 

    ψ[1] = ψ_EE
    ψ[2] = ψ_LDN
    ψ[3] = ψ_MID
    ψ[4] = ψ_NEY
    ψ[5] = ψ_NW
    ψ[6] = ψ_SE
    ψ[7] = ψ_SW
    
    # Set priors for betas
    ## Note how we clone the endpoint of βt
    β = Matrix{T_β}(undef, K, N_regions)
    log_β₀_EE =  zero(T_β)
    log_β₀_LDN = zero(T_β)
    log_β₀_MID = zero(T_β)
    log_β₀_NEY = zero(T_β)
    log_β₀_NW =  zero(T_β)
    log_β₀_SE =  zero(T_β)
    log_β₀_SW =  zero(T_β)

    log_β_EE = Vector{T_β}(undef, K-2) 
    log_β_LDN = Vector{T_β}(undef, K-2) 
    log_β_MID = Vector{T_β}(undef, K-2) 
    log_β_NEY = Vector{T_β}(undef, K-2) 
    log_β_NW = Vector{T_β}(undef, K-2) 
    log_β_SE = Vector{T_β}(undef, K-2) 
    log_β_SW = Vector{T_β}(undef, K-2) 

    sero_sens  ~ Beta(71.5,29.5)
    sero_spec  ~ Beta(777.5,9.5)

    d_I  ~ Gamma(1.43,1/0.549)
    inv_γ = d_I + 2
    γ  = 1 / inv_γ

    p = [γ, σ, ld_start, Δt_C]
   
    βₜσ ~ truncated(Gamma(1,1/100), upper = 0.5)
    β[1,1] = exp(log_β₀_EE)
    β[1,2] = exp(log_β₀_LDN)
    β[1,3] = exp(log_β₀_MID)
    β[1,4] = exp(log_β₀_NEY)
    β[1,5] = exp(log_β₀_NW)
    β[1,6] = exp(log_β₀_SE)
    β[1,7] = exp(log_β₀_SW)

    for j in 2:K-1
        log_β_EE[j-1] ~ Normal(0.0, βₜσ)
        β[j,1] = exp.(log.(β[j-1,1]) .+ log_β_EE[j-1])
 
        log_β_LDN[j-1] ~ Normal(0.0, βₜσ)
        β[j,2] = exp.(log.(β[j-1,2]) .+ log_β_LDN[j-1])
 
        log_β_MID[j-1] ~ Normal(0.0, βₜσ)
        β[j,3] = exp.(log.(β[j-1,3]) .+ log_β_MID[j-1])
 
        log_β_NEY[j-1] ~ Normal(0.0, βₜσ)
        β[j,4] = exp.(log.(β[j-1,4]) .+ log_β_NEY[j-1])
 
        log_β_NW[j-1] ~ Normal(0.0, βₜσ)
        β[j,5] = exp.(log.(β[j-1,5]) .+ log_β_NW[j-1])
 
        log_β_SE[j-1] ~ Normal(0.0, βₜσ)
        β[j,6] = exp.(log.(β[j-1,6]) .+ log_β_SE[j-1])
 
        log_β_SW[j-1] ~ Normal(0.0, βₜσ)
        β[j,7] = exp.(log.(β[j-1,7]) .+ log_β_SW[j-1])
    end

    β[K,:] .= β[K-1,:]

    # Make more generic (pass indices)
    sus_M  = Array{T_sus_M}(undef, 2, NR, NA, NA)
    m_EE = Vector{T_sus_M}(undef, 5)
    m_LDN = Vector{T_sus_M}(undef, 5)
    m_MID = Vector{T_sus_M}(undef, 5)
    m_NEY = Vector{T_sus_M}(undef, 5)
    m_NW = Vector{T_sus_M}(undef, 5)
    m_SE = Vector{T_sus_M}(undef, 5)
    m_SW = Vector{T_sus_M}(undef, 5)

    for j in 1:5
    #    m_EE[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_LDN[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_MID[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_NEY[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_NW[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_SE[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
    #    m_SW[j] ~ truncated(Gamma(4,1/4), upper = 5.0)
        m_EE[j] ~ LogNormal(-0.111571775657105,0.472380727077439)
        m_LDN[j] ~LogNormal(-0.111571775657105,0.472380727077439)
        m_MID[j] ~LogNormal(-0.111571775657105,0.472380727077439)
        m_NEY[j] ~LogNormal(-0.111571775657105,0.472380727077439)
        m_NW[j] ~ LogNormal(-0.111571775657105,0.472380727077439)
        m_SE[j] ~ LogNormal(-0.111571775657105,0.472380727077439)
        m_SW[j] ~ LogNormal(-0.111571775657105,0.472380727077439)
    end

    sus_M[1,:,4:7,:] .= one(T_sus_M)

    sus_M[1,1,1:3,:] .= m_EE[1]
    sus_M[1,2,1:3,:] .= m_LDN[1]
    sus_M[1,3,1:3,:] .= m_MID[1]
    sus_M[1,4,1:3,:] .= m_NEY[1]
    sus_M[1,5,1:3,:] .= m_NW[1]
    sus_M[1,6,1:3,:] .= m_SE[1]
    sus_M[1,7,1:3,:] .= m_SW[1]

    sus_M[1,1,8,:] .= m_EE[2]
    sus_M[1,2,8,:] .= m_LDN[2]
    sus_M[1,3,8,:] .= m_MID[2]
    sus_M[1,4,8,:] .= m_NEY[2]
    sus_M[1,5,8,:] .= m_NW[2]
    sus_M[1,6,8,:] .= m_SE[2]
    sus_M[1,7,8,:] .= m_SW[2]

    sus_M[2,1,1:3,:] .= m_EE[4]
    sus_M[2,2,1:3,:] .= m_LDN[4]
    sus_M[2,3,1:3,:] .= m_MID[4]
    sus_M[2,4,1:3,:] .= m_NEY[4]
    sus_M[2,5,1:3,:] .= m_NW[4]
    sus_M[2,6,1:3,:] .= m_SE[4]
    sus_M[2,7,1:3,:] .= m_SW[4]

    sus_M[2,1,4:7,:] .= m_EE[3]
    sus_M[2,2,4:7,:] .= m_LDN[3]
    sus_M[2,3,4:7,:] .= m_MID[3]
    sus_M[2,4,4:7,:] .= m_NEY[3]
    sus_M[2,5,4:7,:] .= m_NW[3]
    sus_M[2,6,4:7,:] .= m_SE[3]
    sus_M[2,7,4:7,:] .= m_SW[3]

    sus_M[2,1,8,:] .= m_EE[5]
    sus_M[2,2,8,:] .= m_LDN[5]
    sus_M[2,3,8,:] .= m_MID[5]
    sus_M[2,4,8,:] .= m_NEY[5]
    sus_M[2,5,8,:] .= m_NW[5]
    sus_M[2,6,8,:] .= m_SE[5]
    sus_M[2,7,8,:] .= m_SW[5]

    log_λ₀_EE ~ Normal(-17.5, 1.25)
    log_λ₀_LDN ~ Normal(-17.5, 1.25)
    log_λ₀_MID ~ Normal(-17.5, 1.25)
    log_λ₀_NEY ~ Normal(-17.5, 1.25)
    log_λ₀_NW ~ Normal(-17.5, 1.25)
    log_λ₀_SE ~ Normal(-17.5, 1.25)
    log_λ₀_SW ~ Normal(-17.5, 1.25)

    λ₀ = Vector{T_λ}(undef, N_regions)
    λ₀[1] = exp(log_λ₀_EE)
    λ₀[2] = exp(log_λ₀_LDN)
    λ₀[3] = exp(log_λ₀_MID)
    λ₀[4] = exp(log_λ₀_NEY)
    λ₀[5] = exp(log_λ₀_NW)
    λ₀[6] = exp(log_λ₀_SE)
    λ₀[7] = exp(log_λ₀_SW)

    # Write the priors for the IFR
    IFR_vec = Vector{T_IFR}(undef, NA)
    IFR_vec_init = Vector{T_IFR}(undef, NA - 1)
    for i in 1:(NA-1)
        IFR_vec_init[i] ~ Normal(IFR_means[i], IFR_sds[i])
    end

    IFR_vec[2:end] .= trans_unconstrained_IFR.(IFR_vec_init)
    IFR_vec[1] = IFR_vec[2]
    
    # Check that the IFR is not too high
    unconstrained_IFR_means = trans_unconstrained_IFR.(IFR_means)
    # println("IFR_means: $unconstrained_IFR_means")
    # println("IFR_vec: $IFR_vec")

    # if(any(β .>  1 / maximum(C .* 0.22 ./ minimum(N))) | any(isnan.(β)))
    #     # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext(  )
    #         @DynamicPPL.addlogprob! -Inf
    #         return
    #     # end
    # end

    # Set up the β β_functions
    β_functions = map(x -> ConstantInterpolation(x, knots), eachcol(β))

    infections = zeros(NA, N_regions)
    infectious_1 = zeros(NA, N_regions)
    infectious_2 = zeros(NA, N_regions)
    recovery_1 = zeros(NA, N_regions)
    recovery_2 = zeros(NA, N_regions)
    βt = zeros(N_regions)
    b = zeros(NA, N_regions)
    cache_SIRderivs = [infections, infectious_1, infectious_2, recovery_1, recovery_2, b]
    cache_beta = [βt,]
    cache_C = zeros(N_regions, NA, NA)

    inv_σ = 1 / σ
    R0_num = @. 1 + (ψ * inv_σ * 0.5) 
    R0_denom = @. 1 - (1 / ((1 + (ψ * inv_γ * 0.5))^2))
    a0_num = @. ψ * inv_γ * (R0_num^2) ./ R0_denom
   
    #  Fix the issue with normalising as indexing isn't right on Ĉ yet
    # Ĉ_base = stack(map(k -> sus_M[1,k,:,:] .* normalize_by_max_eigenvalue(C_base .* NA_N[:,k]), 1:NR), dims = 1)
    Ĉ_base = stack(map(k -> sus_M[1,k,:,:] .* normalize_by_max_eigenvalue(C_base), 1:NR), dims = 1)
    # Ĉ = stack(map(j -> stack(map(k -> sus_M[2,k,:,:] .* normalize_by_max_eigenvalue(C[j,:,:] .* NA_N[:,k]), 1:NR), dims = 1), 1:size(C)[1]), dims = 1)
    Ĉ = stack(map(j -> stack(map(k -> sus_M[2,k,:,:] .* normalize_by_max_eigenvalue(C[j,:,:]), 1:NR), dims = 1), 1:size(C)[1]), dims = 1)

    # Add priors for p_lambda_0
    # a0_denom = map(k -> β_functions[k](0) * inv_γ * real(eigmax(diagm(NA_N[:,k]) * (sus_M[1,k,:,:] .* C_base))), 1:NR)
    a0_denom = map(k -> β_functions[k](0) * inv_γ * real(eigmax(diagm(NA_N[:,k]) * Ĉ_base[k,:,:])), 1:NR)
    a0 = a0_num ./ a0_denom

    # @tullio b0[i,k] := log(1 - (a0[k] * β_functions[k](0) * sus_M[1,k,i,j] *  C_base[i,j])) |> (1 - exp((_)))
    @tullio b0[i,k] := log(1 - (a0[k] * β_functions[k](0) * Ĉ_base[k,i,j])) |> (1 - exp((_)))
    tot_I₀ = inv_γ * map(k -> λ₀[k] .* sum(NA_N[:,k]) / a0_num[k], 1:N_regions)
    # χ = map(k -> normalize(abs.(eigen(diagm(NA_N[:,k]) * sus_M[1,k,:,:] .* C_base, sortby=x -> abs(x)).vectors[:,end]), 1), 1:NR) |> stack
    χ = map(k -> normalize(abs.(eigen(diagm(NA_N[:,k]) * Ĉ_base[k,:,:], sortby=x -> abs(x)).vectors[:,end]), 1), 1:NR) |> stack
    tot_I₀_ages = χ .* transpose(tot_I₀)
    u0 = zeros(T_u0,7,NA,NR)
    α = exp.(ψ) .- 1
    ϵ = 1
    u0[4,:,:] = tot_I₀_ages .* inv.(transpose(1 .+ ((ϵ .* γ) ./ (α .+ γ))))
    u0[5,:,:] = tot_I₀_ages .- @view u0[4,:,:]
    u0[3,:,:] = transpose((α .+ γ) ./ σ) .* @view(u0[4,:,:])
    u0[2,:,:] = transpose((α .+ σ) ./ σ) .* @view(u0[3,:,:])
    u0[1,:,:] = NA_N .- tot_I₀_ages .- @view(u0[2,:,:]) .- @view u0[3,:,:]
    
    params_test = idd_params(p, N, β_functions, Ĉ_base, Ĉ, a0, cache_SIRderivs, cache_beta, cache_C)

    # Run model
    ## Remake with new initial conditions and parameter values
    prob = remake(ode_prob, u0 = u0, p = params_test)

    ## Solve
    sol = 
        solve(prob,
            Tsit5(),
            saveat = obstimes,
            d_discontinuities = knots[2:end-1],
            tstops = knots)

    if any(!SciMLBase.successful_retcode(sol))
        println("hit 2")
        println("η: $η \nm_1: $(m_1)\nm_2: $(m_2)\nsero_sens: $sero_sens \nsero_spec: $sero_spec\n IFR_vec: $IFR_vec \n d_I: $(inv_γ)")
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    ## Calculate new infections per day, X
    sol_X =  stack(sol.u)[7,:,:,:] |>
        rowadjdiff3d
    
    if (any(sol_X .<= -(1e-3)) | any(stack(sol.u)[4,:,:,:] .<= -(1e-3)) | any(stack(sol.u)[5,:,:,:] .<= -1e-3))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    y_μ = Array{T_Seir}(undef, NA, N_regions, length(obstimes))

    for i in 1:N_regions
        y_μ[:,i,:] .= (conv_mat * (IFR_vec .* sol_X[:,i,:])') |>
            transpose
    end

    ## Calculate number of timepoints
    if (any(isnan.(y_μ)))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, η))

    # Introduce Serological data into the model (for the first region)
    sus_pop = map(x -> x[1,:,:], sol.u) |>
        stack
    sus_pop_samps = @view (sus_pop[:,:,:] ./ NA_N)[sus_pop_mask]
    
    z ~ product_distribution(@. Binomial(
        sero_sample_sizes,
        (sero_sens * (1 - (sus_pop_samps))) + ((1 - sero_spec) * (sus_pop_samps))
    ))

    sus = stack(sol.u)[1,:,:,:]
    Rt = zeros(T_Seir, N_regions, length(obstimes))
    for i in 1:length(obstimes)
        for j in 1:NR
            C_i = i < ld_start ? Ĉ_base[j,:,:] : Ĉ[Int(ceil(i / Δt_C)) + 2, j,:,:]
            Rt[j,i] = a0[j] * β_functions[j](i) * inv_γ * real(eigmax(diagm(sus[:,j,i]) * C_i[:,:]))
        end
    end

    return (; sol, y_μ, y, z, a0_num, β, β_functions, sus_M, Rt)
end;

post_params = TOML.parsefile("posterior_inputs/mod_pars_$seed_idx.toml")

knots_init = vcat([0],post_params["log_beta_rw"]["time_breakpoints"])
knots_init = (knots_init[end] != tmax)  && (knots_init[end] < tmax) ? vcat(knots_init, tmax) : knots_init

# Define the parameters for the model given the known window size (i.e. vectors that fit within the window) 
#knots_window = collect(0:Δ_βt:Window_size)
#knots_window = knots_window[end] != Window_size ? vcat(knots_window, Window_size) : knots_window
#K_window = length(knots_window)
#obstimes_window = 1.0:1.0:Window_size
#conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, length(obstimes_window))

#sus_pop_mask_window = sus_pop_mask[:,:,1:length(obstimes_window)]
#sample_sizes_non_zero_window = @view sample_sizes[:,:,1:length(obstimes_window)][sus_pop_mask_window]

#knots_init = collect(0:Δ_βt:tmax)
#knots_init = knots_init[end] != tmax ? vcat(knots_init, tmax) : knots_init
K_init = length(knots_init)

my_infection = zeros(NA, NR)
my_infectious_1 = zeros(NA, NR)
my_infectious_2  = zeros(NA, NR)
my_recovery_1 = zeros(NA, NR)
my_recovery_2 = zeros(NA, NR)
my_βt = zeros(NR)
my_b = zeros(NA, NR)
# my_sus_M = zeros(Float64, NR, NA, NA)
my_C = zeros(Float64, NR, NA, NA)

u0 = zeros(Float64, 7, NA, NR)
u0[1,:,:] = NA_N

params_window = idd_params([0.0, 0.0, 35.0, 7.0], 
    N_pop_R,
    map(x -> ConstantInterpolation(x, knots_init), eachcol(ones(K_init, NR))),
    zeros(NR,NA,NA),
    zeros(size(C_ldwks)[1], NR, NA, NA),
#    ones(2,NR,NA, NA),
    ones(NR),
    [my_infection, my_infectious_1, my_infectious_2, my_recovery_1, my_recovery_2, my_b],
    [my_βt,],
    my_C)

# ode_prob_window = ODEProblem{true}(sir_tvp_ode!, u0, (0.0, obstimes_window[end]), params_window)

ode_prob_init = ODEProblem{true}(sir_tvp_ode!, deepcopy(u0), (0.0, tmax), deepcopy(params_window))

const σ = 1 / 3.0
# Sample the parameters to construct a "correct order" list of parameters
init_model = bayes_sir_tvp(
        ode_prob_init,
        K_init,
        σ,
        N_pop_R,
        NA,
        NA_N,
        NR,
        conv_mat,
        knots_init,
        C_base,
        C_ldwks,
        sample_sizes_non_zero,
        sus_pop_mask,
        obstimes,
        36.0,
        7.0,
        IFR_means,
        IFR_sds,
        inverse(IFR_bijector))

rw_length = K_init-1

conditions = (
    η = post_params["hosp_negbin_overdispersion"]["param_value"],
    ψ_EE = post_params["exponential_growth_rate"]["param_value"][1],
    ψ_LDN = post_params["exponential_growth_rate"]["param_value"][2],
    ψ_MID = post_params["exponential_growth_rate"]["param_value"][3],
    ψ_NEY = post_params["exponential_growth_rate"]["param_value"][4],
    ψ_NW = post_params["exponential_growth_rate"]["param_value"][5],
    ψ_SE = post_params["exponential_growth_rate"]["param_value"][6],
    ψ_SW = post_params["exponential_growth_rate"]["param_value"][7],
    log_β_EE = post_params["log_beta_rw"]["param_value"][2:rw_length],
    log_β_LDN = post_params["log_beta_rw"]["param_value"][rw_length+2:2*rw_length],
    log_β_MID = post_params["log_beta_rw"]["param_value"][2*rw_length+2:3*rw_length],
    log_β_NEY = post_params["log_beta_rw"]["param_value"][3*rw_length+2:4*rw_length],
    log_β_NW = post_params["log_beta_rw"]["param_value"][4*rw_length+2:5*rw_length],
    log_β_SE = post_params["log_beta_rw"]["param_value"][5*rw_length+2:6*rw_length],
    log_β_SW = post_params["log_beta_rw"]["param_value"][6*rw_length+2:7*rw_length],
    sero_sens = post_params["sero_test_sensitivity"]["param_value"],  
    sero_spec = post_params["sero_test_specificity"]["param_value"], 
    d_I = post_params["infectious_period"]["param_value"],
    βₜσ = post_params["log_beta_rw_sd"]["param_value"][2],
    m_EE = post_params["contact_parameters"]["param_value"][2:6] .|> exp,
    m_LDN = post_params["contact_parameters"]["param_value"][6+2:2*6] .|> exp,
    m_MID = post_params["contact_parameters"]["param_value"][2*6+2:3*6] .|> exp,
    m_NEY = post_params["contact_parameters"]["param_value"][3*6+2:4*6] .|> exp,
    m_NW = post_params["contact_parameters"]["param_value"][4*6+2:5*6] .|> exp,
    m_SE = post_params["contact_parameters"]["param_value"][5*6+2:6*6] .|> exp,
    m_SW = post_params["contact_parameters"]["param_value"][6*6+2:end] .|> exp,
    log_λ₀_EE = post_params["log_p_lambda_0"]["param_value"][1],
    log_λ₀_MID = post_params["log_p_lambda_0"]["param_value"][2],
    log_λ₀_LDN = post_params["log_p_lambda_0"]["param_value"][3],
    log_λ₀_NEY = post_params["log_p_lambda_0"]["param_value"][4],
    log_λ₀_NW = post_params["log_p_lambda_0"]["param_value"][5],
    log_λ₀_SE = post_params["log_p_lambda_0"]["param_value"][6],
    log_λ₀_SW = post_params["log_p_lambda_0"]["param_value"][7],
    IFR_vec_init = post_params["prop_case_to_hosp"]["param_value"] .|> IFR_bijector  
)

# conditions = (IFR_vec_init = IFR_bijector.([0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]), d_I = 2.2, sero_spec = 0.99, βₜσ=0.1, log_λ₀_EE = -13.1, log_λ₀_LDN = -12.9, log_λ₀_MID = -12.7, log_λ₀_NEY = -12.5, log_λ₀_NW = -12.3, log_λ₀_SE = -13.3, log_λ₀_SW = -13.0, ψ_EE = 0.2, ψ_LDN = 0.2, ψ_MID = 0.2, ψ_NEY = 0.2, ψ_NW = 0.2, ψ_SE = 0.2, ψ_SW = 0.2);

ode_prior = sample(init_model | conditions, Prior(), 1)
#        | (y = Y[:,:,1:length(obstimes_window)],
#     z = 💉[1:length(sample_sizes_non_zero_window)]
#     ), Prior(), 1, discard_initial = 0, thinning = 1);

data = generated_quantities(init_model | conditions, ode_prior)[1]

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(1:length(obstimes), data.y[:,i,:]', legend = false, alpha = 0.3)
end
StatsPlots.plot(a..., layout = (4,2), size = (1800,1200))

b = Vector{Plots.Plot}(undef, NA * NR)
positives = similar(sample_sizes, Float64)
positives[sus_pop_mask] .= data.z ./ sample_sizes_non_zero
positives[.!sus_pop_mask] .= NaN
for i in 1:NA
    for j in 1:NR
        b[(i-1) * NR + j] = scatter(obstimes, positives[i,j,:], title = "NA: $i, NR: $j", legend = false, ylims = (0,1))
    end
end
plot(b..., layout = (NA, NR), size = (2800, 3400))

c = Vector{Plots.Plot}(undef, NR)
dat_SEIR = stack(data.sol(obstimes))
for i in 1:NR
    c[i] = plot(obstimes, dat_SEIR[4,:,i,:]' .+ dat_SEIR[5,:,i,:]')
end
plot(c..., layout = (4,2), size = (1800,1200))

d = Vector{Plots.Plot}(undef, NR)
dat_SEIR = stack(data.sol(obstimes))
for i in 1:NR
    plot_part = plot(obstimes, dat_SEIR[1,:,i,:]')
    d[i] = plot!(plot_part, obstimes, dat_SEIR[6,:,i,:]')
end
plot(d..., layout = (4,2), size = (1800,1200))

e = Vector{Plots.Plot}(undef, NR)
dat_SEIR = stack(data.sol(obstimes))
for i in 1:NR
    e[i] = plot(obstimes, rowadjdiff3d(dat_SEIR[7,:,:,:])[:,i,:]')
end
plot(e..., layout = (4,2), size = (1800,1200))


# e = Vector{Plots.Plot}(undef, NR)
# dat_SEIR = stack(data.sol(obstimes))
# for i in 1:NR
#     e[i] = plot(obstimes, (dat_SEIR[7,:,:,:] |> rowadjdiff3d)[:,i,:]')
# end
# 
# plot(e..., layout = (4,2), size = (1800,1200))
# res = dat_SEIR[7,:,:,:] |> rowadjdiff3d
# res_alt = (NA_N .- dat_SEIR[1,:,:,:]) |> rowadjdiff3d
# 
# 
# IFR_Vec =[0.0000078, 0.0000078, 0.000017, 0.000038, 0.00028, 0.0041, 0.025, 0.12]
# f = Vector{Plots.Plot}(undef, NR)
# dat_SEIR = stack(data.sol(obstimes))
# for i in 1:NR
#     tmp = plot(obstimes, (conv_mat * (IFR_Vec .* res[:,i,:])'))
#     f[i] = StatsPlots.scatter!(tmp, obstimes, data.y[:,i,:]', legend = false, alpha = 0.3)
# end
# plot(f..., layout = (4,2), size = (1800,1200))
# 
# g = Vector{Plots.Plot}(undef, NR)
# dat_SEIR = stack(data.sol(obstimes))
# for i in 1:NR
#     tmp = plot(obstimes, (conv_mat * (IFR_Vec .* res_alt[:,i,:])'))
#     g[i] = StatsPlots.scatter!(tmp, obstimes, data.y[:,i,:]', legend = false, alpha = 0.3)
# end
# plot(g..., layout = (4,2), size = (1800,1200))
# 
# 
# h = Vector{Plots.Plot}(undef,NR)
# sol_X = stack(data.sol.u)[7,:,:,:] |> rowadjdiff3d
# y_μ = Array{Float64}(undef, NA, NR, length(obstimes))
# for i in 1:NR
#     y_μ[:,i,:] .= (conv_mat * (IFR_Vec .* sol_X[:,i,:])') |>
#     transpose
# end
# for i in 1:NR
#     tmp = plot(obstimes, y_μ[:,i,:]') 
#     tmp = plot!(tmp, obstimes, data.y_μ[:,i,:]')
#     h[i] = StatsPlots.scatter!(tmp, obstimes, data.y[:,i,:]', legend = false, alpha = 0.3)
# end
# plot(h..., layout = (4,2), size = (1800,1200))
# 
# 
# res_y = Random.rand(product_distribution(NegativeBinomial3.(data.y_μ .+ 1e-3, 4.028)))
# a2 = Vector{Plots.Plot}(undef, NR)
# for i in 1:NR
#     tmp = StatsPlots.scatter(obstimes, res_y[:,i,:]', legend = false, alpha = 0.3)
#     a2[i] = StatsPlots.scatter!(tmp, obstimes, data.y[:,i,:]', legend = false, alpha = 0.3)
# end
# plot(a2..., layout = (4,2), size = (1800, 1200))
#
#
model_window_unconditioned = bayes_sir_tvp(
        ode_prob_window,
        K_window,
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
        obstimes_window,
        3 * Δ_βt,
        IFR_means,
        IFR_sds,
        inverse(IFR_bijector)
)
 
model_window = model_window_unconditioned | (y = data.y[:,:,1:length(obstimes_window)],
    z = data.z[1:length(sample_sizes_non_zero_window)]
) 

conditioned_prior = sample(model_window, Prior(), 1)

name_map_correct_order = ode_prior.name_map.parameters
using AMGS

myamgsEE = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsLDN = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsMID = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsNEY = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsNW = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsSE = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)
myamgsSW = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-5)

myamgs_η = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)

myamgs_annoy1 = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)
myamgs_annoy2 = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)
myamgs_annoy3 = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)
myamgs_annoy4 = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)
myamgs_annoy5 = AMGS.new_AMGS_Sampler(0.234, 0.6, true, 1e-6)

global_varnames = (@varname(η),) 

annoyingvarnames_1 = (@varname(d_I),)
annoyingvarnames_2 = (@varname(βₜσ),)
annoyingvarnames_3 = (@varname(IFR_vec_init),)
annoyingvarnames_5 = (@varname(sero_sens), @varname(sero_spec),)
EE_varnames = (@varname(ψ_EE), @varname(log_β_EE), @varname(log_λ₀_EE), @varname(m_EE))
LDN_varnames = (@varname(ψ_LDN), @varname(log_β_LDN), @varname(log_λ₀_LDN), @varname(m_LDN))
MID_varnames = (@varname(ψ_MID), @varname(log_β_MID), @varname(log_λ₀_MID), @varname(m_MID))
NEY_varnames = (@varname(ψ_NEY), @varname(log_β_NEY), @varname(log_λ₀_NEY), @varname(m_NEY))
NW_varnames = (@varname(ψ_NW), @varname(log_β_NW), @varname(log_λ₀_NW), @varname(m_NW))
SE_varnames = (@varname(ψ_SE), @varname(log_β_SE), @varname(log_λ₀_SE), @varname(m_SE))
SW_varnames = (@varname(ψ_SW), @varname(log_β_SW), @varname(log_λ₀_SW), @varname(m_SW))

# const global_regional_gibbs_mine2 = Gibbs(global_varnames => myamgs_η2,
#     EE_varnames => myamgsEE,
#     LDN_varnames => myamgsLDN,
#     MID_varnames => myamgsMID,
#     NEY_varnames => myamgsNEY,
#     NW_varnames => myamgsNW,
#     SE_varnames => myamgsSE,
#     SW_varnames => myamgsSW)


global_regional_gibbs_mine = Gibbs(global_varnames => myamgs_η,
    annoyingvarnames_1 => myamgs_annoy1,
    annoyingvarnames_2 => myamgs_annoy2,
    annoyingvarnames_3 => myamgs_annoy3,
    annoyingvarnames_5 => myamgs_annoy5,
    EE_varnames => myamgsEE,
    LDN_varnames => myamgsLDN,
    MID_varnames => myamgsMID,
    NEY_varnames => myamgsNEY,
    NW_varnames => myamgsNW,
    SE_varnames => myamgsSE,
    SW_varnames => myamgsSW)

t1_compile_time = time_ns()
ode_nuts = sample(
    model_window, 
    global_regional_gibbs_mine,
    MCMCThreads(),
    3,
    n_chains,
    num_warmup = 2,
    # num_warmup = n_warmup_samples,
    thinning = 1)
t2_compile_time = time_ns()
runtime_compile = convert(Int64, t2_compile_time-t1_compile_time)

t1_init = time_ns()
ode_nuts_alt = sample(
    model_window, 
    global_regional_gibbs_mine,
    MCMCThreads(),
    1000,
    n_chains,
    num_warmup = 2000,
    # num_warmup = n_warmup_samples,
    thinning = 20,
    # discard_initial = 377900,
    # discard_initial = 380000,
    discard_initial = 0 
) 
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)


t1_init = time_ns()
ode_nuts = sample(
    model_window,
    global_regional_gibbs_mine,
    MCMCThreads(),
    10000,
    n_chains,
    num_warmup = 10000,
    # num_warmup = n_warmup_samples,
    thinning = 20,
    # discard_initial = 377900,
    # discard_initial = 380000,
    discard_initial = 0 
) 
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)

lj = logjoint(model_window, ode_nuts)

data = generated_quantities(model_window, ode_nuts[end,:,:])[1]

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(1:length(obstimes_window), data.y[:,i,:]', legend = false, alpha = 0.3)
end
StatsPlots.plot(a..., layout = (4,2), size = (1800,1200))

savefig("test1.png")

b = Vector{Plots.Plot}(undef, NA * NR)
sample_sizes_window = sample_sizes[:,:,1:length(obstimes_window)]
positives = similar(sample_sizes_window, Float64)
positives[sus_pop_mask_window] .= data.z ./ sample_sizes_non_zero_window
positives[.!sus_pop_mask_window] .= NaN
for i in 1:NA
    for j in 1:NR
        b[(i-1) * NR + j] = scatter(obstimes_window, positives[i,j,:], title = "NA: $i, NR: $j", legend = false, ylims = (0,1))
    end
end
plot(b..., layout = (NR * 2, 4), size = (2800, 3400))

savefig("test2.png")

c = Vector{Plots.Plot}(undef, NR)
dat_SEIR = stack(data.sol(obstimes_window))
for i in 1:NR
    c[i] = plot(obstimes_window, dat_SEIR[4,:,i,:]' .+ dat_SEIR[5,:,i,:]')
end
plot(c..., layout = (4,2), size = (1800,1200))

savefig("test3.png")

d = Vector{Plots.Plot}(undef, NR)
dat_SEIR = stack(data.sol(obstimes_window))
for i in 1:NR
    plot_part = plot(obstimes_window, dat_SEIR[1,:,i,:]')
    d[i] = plot!(plot_part, obstimes_window, dat_SEIR[6,:,i,:]')
end
plot(d..., layout = (4,2), size = (1800,1200))

savefig("test4.png")

t1_init = time_ns()
ode_nuts = sample(
    model_window,
    global_regional_gibbs_mine,
    MCMCThreads(),
    5000,
    n_chains,
    num_warmup = 2000,
    # num_warmup = n_warmup_samples,
    thinning = 20,
    # discard_initial = 377900,
    # discard_initial = 380000,
    discard_initial = 20000
) 
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)
#105 minutes

# t1_init2 = time_ns()
# ode_nuts2 = sample(
#     model_window,
#     global_regional_gibbs_non_loop,
#     MCMCThreads(),
#     1,
#     n_threads,
#     # num_warmup = 1000,
#     num_warmup = n_warmup_samples,
#     # thinning = 20,
#     # discard_initial = 377900,
#     # discard_initial = 380000,
#     discard_initial = discard_init
# ) 
# t2_init2 = time_ns()
# runtime_init2 = convert(Int64, t2_init2-t1_init2)

lj = logjoint(model_window, ode_nuts)

# Save the chains   
using JLD2
JLD2.jldsave(string(outdir, "chains.jld2"), chains = ode_nuts)
# ode_nuts = JLD2.load(string(outdir, "chains.jld2"))["chains"]

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

# a = Vector{Plots.Plot}(undef, 3 * NR)
# for i in 1:NR
#     a[i] = StatsPlots.plot( stack(map(x -> x[3,:,i], res[1].sol.u))',
#         xlabel="Time",
#         ylabel="Number",
#         linewidth = 1,
#         legend = false)
#     a[i+NR] = StatsPlots.plot(obstimes_window, stack(map(x -> x[3,:,1], sol_ode.u))[:,1:Window_size]')
#     a[i + (2* NR)] = StatsPlots.plot( stack(map(x -> x[3,:,1], sol_ode.u))[:,1:Window_size]' - stack(map(x -> x[3,:,i], res[1].sol.u))',
#         xlabel="Time",
#         ylabel="diff",
#         linewidth = 1,
#         legend = false)
# end
# StatsPlots.plot(a..., layout = (3,NR), size = (1800,2150))

I_tot_win = Array(res[1].sol(obstimes_window))[7,:,:,:]
X_win = rowadjdiff3d(I_tot_win)

Y_win_mu = mapslices(x->(conv_mat_window * (IFR_vec .* x)')', X_win, dims = (1,3))

a = Vector{Plots.Plot}(undef, 2 * NR)
for i in 1:NR
    a[i] = StatsPlots.scatter(obstimes_window, Y[:,i,1:Window_size]', legend = false, alpha = 0.3)
    a[i+NR] = StatsPlots.plot(a[i],obstimes_window, eachrow(Y_win_mu[:,i,:]))
    a[i] = StatsPlots.plot(a[i],obstimes_window, eachrow(Y_mu[:,i,1:Window_size]))
end
StatsPlots.plot(a..., layout = (2,NR), size = (2800,1800))
# plot(ode_nuts[450:end,:,:])

# plot(loglikvals[150:end,:])
# savefig(string("loglikvals_try_fix_I0_$seed_idx.png"))


# Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)

using Statistics

function generate_confint_infec_init(chn, model; cri = 0.95)
    chnm_res = generated_quantities(
        model,
        chn) 

    infecs = stack(map(x -> Array(x.sol)[4,:,:,:] .+ Array(x.sol)[5,:,:,:], chnm_res[1,:]))
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

    infecs = stack(map(x -> Array(x.sol)[6,:,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> Statistics.quantile(x,(1-cri) / 2), infecs, dims = 4)[:,:,:,1]
    medci_inf = mapslices(x -> Statistics.quantile(x, 0.5), infecs, dims = 4)[:,:,:,1]
    uppci_inf = mapslices(x -> Statistics.quantile(x, cri + (1-cri) / 2), infecs, dims = 4)[:,:,:,1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

I_dat = Array(sol_ode(obstimes))[4,:,:,:] .+ Array(sol_ode(obstimes))[5,:,:,:] # Population of infecteds at times
R_dat = Array(sol_ode(obstimes))[6,:,:,:] # Population of recovereds at times


# convert the above code into a function which evaluates the different beta get_beta_quantiles
# TODO Rewrite this function with new β names
function get_beta_quantiles(chn, K_window; quantile = 0.95, NR = NR)
    reg_names = ["EE", "LDN", "MID", "NEY", "NW", "SE", "SW"]
    # β₀_syms = [Symbol("log_β₀_$reg_name") for reg_name in reg_names]
    β_syms = [[Symbol("log_β_$reg_name[$j]") for reg_name in reg_names] for j in 1:K_window - 2]

    tst_arr = Array{Float64}(undef, K_window-1, NR, n_chains)

    for i in 1:NR
        tst_arr[1,i,:] = zeros(Float64, n_chains)
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
algorithm_times[1] = runtime_init * 60

# Construct store for the resulting chains
list_chains = Vector{Chains}(undef, length(each_end_time))
list_chains[1] = ode_nuts

# Define function to determine the names of the parameters in the model for some number of betas
get_params_varnames_fix = function(n_old_betas)
    # reg_names = ["EE", "LDN", "MID", "NEY", "NW", "SE", "SW"]
    params_return = Vector{Turing.VarName}()
    if(n_old_betas >= 0) append!(params_return,
        [
            @varname(logit_I₀_EE),
            @varname(logit_I₀_LDN),
            @varname(logit_I₀_MID),
            @varname(logit_I₀_NEY),
            @varname(logit_I₀_NW),
            @varname(logit_I₀_SE),
            @varname(logit_I₀_SW),
        ]) end
    if(n_old_betas >= 1) append!(params_return,
        [
            @varname(log_β₀_EE),
            @varname(log_β₀_LDN),
            @varname(log_β₀_MID),
            @varname(log_β₀_NEY),
            @varname(log_β₀_NW),
            @varname(log_β₀_SE),
            @varname(log_β₀_SW),
        ]) end
    if(n_old_betas >= 2)
        append!(params_return,
       vec(stack( [[
            @varname(log_β_EE[beta_idx]),
            @varname(log_β_LDN[beta_idx]),
            @varname(log_β_MID[beta_idx]),
            @varname(log_β_NEY[beta_idx]),
            @varname(log_β_NW[beta_idx]),
            @varname(log_β_SE[beta_idx]),
            @varname(log_β_SW[beta_idx]),
        ] for beta_idx in 1:n_old_betas-1])))
    end
         
    return params_return
end

get_params_varnames_all = function(n_old_betas)
    params_return = Vector{Turing.VarName}()
    if(n_old_betas >= 0) append!(params_return,
        [
            @varname(logit_I₀_EE),
            @varname(logit_I₀_LDN),
            @varname(logit_I₀_MID),
            @varname(logit_I₀_NEY),
            @varname(logit_I₀_NW),
            @varname(logit_I₀_SE),
            @varname(logit_I₀_SW),
        ], [@varname(η)]) end
    if(n_old_betas >= 1) append!(params_return,
        [
            @varname(log_β₀_EE),
            @varname(log_β₀_LDN),
            @varname(log_β₀_MID),
            @varname(log_β₀_NEY),
            @varname(log_β₀_NW),
            @varname(log_β₀_SE),
            @varname(log_β₀_SW),
        ]) end
        if(n_old_betas >= 2)
            append!(params_return,
           vec(stack( [[
                @varname(log_β_EE[beta_idx]),
                @varname(log_β_LDN[beta_idx]),
                @varname(log_β_MID[beta_idx]),
                @varname(log_β_NEY[beta_idx]),
                @varname(log_β_NW[beta_idx]),
                @varname(log_β_SE[beta_idx]),
                @varname(log_β_SW[beta_idx]),
            ] for beta_idx in 1:n_old_betas-1])))
        end
         
    return params_return
end


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

    local sus_pop_mask_window = sus_pop_mask[:,:,1:curr_t]
    local sample_sizes_non_zero_window = @view sample_sizes[:,:,1:curr_t][sus_pop_mask_window]

    window_param_names = get_params_varnames_all(Int(ceil(curr_t / Δ_βt)))
    fixed_param_names = get_params_varnames_fix(n_old_betas)

    y_data_window = Y[:,:,1:curr_t]
    z_data_window = 💉[1:length(sample_sizes_non_zero_window)]

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
        trans_unconstrained_I0
    )

    local model_window = 
        model_window_unconditioned |
            (y = y_data_window,
                z = z_data_window,
            )

    # mydramgs = DRAMGS_Sampler(0.234, 0.6; noise_stabiliser = true, noise_stabiliser_scaling = 1e-5, max_delayed_rejection_steps = 2, rejection_step_sizes = [0.6, 0.1])

    
    local global_varnames = (@varname(η),)
    if n_old_betas == 0
        local EE_varnames = (@varname(log_β₀_EE), @varname(log_β_EE))
        local LDN_varnames = (@varname(log_β₀_LDN), @varname(log_β_LDN))
        local MID_varnames = (@varname(log_β₀_MID), @varname(log_β_MID))
        local NEY_varnames = (@varname(log_β₀_NEY), @varname(log_β_NEY))
        local NW_varnames = (@varname(log_β₀_NW), @varname(log_β_NW))
        local SE_varnames = (@varname(log_β₀_SE), @varname(log_β_SE))
        local SW_varnames = (@varname(log_β₀_SW), @varname(log_β_SW))
    else
        local EE_varnames = ( @varname(log_β_EE),)
        local LDN_varnames = ( @varname(log_β_LDN),)
        local MID_varnames = ( @varname(log_β_MID),)
        local NEY_varnames = ( @varname(log_β_NEY),)
        local NW_varnames = ( @varname(log_β_NW),)
        local SE_varnames = ( @varname(log_β_SE),)
        local SW_varnames = ( @varname(log_β_SW),)
    end

    local global_regional_gibbs_mine = Gibbs(
        global_varnames => myamgs_η,
        EE_varnames => myamgsEE,
        LDN_varnames => myamgsLDN,
        MID_varnames => myamgsMID,
        NEY_varnames => myamgsNEY,
        NW_varnames => myamgsNW,
        SE_varnames => myamgsSE,
        SW_varnames => myamgsSW
    )

    t1 = time_ns()
    list_chains[idx_time] = sample(
        model_window,
        PracticalFiltering.PracticalFilter(
            fixed_param_names,
            window_param_names,
            list_chains[idx_time - 1][end,:,:],
            global_regional_gibbs_mine
        ),
        MCMCThreads(),
        1,
        # 2000,
        n_chains;
        discard_initial = discard_init,
        # discard_initial = 1500 * 20,
        # num_warmup = 1000,
        num_warmup = n_warmup_samples
        # thinning = 20
    )
    t2 = time_ns()
    algorithm_times[idx_time] = convert(Int64, t2 - t1)

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
obs_exp_window = obs_exp[:,:,1:curr_t]
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
    # "algorithm_times_each_sample" => algorithm_times_each_sample
    )

open(string(outdir, "timings.toml"), "w") do io
        TOML.print(io, params)
end


# Save all chains
jldsave(string(outdir, "chains.jld2"), chains =  list_chains)
