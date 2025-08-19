using AMGS: AdvancedMH
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
@assert length(ARGS) == 3

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
const seed_array_idx = parse(Int, ARGS[3])
 
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
    (γ2, σ2, ld_start, Δt_C) = p_.params_floats

    copyto!(p_.cache_C, t <= ld_start ? p_.C_base : @view p_.C[Int(ceil(t / Δt_C)) + 2,:,:,:])
    
    # Calculate the force of infection
    map!(x-> x(t), p_.cache_beta[1], p_.β_functions)
    
    @tullio grad = false threads = false p_.cache_SIRderivs[6][i,k] = (I1[j,k] + I2[j,k]) * log1p(-(p_.cache_beta[1][k] * p_.cache_C[k,i,j])) |> (1 - exp((_)))

    @inbounds @simd for k in axes(S, 2)
        for i in axes(S, 1)
            p_.cache_SIRderivs[1][i,k] = S[i,k] * p_.cache_SIRderivs[6][i,k]
            p_.cache_SIRderivs[2][i,k] = σ2 * E1[i,k]
            p_.cache_SIRderivs[3][i,k] = σ2 * E2[i,k]
            p_.cache_SIRderivs[4][i,k] = γ2 * I1[i,k]
            p_.cache_SIRderivs[5][i,k] = γ2 * I2[i,k]
        end
    end

    @inbounds begin
        du[1,:,:] .= .- p_.cache_SIRderivs[1] 
        du[2,:,:] .= p_.cache_SIRderivs[1] .- p_.cache_SIRderivs[2]
        du[3,:,:] .= p_.cache_SIRderivs[2] .- p_.cache_SIRderivs[3]
        du[4,:,:] .= p_.cache_SIRderivs[3] .- p_.cache_SIRderivs[4]
        du[5,:,:] .= p_.cache_SIRderivs[4] .- p_.cache_SIRderivs[5]
        du[6,:,:] .= p_.cache_SIRderivs[5]
        du[7,:,:] .= p_.cache_SIRderivs[1]
    end
end;

struct idd_params{T <: Real, T2 <: Real, T3 <: DataInterpolations.AbstractInterpolation, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real}
    params_floats::Vector{T}
    NR_pops::Vector{T2}
    β_functions::Vector{T3}
    C_base::Array{T4}
    C::Array{T4}
    cache_SIRderivs::Vector{Matrix{T6}}
    cache_beta::Vector{Vector{T7}}
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
    return A
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

@model function β_prior(K, β_σ, ::Type{T_β} = Float64) where {T_β <: Real}
    β = Vector{T_β}(undef,K)

    log_β₀ = zero(T_β)
    β[1] = exp(log_β₀)
    
    log_β = Vector{T_β}(undef, K-2)
    for i in 2:K-1
        log_β[i-1] ~ Normal(0.0, β_σ)
        β[i] = exp.(log.(β[i-1]) .+ log_β[i-1])
    end
   
    β[K] = β[K-1]

    return(β)
end

@model function sus_prior(assign_mats, NA, n_groups, ::Type{T_sus_M} = Float64) where {T_sus_M <: Real}
    sus_M = Array{T_sus_M}(undef, 2, NA, NA)
    sus_M_pars = Vector{T_sus_M}(undef, n_groups)
    all_sus_M_pars = Vector{T_sus_M}(undef, n_groups+1)
    all_sus_M_pars[1] = one(T_sus_M)
    for j in 1:n_groups
        sus_M_pars[j] ~ LogNormal(-0.111571775657105, 0.472380727077439) # This is the prior for the first 3 age groups
    end
    all_sus_M_pars[2:end] .= sus_M_pars
    sus_M .= all_sus_M_pars[assign_mats .+ 1]
    return(sus_M)
end

sus_M_locs = stack(
    [
        readdlm("ContactMats/multmod/ag8_mult_mod4levels0.txt", Int64),
        readdlm("ContactMats/multmod/ag8_mult_mod4levels1.txt", Int64)
    ], dims = 1)

# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp(
    ode_prob,
    K,
    σ = σ,
    p_GP = 0.1,
    N = N_pop_R,
    NA = NA,
    NA_N = NA_N,
    N_regions = NR,
    conv_mat = conv_mat,
    knots = knots,
    C_base = C_base,
    C = C_ldwks,
    eigmax_C_base = real(eigmax(C_base)),
    sero_sample_sizes = sample_sizes_non_zero,
    sus_pop_mask = sus_pop_mask,
    obstimes = obstimes,
    ld_start = 35,
    Δt_C = 7,
    sus_M_locs = sus_M_locs,
    N_M_groups = maximum(sus_M_locs),
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
    βₜσ ~ Gamma(1,1/100)

    β_EE ~ to_submodel(β_prior(K, βₜσ))
    β_LDN ~ to_submodel(β_prior(K, βₜσ))
    β_MID ~ to_submodel(β_prior(K, βₜσ))
    β_NEY ~ to_submodel(β_prior(K, βₜσ))
    β_NW ~ to_submodel(β_prior(K, βₜσ))
    β_SE ~ to_submodel(β_prior(K, βₜσ))
    β_SW ~ to_submodel(β_prior(K, βₜσ))

    β[:,1] = β_EE
    β[:,2] = β_LDN
    β[:,3] = β_MID
    β[:,4] = β_NEY
    β[:,5] = β_NW
    β[:,6] = β_SE
    β[:,7] = β_SW
    
    sero_sens  ~ Beta(71.5,29.5)
    sero_spec  ~ Beta(777.5,9.5)

    d_I  ~ Gamma(1.43,1/0.549)
    inv_γ = d_I + 2
    γ  = 1 / inv_γ

    p = [2 * γ, 2 * σ, ld_start, Δt_C]
   
    # Make more generic (pass indices)
    sus_M  = Array{T_sus_M}(undef, 2, NR, NA, NA)
    sus_M_EE ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_LDN ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_MID ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_NEY ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_NW ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_SE ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))
    sus_M_SW ~ to_submodel(sus_prior(sus_M_locs, NA, N_M_groups))

    sus_M[:,1,:,:] = sus_M_EE
    sus_M[:,2,:,:] = sus_M_LDN
    sus_M[:,3,:,:] = sus_M_MID
    sus_M[:,4,:,:] = sus_M_NEY
    sus_M[:,5,:,:] = sus_M_NW
    sus_M[:,6,:,:] = sus_M_SE
    sus_M[:,7,:,:] = sus_M_SW
    
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

    λ₀ .= λ₀ ./ p_GP

    # Write the priors for the IFR
    IFR_vec = Vector{T_IFR}(undef, NA)
    IFR_vec_init = Vector{T_IFR}(undef, NA - 1)
    @inbounds for i in 1:(NA-1)
        IFR_vec_init[i] ~ Normal(IFR_means[i], IFR_sds[i])
    end

    IFR_vec[2:end] .= trans_unconstrained_IFR.(IFR_vec_init)
    IFR_vec[1] = IFR_vec[2]
    
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
    cache_sus_M = zeros(T_sus_M, N_regions, NA, NA)

    inv_σ = 1 / σ
    R0_num = @. 1 + (ψ * inv_σ * 0.5) 
    R0_denom = @. 1 - (1 / ((1 + (ψ * inv_γ * 0.5))^2))
    a0_num = @. ψ * inv_γ * (R0_num^2) ./ R0_denom
    
    C_base_norm  = (C_base ./ eigmax_C_base)
    Ĉ_base = Array{T_sus_M}(undef, N_regions, NA, NA)
    @inbounds for k in 1:NR
        @. Ĉ_base[k, :, :] = sus_M[1,k,:,:] * C_base_norm 
    end
    Ĉ = Array{T_sus_M}(undef, size(C)[1], NR, NA, NA)
    for j in 1:size(C)[1]
        tmp = (C[j,:,:] / eigmax_C_base)
        for k in 1:N_regions
            @inbounds @. Ĉ[j, k, :, :] = sus_M[2,k,:,:] * tmp
        end
    end

    # Add priors for p_lambda_0
    a0_denom = similar(a0_num)
    for k in 1:NR
        @inbounds a0_denom[k] = β_functions[k](0) * inv_γ * real(eigmax(Ĉ_base[k,:,:] .* NA_N[:,k])) 
    end
    a0 = a0_num ./ a0_denom

    @tullio grad=false threads=false b0[i,k] := log(1 - (a0[k] * β_functions[k](0) * Ĉ_base[k,i,j])) |> (1 - exp((_)))
    tot_I₀ = inv_γ * map(k -> λ₀[k] .* sum(NA_N[:,k]) / a0_num[k], 1:N_regions)

    # Faster max eigenvector eval and normalise
    χ = Array{T_sus_M}(undef, NA, N_regions)
    tmp_eig = Array{T_sus_M}(undef, NA, NA)    # reusable matrix workspace
    eigres = Eigen(tmp_eig[1,:], tmp_eig)# preallocate structure 
    eigvec = Vector{T_sus_M}(undef, NA)    # reusable vector
     
    @inbounds for k in 1:N_regions
         # Fill tmp in-place
         @. tmp_eig = Ĉ_base[k, :, :] * NA_N[:, k]
     
    #     # Recompute eigenvalues/vectors
         eigres = eigen(tmp_eig)  
    # 
    #     # Find index of max abs eigenvalue
         idx = argmax(abs.(eigres.values))
    # 
    #     # Get corresponding eigenvector
         @. eigvec = abs(eigres.vectors[:, idx])
    # 
    #     # Normalize in-place
         χ[:, k] .= eigvec ./ sum(eigvec)
    end

    tot_I₀_ages = χ .* transpose(tot_I₀)
    u0 = zeros(T_u0,7,NA,NR)
    α = exp.(ψ) .- 1
    ϵ = 1
    u0[4,:,:] .= tot_I₀_ages .* inv.(transpose(1 .+ ((ϵ .* γ) ./ (α .+ γ))))
    u0[5,:,:] .= tot_I₀_ages .- @view u0[4,:,:]
    u0[3,:,:] .= transpose((α .+ γ) ./ σ) .* @view(u0[4,:,:])
    u0[2,:,:] .= transpose((α .+ σ) ./ σ) .* @view(u0[3,:,:])
    u0[1,:,:] .= NA_N .- tot_I₀_ages .- @view(u0[2,:,:]) .- @view u0[3,:,:]
   
    aCbase = similar(Ĉ_base)
    aCbase .= a0 .* Ĉ_base

    aC = similar(Ĉ)
    for k in 1:N_regions
        @. aC[:, k, :, :] = a0[k] * Ĉ[:, k, :, :]
    end

    params_test = idd_params(p, N, β_functions, aCbase, aC, cache_SIRderivs, cache_beta, cache_sus_M)

    # Run model
    ## Remake with new initial conditions and parameter values
    prob = remake(ode_prob, u0 = u0, p = params_test)

    ## Solve
    sol = 
        solve(prob,
               Euler(),
               dt = 0.5,
               adaptive = false,
            # Tsit5(),
            saveat = obstimes,
            d_discontinuities = knots[2:end-1],
            tstops = knots)

    if any(!SciMLBase.successful_retcode(sol))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    ## Calculate new infections per day, X
    sol_Array = Array(sol)
    sol_X =  sol_Array[7,:,:,:] |>
        rowadjdiff3d
    
    if (any(sol_X .<= -(1e-3)) | any(stack(sol.u)[4,:,:,:] .<= -(1e-3)) | any(stack(sol.u)[5,:,:,:] .<= -1e-3))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    y_μ = similar(sol_X) 
    scaled = IFR_vec .* sol_X
    scaled_2d = reshape(scaled, NA * N_regions, length(obstimes))
    y_μ .= reshape(scaled_2d * transpose(conv_mat), NA, N_regions, length(obstimes))

    ## Calculate number of timepoints
    if (any(isnan.(y_μ)))
        @DynamicPPL.addlogprob! -Inf
        return
    end
    
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, η))

    # Introduce Serological data into the model (for the first region)
    sus_pop = @view sol_Array[1,:,:,:]
    sus_pop_samps = @view (sus_pop[:,:,:] ./ NA_N)[sus_pop_mask]
    
    z ~ product_distribution(@. Binomial(
        sero_sample_sizes,
        (sero_sens * (1 - (sus_pop_samps))) + ((1 - sero_spec) * (sus_pop_samps))
    ))
    
    # Rt = calculate_Rt(N_regions, sol, obstimes, ld_start, inv_γ, Ĉ_base, Ĉ, Δt_C, 2, a0, β_functions, T_Seir)

    return (; sol, y_μ, y, z, a0_num, β, β_functions, sus_M)
end;

function calculate_Rt(NR, sol, obstimes, ld_start, inv_γ, Ĉ_base, Ĉ, Δt_C, offset, a0, β_functions, T_SEIR)
    Rt = zeros(T_SEIR, NR, length(obstimes))
    S = Array(sol)[1,:,:,:]
    C_i = zeros(NA,NA)
    for i in 1:length(obstimes)
        for j in 1:NR
            copyto!(C_i, i <= ld_start ? Ĉ_base[j,:,:] : Ĉ[Int(ceil(i / Δt_C)) + offset, j,:,:])
            @inbounds Rt[j,i] = a0[j] * β_functions[j](i) * inv_γ * real(eigmax(C_i .* S[:,j,i]))
        end
    end
    return Rt
end

post_params = TOML.parsefile("posterior_inputs/mod_pars_$seed_idx.toml")

knots_init = vcat([0],post_params["log_beta_rw"]["time_breakpoints"])
K_init = length(knots_init)
knots_init = (knots_init[end] != tmax)  && (knots_init[end] < tmax) ? vcat(knots_init, tmax) : knots_init

my_infection = zeros(NA, NR)
my_infectious_1 = zeros(NA, NR)
my_infectious_2  = zeros(NA, NR)
my_recovery_1 = zeros(NA, NR)
my_recovery_2 = zeros(NA, NR)
my_βt = zeros(NR)
my_b = zeros(NA, NR)
my_sus_M = zeros(Float64, NR, NA, NA)

u0 = zeros(Float64, 7, NA, NR)
u0[1,:,:] = NA_N

params_window = idd_params([0.0, 0.0, 35.0, 7.0], 
    N_pop_R,
    map(x -> ConstantInterpolation(x, knots_init), eachcol(ones(K_init, NR))),
    zeros(NR,NA,NA),
    zeros(size(C_ldwks)[1], NR, NA, NA),
    [my_infection, my_infectious_1, my_infectious_2, my_recovery_1, my_recovery_2, my_b],
    [my_βt,],
    my_sus_M)

ode_prob_init = ODEProblem{true}(sir_tvp_ode!, deepcopy(u0), (0.0, tmax), deepcopy(params_window))

const σ = 1 / 3.0

rw_length = K_init

new_knots = knots_init[knots_init .<= tmax]
new_knots = new_knots[end] == tmax ? new_knots : vcat(new_knots, tmax)
K = length(new_knots)
p_GP = post_params["prop_case_to_GP_consultation"]["param_value"]
init_model = bayes_sir_tvp(
        ode_prob_init,
        K,
        σ,
        p_GP,
        N_pop_R,
        NA,
        NA_N,
        NR,
        conv_mat,
        new_knots,
        C_base,
        C_ldwks,
        real(eigmax(C_base)),
        sample_sizes_non_zero,
        sus_pop_mask,
        obstimes,
        36.0,
        7.0,
        sus_M_locs,
        maximum(sus_M_locs),
        IFR_means,
        IFR_sds,
        inverse(IFR_bijector))

N_M_vals = maximum(sus_M_locs) + 1

conditions = Dict(
    @varname(η) => post_params["hosp_negbin_overdispersion"]["param_value"],
    @varname(ψ_EE) => post_params["exponential_growth_rate"]["param_value"][1],
    @varname(ψ_LDN) => post_params["exponential_growth_rate"]["param_value"][2],
    @varname(ψ_MID) => post_params["exponential_growth_rate"]["param_value"][3],
    @varname(ψ_NEY) => post_params["exponential_growth_rate"]["param_value"][4],
    @varname(ψ_NW) => post_params["exponential_growth_rate"]["param_value"][5],
    @varname(ψ_SE) => post_params["exponential_growth_rate"]["param_value"][6],
    @varname(ψ_SW) => post_params["exponential_growth_rate"]["param_value"][7],
    @varname(β_EE.log_β) => post_params["log_beta_rw"]["param_value"][2:rw_length][1:K-2],
    @varname(β_LDN.log_β) => post_params["log_beta_rw"]["param_value"][rw_length+2:2*rw_length][1:K-2],
    @varname(β_MID.log_β) => post_params["log_beta_rw"]["param_value"][2*rw_length+2:3*rw_length][1:K-2],
    @varname(β_NEY.log_β) => post_params["log_beta_rw"]["param_value"][3*rw_length+2:4*rw_length][1:K-2],
    @varname(β_NW.log_β) => post_params["log_beta_rw"]["param_value"][4*rw_length+2:5*rw_length][1:K-2],
    @varname(β_SE.log_β) => post_params["log_beta_rw"]["param_value"][5*rw_length+2:6*rw_length][1:K-2],
    @varname(β_SW.log_β) => post_params["log_beta_rw"]["param_value"][6*rw_length+2:7*rw_length][1:K-2],
    @varname(sero_sens) => post_params["sero_test_sensitivity"]["param_value"][1],  
    @varname(sero_spec) => post_params["sero_test_specificity"]["param_value"][1], 
    @varname(d_I) => post_params["infectious_period"]["param_value"],
    @varname(βₜσ) => post_params["log_beta_rw_sd"]["param_value"][2],
   @varname(sus_M_EE.sus_M_pars) => post_params["contact_parameters"]["param_value"][2:N_M_vals] .|> exp,
   @varname(sus_M_LDN.sus_M_pars) => post_params["contact_parameters"]["param_value"][N_M_vals+2:2*N_M_vals] .|> exp,
   @varname(sus_M_MID.sus_M_pars) => post_params["contact_parameters"]["param_value"][2*N_M_vals+2:3*N_M_vals] .|> exp,
   @varname(sus_M_NEY.sus_M_pars) => post_params["contact_parameters"]["param_value"][3*N_M_vals+2:4*N_M_vals] .|> exp,
   @varname(sus_M_NW.sus_M_pars) => post_params["contact_parameters"]["param_value"][4*N_M_vals+2:5*N_M_vals] .|> exp,
   @varname(sus_M_SE.sus_M_pars) => post_params["contact_parameters"]["param_value"][5*N_M_vals+2:6*N_M_vals] .|> exp,
   @varname(sus_M_SW.sus_M_pars) => post_params["contact_parameters"]["param_value"][6*N_M_vals+2:end] .|> exp,
    @varname(log_λ₀_EE) => post_params["log_p_lambda_0"]["param_value"][1],
    @varname(log_λ₀_LDN) => post_params["log_p_lambda_0"]["param_value"][2],
    @varname(log_λ₀_MID) => post_params["log_p_lambda_0"]["param_value"][3],
    @varname(log_λ₀_NEY) => post_params["log_p_lambda_0"]["param_value"][4],
    @varname(log_λ₀_NW) => post_params["log_p_lambda_0"]["param_value"][5],
    @varname(log_λ₀_SE) => post_params["log_p_lambda_0"]["param_value"][6],
    @varname(log_λ₀_SW) => post_params["log_p_lambda_0"]["param_value"][7],
    @varname(IFR_vec_init) => post_params["prop_case_to_hosp"]["param_value"][1:NA-1] .|> IFR_bijector)

ode_prior = sample(init_model | conditions, Prior(), 1)

data = generated_quantities(init_model | conditions, ode_prior)[1]

a = Vector{Plots.Plot}(undef, NR)
for i in 1:NR
    tmp= StatsPlots.scatter(1:length(obstimes), data2.y[:,i,:]', alpha = 0.3)
    a[i] = StatsPlots.scatter!(tmp, data.y[:,i,:]', alpha = 1)
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
dat_SEIR = stack(data2.sol(obstimes))
dat_SEIR2 = stack(data.sol(obstimes))
for i in 1:NR
    tmp = plot(obstimes, rowadjdiff3d(dat_SEIR2[7,:,:,:])[7,i,:])
    e[i] = plot(tmp, rowadjdiff3d(dat_SEIR[7,:,:,:])[7,i,:])
end
plot(e..., layout = (4,2), size = (1800,1200))


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
EE_varnames = (@varname(ψ_EE), @varname(β_EE.log_β), @varname(sus_M_EE.sus_M_pars), @varname(log_λ₀_EE))
LDN_varnames = (@varname(ψ_LDN), @varname(β_LDN.log_β), @varname(log_λ₀_LDN), @varname(sus_M_LDN.sus_M_pars), @varname(log_λ₀_LDN))
MID_varnames = (@varname(ψ_MID), @varname(β_MID.log_β), @varname(log_λ₀_MID), @varname(sus_M_MID.sus_M_pars), @varname(log_λ₀_MID))
NEY_varnames = (@varname(ψ_NEY), @varname(β_NEY.log_β), @varname(log_λ₀_NEY), @varname(sus_M_NEY.sus_M_pars), @varname(log_λ₀_NEY))
NW_varnames = (@varname(ψ_NW), @varname(β_NW.log_β), @varname(log_λ₀_NW), @varname(sus_M_NW.sus_M_pars), @varname(log_λ₀_NW))
SE_varnames = (@varname(ψ_SE), @varname(β_SE.log_β), @varname(log_λ₀_SE), @varname(sus_M_SE.sus_M_pars), @varname(log_λ₀_SE))
SW_varnames = (@varname(ψ_SW), @varname(β_SW.log_β), @varname(log_λ₀_SW), @varname(sus_M_SW.sus_M_pars), @varname(log_λ₀_SW))

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

model_window = init_model | (y = data.y,
    z = data.z
)

t1_compile_time = time_ns()
ode_nuts = sample(
    model_window, 
    global_regional_gibbs_mine,
    MCMCSerial(),
    3,
    n_chains,
    num_warmup = 2,
    thinning = 1,
    discard_initial = 0)
t2_compile_time = time_ns()
runtime_compile = convert(Int64, t2_compile_time-t1_compile_time)

t1_init = time_ns()
ode_nuts_alt = sample(
    model_window, 
    global_regional_gibbs_mine,
    MCMCSerial(),
    30000,
    n_chains,
    num_warmup = 2000,
    thinning = 1,
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
