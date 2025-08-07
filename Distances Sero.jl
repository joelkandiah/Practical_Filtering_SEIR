using StatsBase
using Distances
using LinearAlgebra
using StatsPlots
using Random
using Distributions
using EmpiricalDistributions
using Turing 
using JLD2
using DelimitedFiles
using DataInterpolations
using OrdinaryDiffEq
using Plots
using TOML

seed_idx = parse(Int, ARGS[1])
Δ_βt = parse(Int, ARGS[2])
Window_size = parse(Int, ARGS[3])
Data_update_window = parse(Int, ARGS[4])
tmax = parse(Float64, ARGS[5])

tmax_int = Integer(tmax)

no_win_chains = load("Results/SEIR_MASS_ACTION_SERO_AMGS_Final/40_beta/no_window/chains_6/n_threads_6/Plot attempt $seed_idx/chains.jld2")["chains"]

win_chains = load("/home/joelk/ProgrammingProjects/julia/Practical_Filtering_SEIR_Final/Results/SEIR_MASS_ACTION_SERO_AMGS/40_beta/window_180/chains_200/n_threads_25/Plot attempt $seed_idx/chains.jld2")["chains"]

outdir = "/home/joelk/ProgrammingProjects/julia/Practical_Filtering_SEIR_Final/Results/SEIR_MASS_ACTION_SERO_AMGS/40_beta/cf/Plot attempt $seed_idx/"

if !isdir(outdir)
    mkpath(outdir)
end

# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# # Set and Create locations to save the plots and Chains
# outdir = string("Results/SEIR_MASS_ACTION_SERO/$Δ_βt","_beta/no_window/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")
# tmpstore = string("Chains/SEIR_MASS_ACTION_SERO/$Δ_βt","_beta/no_window/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")

# if !isdir(outdir)
#     mkpath(outdir)
# end
# if !isdir(tmpstore)
#     mkpath(tmpstore)
# end

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
# params = Dict(
#     "seed" => seeds_list[seed_idx],
#     "n_chains" => n_chains,
#     "discard_init" => discard_init,
#     "tmax" => tmax,
#     "N" => N,
#     "I0" => I0,
#     "γ" => γ,
#     "σ" => σ,
#     "β" => true_beta,
#     "Δ_βt" => Δ_βt,
#     "knots" => knots,
#     "inital β mean" => β₀μ,
#     "initial β sd" => β₀σ,
#     "β sd" => βσ,
#     "Gamma alpha" => inf_to_hosp.α,
#     "Gamma theta" => inf_to_hosp.θ
# )

# open(string(outdir, "params.toml"), "w") do io
#         TOML.print(io, params)
# end

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

# Sample the parameters to construct a "correct order" list of parameters
ode_prior = sample(bayes_sir_tvp(K,
        γ,
        σ,
        N,
        NA,
        NA_N,
        NR,
        conv_mat,
        knots,
        C,
        sample_sizes_non_zero,
        sus_pop_mask,
        sero_sens,
        sero_spec,
        obstimes,
        I0_μ_prior,
        β₀μ,
        β₀σ,
        βσ,
        IFR_vec,
        trans_unconstrained_I0,
    ) | (y = Y,
     z = 💉,
     ), Prior(), 1, discard_initial = 0, thinning = 1);


# Create an array to store the window ends to know when to run the model
each_end_time = collect(Window_size:Data_update_window:tmax)
each_end_time = each_end_time[end] ≈ tmax ? each_end_time : vcat(each_end_time, tmax)
each_end_time = Integer.(each_end_time)

model_list = Vector{Turing.Model}(undef, length(each_end_time))


for idx_time in eachindex(each_end_time)
    
    # Determine which betas to fix and which ones to sample
    curr_t = each_end_time[idx_time]
    n_old_betas = Int(floor((curr_t - Window_size) / Δ_βt))
    window_betas = Int(ceil(curr_t / Δ_βt) - n_old_betas)

    # Set the values for the current window: knots, K, convolution matrix...
    knots_window = collect(0:Δ_βt:curr_t)
    knots_window = knots_window[end] != curr_t ? vcat(knots_window, curr_t) : knots_window
    K_window = length(knots_window)
    conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, curr_t)
    obstimes_window = 1.0:1.0:curr_t

    sus_pop_mask_window = sus_pop_mask[:,1:length(obstimes_window)]
    sample_sizes_non_zero_window = @view sample_sizes[:,1:length(obstimes_window)][sus_pop_mask_window]

    y_data_window = Y[:,1:length(obstimes_window)]
    z_data_window = 💉[1:length(sample_sizes_non_zero_window)]

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

    model_window = model_window_unconditioned| (y = y_data_window,
        z = z_data_window,
    )

    model_list[idx_time] = model_window
end

list_log_dens_win = [logjoint(model_list[i], win_chains[i])[1,:] for i in 1:length(model_list)]
list_log_dens_no_win = [logjoint(model_list[i], no_win_chains[i]) for i in 1:length(model_list)]


# list_log_dens_no_win[1]
# list_log_dens_win[1]


plot_vec = Vector{Plots.Plot}(undef, length(model_list))

for i in eachindex(model_list)
    plot_part = histogram(list_log_dens_no_win[i][:,1], normalize = :pdf)
    for j in 2:size(list_log_dens_no_win[i],2)
        histogram!(plot_part, list_log_dens_no_win[i][:,j], normalize = :pdf, alpha = 0.5)
    end
    plot_vec[i] = histogram!(plot_part, list_log_dens_win[i], alpha = 0.5, normalize = :pdf)
end
plot(plot_vec..., layout = (3,2), size = (1200,800))


for i in eachindex(model_list)
    # Create histograms
    list_hists = Vector{Histogram}(undef, size(list_log_dens_no_win[i],2) + 1)
    for j in 1:size(list_log_dens_no_win[i],2)
        list_hists[j] = fit(Histogram, list_log_dens_no_win[i][:,j]; nbins = 20)
    end
    list_hists[end] = fit(Histogram, list_log_dens_win[i]; nbins = 20)

    lower_bound = minimum([first(list_hists[j].edges[1]) for j in eachindex(list_hists)])
    upper_bound = maximum([last(list_hists[j].edges[1]) for j in eachindex(list_hists)])
    step_min = minimum([step(list_hists[j].edges[1]) for j in eachindex(list_hists)])

    # Fit histograms over the same set of bins
    same_bins_list_hists = Vector{Histogram}(undef, size(list_log_dens_no_win[i],2))
    for j in 1:size(list_log_dens_no_win[i],2)
        same_bins_list_hists[j] = fit(Histogram, list_log_dens_no_win[i][:,j], lower_bound:step_min:upper_bound)
    end
    same_bins_hist_win = fit(Histogram, list_log_dens_win[i], lower_bound:step_min:upper_bound)

    # Normalise histograms
    norm_list_hists = Vector{Histogram}(undef, size(list_log_dens_no_win[i],2))
    for j in eachindex(norm_list_hists)
        norm_list_hists[j] = normalize(same_bins_list_hists[j], mode = :pdf)
    end
    norm_hist_win = normalize(same_bins_hist_win, mode = :pdf)

    # Plot histograms
    plots_list = Vector{Plots.Plot}(undef, size(list_log_dens_no_win[i],2))
    for j in eachindex(plots_list)
        plot_part = plot(norm_list_hists[j], label = "No Window Chain $j" )
        plots_list[j] = plot!(plot_part, norm_hist_win, alpha = 0.5, label = "Window Chain")
    end
    plot(plots_list..., layout = (3,2), size = (1200,800))

    savefig(string(outdir, "window_period_", i, ".png"))

    # Add small value to all bins
    for j in eachindex(norm_list_hists)
        norm_list_hists[j].weights = norm_list_hists[j].weights .+ 1e-11
    end
    norm_hist_win.weights = norm_hist_win.weights .+ 1e-11

    # renormalise the weights
    for j in eachindex(norm_list_hists)
        norm_list_hists[j].weights = norm_list_hists[j].weights ./ sum(norm_list_hists[j].weights)
    end
    norm_hist_win.weights = norm_hist_win.weights ./ sum(norm_hist_win.weights)

    # Evaluate Pairwise Kullback-Leibler Divergence between no win chains
    kldivs_within =[
        kldivergence(
            UvBinnedDist(norm_list_hists[j]),
            UvBinnedDist(norm_list_hists[k]))
            for j in eachindex(norm_list_hists)
            for k in j+1:length(norm_list_hists)
    ] 


    kldivs_outer = Vector{Float64}(undef, size(list_log_dens_no_win[i],2))
    for j in eachindex(kldivs_outer)
        d1 = UvBinnedDist(norm_list_hists[j])
        d2 = UvBinnedDist(norm_hist_win)
        kldivs_outer[j] = kldivergence(d1, d2)
    end

    # Perform the same for the euclidean distance (wasserstein distance)
    euclidean_within = [
        evaluate(Euclidean(), list_log_dens_no_win[i][:,j], list_log_dens_no_win[i][:,k])
        for j in eachindex(norm_list_hists)
        for k in j+1:length(norm_list_hists)
    ] 

    euclidean_outer = [
        evaluate(Euclidean(), list_log_dens_no_win[i][:,j], list_log_dens_win[i])
        for j in eachindex(norm_list_hists)
    ]

    res = Dict(
        "KLDivs_within" => kldivs_within,
        "KLDivs_outer" => kldivs_outer,
        "Euclidean_within" => euclidean_within,
        "Euclidean_outer" => euclidean_outer
    )

    open(string(outdir, "results_window_$i.toml"), "w") do io
        TOML.print(io, res)
    end
    
end

# Save the Kullback-Leibler Divergence and Euclidean Distance values to a toml

