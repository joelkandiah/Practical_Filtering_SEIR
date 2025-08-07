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
# using PracticalFiltering
# using DelimitedFiles

@assert length(ARGS) == 7

# Read parameters from command line for the varied submission scripts
seed_idx = parse(Int, ARGS[1])
Δ_βt = parse(Int, ARGS[2])
Window_size = parse(Int, ARGS[3])
Data_update_window = parse(Int, ARGS[4])
n_chains = parse(Int, ARGS[5])
discard_init = parse(Int, ARGS[6])
tmax = parse(Float64, ARGS[7])

# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
outdir = string("Results/10bp 1e6/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")
tmpstore = string("Chains/10bp 1e6/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/n_threads_$n_threads/Plot attempt $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
# NA_N= [74103.0, 318183.0, 804260.0, 704025.0, 1634429.0, 1697206.0, 683583.0, 577399.0]
NA = 1
N = 1_000_000
NA_N = [N]
I0 = [100]
u0 = zeros(NA,5)
u0[:,1] = NA_N - I0
u0[:,3] = I0
inv_γ = 10  
inv_σ = 3
γ = 1/ inv_γ
σ = 1/ inv_σ
p = [γ, σ, N];

I0_μ_prior_orig = log.(I0 ./ N)
I0_μ_prior = -9.0

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
    It = sum(I)
    for a in axes(du,2)
        local infection = (1.0 - exp(It * log(1.0 - βt / N))) * S[a]
        local infectious = σ * E[a]
        local recovery = γ * I[a]
        @inbounds begin
            du[1,a] = - infection
            du[2,a] = infection - infectious
            du[3,a] = infectious - recovery
            du[4,a] = infection
            du[5,a] = infectious
        end
    end
end;# Construct an ODE for the SEIR model



struct idd_params{T <: Real, T2 <: DataInterpolations.AbstractInterpolation}
    params_floats::Vector{T}
    β_function::T2
    N_regions::Int
end

NR = 1

params_test = idd_params(p, ConstantInterpolation(true_beta, knots), NR)

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem(sir_tvp_ode!, u0', tspan, params_test)
#? Note the choice of tstops and d_discontinuities to note the changepoints in β
#? Also note the choice of solver to resolve issues with the "stiffness" of the ODE system
sol_ode = solve(prob_ode,
            Tsit5(; thread = OrdinaryDiffEq.False()),
            maxiters = 1e6,
            abstol = 1e-8,
            reltol = 1e-5,
            # callback = cb,
            saveat = 1.0,
            tstops = knots[2:end-1],
            d_discontinuities = knots);

# Optionally plot the SEIR system
plot(stack(map(x -> x[3,:], sol_ode.u))',
    xlabel="Time",
    ylabel="Number",
    linewidth = 1)


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


# Evaluate mean number of hospitalisations (using proportion of 0.3)
conv_mat = construct_pmatrix(;)  
Y_mu = mapreduce(x -> 0.3 * conv_mat * x, hcat, eachrow(X))'

# Create function to construct Negative binomial with properties matching those in Birell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

# Draw sample of hospitalisations
Y = @. rand(NegativeBinomial3(Y_mu + 1e-3, 10));

# Plot mean hospitalisations over hospitalisations
bar(obstimes, Y', legend=true, alpha = 0.3)
plot!(obstimes, eachrow(Y_mu))

# Store ODE system parameters in a dictionary and write to a file
params = Dict(
    "seed" => seeds_list[seed_idx],
    "Window_size" => Window_size,
    "Data_update_window" => Data_update_window,
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
    obstimes = obstimes,
    I0_μ_prior = I0_μ_prior,
    β₀μ = β₀μ,
    β₀σ = β₀σ,
    βσ = βσ,
    ::Type{T} = Float64,
    ::Type{T2} = Float64,
    ::Type{T3} = Float64;
) where {T <: Real, T2 <: Real, T3 <: Real}

    # Set prior for initial infected
    log_I₀  ~ truncated(Normal(I0_μ_prior, 0.2); lower = log(1.0 / N), upper = 0.0)
    I = exp(log_I₀) * N
    
    I_list = zero(Vector{T2}(undef, NA))
    I_list[1] = I
    u0 = zero(Matrix{T3}(undef, 5, NA))
    u0[1,:] = NA_N - I_list
    u0[3,:] = I_list
    

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    β = Vector{T}(undef, K)
    log_β = Vector{T}(undef, K-2)
    p = [γ, σ, N]
    # log_β₀ ~ Normal(log(0.4), β₀σ)
    log_β₀ ~ Normal(β₀μ, β₀σ)
    # βσ ~ Gamma(0.1,100)
    β[1] = exp(log_β₀)
    for i in 2:K-1
        log_β[i-1] ~ Normal(0.0, βσ)
        β[i] = exp(log(β[i-1]) + log_β[i-1])
    end
    β[K] = β[K-1]

    if(I < 1)
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end

    if(any(β .> N) | any(isnan.(β)))
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end

    params_test = idd_params(p, ConstantInterpolation(β, knots), 1) 
    # Run model
    ## Remake with new initial conditions and parameter values
    tspan = (0, maximum(obstimes))
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
    #         # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
    #             @DynamicPPL.addlogprob! -Inf
    #             return
    #         # end
    #     else
    #         rethrow(e)
    #     end
    # end

    if any(sol.retcode != :Success)
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end
    
    ## Calculate new infections per day, X
    sol_I_tot = Array(sol(obstimes))[5,:,:]
    sol_X = rowadjdiff(sol_I_tot)
    # println(sol_X)
    if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[3,:,:] .< -1e-3))
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end
    check = minimum(sol_X)
    # println(check)
    # y_μ = mapreduce(x -> 0.3 * conv_mat * x, hcat, eachrow(sol_X))'
    y_μ = (conv_mat * (0.3 .* sol_X)') |> transpose
    # y_μ = reduce(hcat, map(x -> 0.3 * conv_mat * x, eachrow(sol_X)))'

    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    if (any(isnan.(y_μ)))
        # if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        # end
    end
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, 10))

    return (; sol, p, check)
end;

# Define the parameters for the model given the known window size (i.e. vectors that fit within the window)
knots_window = collect(0:Δ_βt:Window_size)
knots_window = knots_window[end] != Window_size ? vcat(knots_window, Window_size) : knots_window
K_window = length(knots_window)
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, Window_size)
obstimes_window = 1.0:1.0:Window_size

# Sample the parameters to construct a "correct order" list of parameters
ode_prior = sample(
    bayes_sir_tvp(
    K_window,
    γ,
    σ,
    N,
    NA,
    NA_N,
    NR,
    conv_mat_window,
    knots_window,
    obstimes_window,
    I0_μ_prior,
    β₀μ,
    β₀σ,
    βσ;
    ) | (y = Y[:,1:Window_size],), Prior(), 1, discard_initial = 0, thinning = 1);

model_window_unconditioned = bayes_sir_tvp(
    K_window,
    γ,
    σ,
    N,
    NA,
    NA_N,
    NR,
    conv_mat_window,
    knots_window,
    obstimes_window,
    I0_μ_prior,
    β₀μ,
    β₀σ,
    βσ;
    )

# name_map_correct_order = ode_prior.name_map.parameters

# Perform the chosen inference algorithm
t1_init = time_ns()
ode_nuts = sample(model_window_unconditioned| (y = Y[:,1:Window_size],), Turing.NUTS(1500, 0.65;), MCMCThreads(), 100, n_chains, discard_initial = discard_init, thinning = 10);
t2_init = time_ns()
runtime_init = convert(Int64, t2_init-t1_init)

logjoint(model_window_unconditioned | (y = Y[:,1:Window_size],) ,ode_nuts)

# Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)

function generate_confint_infec_init(chn, y_data, K, conv_mat, knots, obstimes; cri = 0.95)
    chnm_res = generated_quantities(
        bayes_sir_tvp(K;
            conv_mat = conv_mat,
            knots = knots,
            obstimes = obstimes
            )| (y = y_data,),
        chn) 


    infecs = stack(map(x -> Array(x.sol)[3,:,:], chnm_res[1,:]))
    lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 3)[:,:,1]
    medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 3)[:, :, 1]
    uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 3)[:, :, 1]
    return (; lowci_inf, medci_inf, uppci_inf)
end

# Create a function to take in the chains and evaluate the number of recovereds and summarise them (at a specific confidence level)
function generate_confint_recov_init(chn, y_data, K, conv_mat, knots, obstimes; cri = 0.95)
    chnm_res = generated_quantities(
        bayes_sir_tvp(K;
            conv_mat = conv_mat,
            knots = knots,
            obstimes = obstimes
            )| (y = y_data, ),
        chn) 

    infecs = stack(map(x -> Array(x.sol)[4,:,:], chnm_res[1,:]))
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
        beta_idx = [collect(2:K); K]
    
        betas[:,2:end] =exp.(cumsum(betas[:,2:end], dims = 2))
        beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
        betas_lci = [quantile(betas[:,i], (1 - ci) / 2) for i in beta_idx]
        betas_uci = [quantile(betas[:,i], 1 - ((1-ci) / 2)) for i in beta_idx]
        return (beta_μ, betas_lci, betas_uci)
end

beta_μ, betas_lci, betas_uci = get_beta_quantiles(ode_nuts, K_window)

betat_no_win = ConstantInterpolation(true_beta, knots)

plot(obstimes[1:Window_size],
    ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size]),
    ribbon = (ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size]) - ConstantInterpolation(betas_lci, knots_window)(obstimes[1:Window_size]), ConstantInterpolation(betas_uci, knots_window)(obstimes[1:Window_size]) - ConstantInterpolation(beta_μ, knots_window)(obstimes[1:Window_size])),
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
plot!(obstimes[1:Window_size],
    betat_no_win(obstimes[1:Window_size]),
    color=:red,
    label="True β",
    lw = 2)
plot!(size = (1200,800))

savefig(string(outdir,"nuts_betas_window_1_$seed_idx.png"))


# Plot the infecteds
confint = generate_confint_infec_init(ode_nuts, Y[:,1:Window_size], K_window, conv_mat_window, knots_window, obstimes_window; cri = 0.9)
plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf') , legend = false)
plot!(I_dat[:,1:Window_size]', linewidth = 2, color = :red)
plot!(size = (1200,800))

savefig(string(outdir,"infections_nuts_window_1_$seed_idx.png"))

# Plot the recovereds
confint = generate_confint_recov_init(ode_nuts, Y[:,1:Window_size], K_window, conv_mat_window, knots_window, obstimes_window; cri = 0.9)
plot(confint.medci_inf', ribbon = (confint.medci_inf' - confint.lowci_inf', confint.uppci_inf' - confint.medci_inf')  , legend = false)
plot!(R_dat[:,1:Window_size]', linewidth = 2, color = :red)
plot!(size = (1200,800))

savefig(string(outdir,"recoveries_nuts_window_1_$seed_idx.png"))

# Convert the samples to an array
ode_nuts_arr = Array(ode_nuts)

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
get_params_varnames = function(n_old_betas)
    params_return = Vector{Turing.VarName}()
    if(n_old_betas >= 0) append!(params_return,[@varname(log_I₀), @varname(βσ)]) end
    if(n_old_betas >= 1) append!(params_return, [@varname(log_β₀)]) end
    if(n_old_betas >= 2)
        append!(params_return, [@varname(log_β[beta_idx]) for beta_idx ∈ 1:(n_old_betas - 1)])
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

    window_param_names = get_params_varnames(Int(ceil(curr_t / Δ_βt)))
    fixed_param_names = get_params_varnames(n_old_betas)

    y_data_window = Y[:,1:curr_t]
    t1 = time_ns()
    list_chains[idx_time] = sample(
        bayes_sir_tvp(K_window;
                conv_mat = conv_mat_window,
                knots = knots_window,
                obstimes = obstimes_window,
        ) | (y = y_data_window,),
        PracticalFiltering.PracticalFilter(
            fixed_param_names,
            window_param_names,
            list_chains[idx_time - 1],
            NUTS(1000,0.65)
        ),
        MCMCThreads(),
        n_chains;
        discard_initial = discard_init
    )
    t2 = time_ns()
    algorithm_times[idx_time] = convert(Int64, t2 - t1)

    beta_win_μ, betas_win_lci, betas_win_uci = get_beta_quantiles(list_chains[idx_time], K_window)

    plot(obstimes_window,
    ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window),
    ribbon = (ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window) - ConstantInterpolation(betas_win_lci, knots_window)(obstimes_window), ConstantInterpolation(betas_win_uci, knots_window)(obstimes_window) - ConstantInterpolation(beta_win_μ, knots_window)(obstimes_window)),
        xlabel = "Time",
        ylabel = "β",
        label="Window $idx_time",
    )
    plot!(obstimes_window,
        betat_no_win(obstimes_window);
        color=:red,
        label="True β")
    plot!(size = (1200,800))

    savefig(string(outdir,"β_nuts_window","$idx_time","_$seed_idx","_95.png"))

    # # Construct plot of infecteds and then recovereds at different confidence intervals
    # infs = generate_confint_infec_init(
    #     list_chains[idx_time],
    #     Y[1:curr_t],
    #     K_window,
    #     conv_mat_window,
    #     knots_window,
    #     obstimes_window)
    
    # plot(infs.medci_inf, ribbon = (infs.medci_inf -infs.lowci_inf, infs.uppci_inf - infs.medci_inf) , legend = false)
    # plot!(I_dat[1:curr_t], linesize = 3)
    # plot!(size = (1200,800))
    # savefig(string(outdir,"infections_nuts_window_$idx_time","_$seed_idx","_95.png"))

    # recovers = generate_confint_recov_init(
    #     list_chains[idx_time],
    #     Y[1:curr_t],
    #     K_window,
    #     conv_mat_window,
    #     knots_window,
    #     obstimes_window)
    
    # plot(recovers.medci_inf, ribbon = (recovers.medci_inf -recovers.lowci_inf, recovers.uppci_inf - recovers.medci_inf) , legend = false)
    # plot!(R_dat[1:curr_t], linesize = 3)
    # plot!(size = (1200,800))
    # savefig(string(outdir,"recoveries_nuts_window_$idx_time","_$seed_idx","_95.png"))

    # infs = generate_confint_infec_init(
    #     list_chains[idx_time],
    #     Y[1:curr_t],
    #     K_window,
    #     conv_mat_window,
    #     knots_window,
    #     obstimes_window; cri = 0.9)
    
    # plot(infs.medci_inf, ribbon = (infs.medci_inf -infs.lowci_inf, infs.uppci_inf - infs.medci_inf) , legend = false)
    # plot!(I_dat[1:curr_t], linesize = 3)
    # plot!(size = (1200,800))
    # savefig(string(outdir,"infections_nuts_window_$idx_time","_$seed_idx","_90.png"))

    # recovers = generate_confint_recov_init(
    #     list_chains[idx_time],
    #     Y[1:curr_t],
    #     K_window,
    #     conv_mat_window,
    #     knots_window,
    #     obstimes_window; cri = 0.9)
    
    # plot(recovers.medci_inf, ribbon = (recovers.medci_inf -recovers.lowci_inf, recovers.uppci_inf - recovers.medci_inf) , legend = false)
    # plot!(R_dat[1:curr_t], linesize = 3)
    # plot!(size = (1200,800))
    # savefig(string(outdir,"recoveries_nuts_window_$idx_time","_$seed_idx","_90.png"))

    # chain_for_densities = Chains(Array(list_chains[idx_time]), list_chains[idx_time].name_map.parameters)

    # joint_dens_array = logjoint(bayes_sir_tvp(K_window;
    #         conv_mat = conv_mat_window,
    #         knots = knots_window,
    #         obstimes = obstimes_window) | (y = Y[1:curr_t],), chain_for_densities)

    # histogram(joint_dens_array; normalize = :pdf)
    # density!(joint_dens_array)
    # savefig(string(outdir,"logjoint__window_$idx_time","_$seed_idx",".png"))
    
    # lik_dens_array = DynamicPPL.loglikelihood(bayes_sir_tvp(K_window;
    #     conv_mat = conv_mat_window,
    #     knots = knots_window,
    #     obstimes = obstimes_window) | (y = Y[1:curr_t],), chain_for_densities)

    # histogram(lik_dens_array; normalize = :pdf)
    # density!(lik_dens_array)
    # savefig(string(outdir,"loglikelihood_window_$idx_time","_$seed_idx",".png"))
    
    # prior_dens_array = logprior(bayes_sir_tvp(K_window;
    #     conv_mat = conv_mat_window,
    #     knots = knots_window,
    #     obstimes = obstimes_window) | (y = Y[1:curr_t],), chain_for_densities)

    # histogram(prior_dens_array; normalize = :pdf)
    # density!(prior_dens_array)
    # savefig(string(outdir,"logprior_window_$idx_time","_$seed_idx",".png"))


end


knots_init = collect(0:Δ_βt:each_end_time[1])
knots_init = knots_init[end] != each_end_time[1] ? vcat(knots_init, each_end_time[1]) : knots_init
beta_μ, betas_lci, betas_uci = get_beta_quantiles(list_chains[1], length(knots_init))

# Sequentially create plots of beta estimates, overlapping previous windows
for my_idx in 1:length(each_end_time)
    plot(obstimes[1:Window_size],
        betat_no_win(beta_μ, obstimes[1:Window_size]),
        ribbon = (betat_no_win(beta_μ, obstimes[1:Window_size]) - betat_no_win(betas_lci, obstimes[1:Window_size]), betat_no_win(betas_uci, obstimes[1:Window_size]) - betat_no_win(beta_μ, obstimes[1:Window_size])), 
        xlabel = "Time",
        ylabel = "β",
        label="Window 1",
        title="\nEstimates of β",
        color=:blue,
        xlimits=(0, each_end_time[end]),
        lw = 2,
        titlefontsize=18,
        guidefontsize=18,
        tickfontsize=16,
        legendfontsize=12,
        fillalpha = 0.4,
        legendposition = :outerbottom,
        # legendtitleposition = :left,
        margin = 10mm,
        bottom_margin = 0mm,
        legend_column = min(my_idx + 1, 4))
    if(my_idx > 1)
        for idx_time in 2:my_idx
            knots_plot = collect(0:Δ_βt:each_end_time[idx_time])
            knots_plot = knots_plot[end] != each_end_time[idx_time] ? vcat(knots_plot, each_end_time[idx_time]) : knots_plot
            beta_win_μ, betas_win_lci, betas_win_uci = get_beta_quantiles(list_chains[idx_time], length(knots_plot))
            plot!(obstimes[1:each_end_time[idx_time]],
                betat_no_win(beta_win_μ, obstimes[1:each_end_time[idx_time]]),
                ribbon = (betat_no_win(beta_win_μ, obstimes[1:each_end_time[idx_time]]) - betat_no_win(betas_win_lci, obstimes[1:each_end_time[idx_time]]), betat_no_win(betas_win_uci, obstimes[1:each_end_time[idx_time]]) - betat_no_win(beta_win_μ, obstimes[1:each_end_time[idx_time]])),
                # xlabel = "Time",
                # ylabel = "β",
                label="Window $idx_time",
                lw=2
                )
        end
    end 
    plot!(obstimes,
        betat_no_win(true_beta, obstimes),
        color=:red,
        label="True β", lw = 2)
    plot!(size = (1200,800))
    plot!([60], seriestype="vline", color = :black, label = missing, linestyle = :dash, lw = 4)


    savefig(string(outdir,"recoveries_nuts_window_combined","$my_idx","_$seed_idx","_95.png"))
end

params = Dict(
    "algorithm_times" => algorithm_times
    # "algorithm_times_each_sample" => algorithm_times_each_sample
    )

open(string(outdir, "timings.toml"), "w") do io
        TOML.print(io, params)
end