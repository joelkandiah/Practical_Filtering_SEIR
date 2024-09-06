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
using HDF5
using MCMCChains
using MCMCChainsStorage
using JLD2
using AdvancedMH
using DynamicHMC
using ProgressMeter
using BenchmarkTools


@assert length(ARGS) == 7

# Read parameters from command line for the varied submission scripts
seed_idx = parse(Int, ARGS[1])
Δ_βt = parse(Int, ARGS[2])
Window_size = parse(Int, ARGS[3])
Data_update_window = parse(Int, ARGS[4])
n_samples = parse(Int, ARGS[5])
discard_init = parse(Int, ARGS[6])
tmax = parse(Float64, ARGS[7])

# List the seeds to generate a set of data scenarios (for reproducibility)
seeds_list = [1234, 1357, 2358, 3581]

# Set seed
Random.seed!(seeds_list[seed_idx])

n_threads = Threads.nthreads()

# Set and Create locations to save the plots and Chains
outdir = string("Results/10bp 1e6/$Δ_βt","_beta/window_$Window_size/chains_$n_samples/n_threads_$n_threads/original_inference/Plot attempt $seed_idx/")
tmpstore = string("Chains/10bp 1e6/$Δ_βt","_beta/window_$Window_size/chains_$n_samples/n_threads_$n_threads/original_inference/Plot attempt $seed_idx/")

if !isdir(outdir)
    mkpath(outdir)
end
if !isdir(tmpstore)
    mkpath(tmpstore)
end

# Initialise the model parameters (fixed)
tspan = (0.0, tmax)
obstimes = 1.0:1.0:tmax
N = 1_000_000.0
I0 = 100.0
u0 = [N-I0, 0.0, I0, 0.0, 0.0] # S,E,I,R,I_tot
inv_γ = 10  
inv_σ = 3
γ = 1/ inv_γ
σ = 1/ inv_σ
p = [γ, σ, N];

# Set parameters for inference and draw betas from prior
β₀σ= 0.15
β₀μ = 0.14
βσ = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax/ Δ_βt)) + 1)
true_beta[1] = exp(rand(Normal(log(β₀μ), β₀σ)))
for i in 2:(length(true_beta) - 1)
    # true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,1.0)))
    true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,βσ)))
end 
true_beta[length(true_beta)] = true_beta[length(true_beta)-1]
knots = collect(0.0:Δ_βt:tmax)
knots = knots[end] != tmax ? vcat(knots, tmax) : knots
K = length(knots)

# Construct function to evaluate the ODE model for the SEIR model
# Create piecewise constant beta function
function betat_no_win(p_, t)
    beta = ConstantInterpolation(p_, knots)
    return beta(t)
end;

# Construct an ODE for the SEIR model
function sir_tvp_ode_no_win!(du, u, p_, t)
    (S, E, I, R, I_tot) = u
    (γ, σ, N) = p_[1:3]
    βt = betat_no_win(p_[4:end], t)
    infection = βt*S*I/N
    infectious = σ*E
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - infectious
        du[3] = infectious - recovery
        du[4] = infection
        du[5] = infectious
    end
end;

# Initialise the specific values for the ODE system and solve
prob_ode = ODEProblem(sir_tvp_ode_no_win!, u0, tspan, vcat(p, true_beta))
#? Note the choice of tstops and d_discontinuities to note the changepoints in β
#? Also note the choice of solver to resolve issues with the "stiffness" of the ODE system
sol_ode = solve(prob_ode,
            AutoVern7(Rodas4()),
            maxiters = 1e6,
            abstol = 1e-8,
            reltol = 1e-5,
            # callback = cb,
            saveat = 1.0,
            tstops = knots[2:end-1],
            d_discontinuities = knots);

# Optionally plot the SEIR system
plot(stack(map(x -> x[1:4], sol_ode.u))',
    xlabel="Time",
    ylabel="Number",
    labels=["S" "E" "I" "R"], linewidth = 1)


# Find the cumulative number of cases
I_tot = [0; Array(sol_ode(obstimes))[5,:]] # Cumulative cases

# Define utility function for the difference between consecutive arguments in a list f: Array{N} x Array{N} -> Array{N-1}
function adjdiff(ary)
    ary1 = @view ary[begin:end-1]
    ary2 = @view ary[begin+1:end]
    return ary2 .- ary1
end

# Number of new infections
X = adjdiff(I_tot)

# Define Gamma distribution by mean and standard deviation
function Gamma_mean_sd_dist(μ, σ)
    α = (μ .* μ) ./ (σ .* σ)
    θ = (σ .* σ) ./ μ
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
inf_to_hosp_array_cdf = cdf(inf_to_hosp,1:160)
inf_to_hosp_array_cdf[2:end] = adjdiff(inf_to_hosp_array_cdf)

# Create function to create a matrix to calculate the discrete convolution (multiply convolution matrix by new infections vector to get mean of number of (eligible) hospitalisations per day)
function construct_pmatrix(
    v = inf_to_hosp_array_cdf,
    l = Integer(tmax))
    rev_v = @view v[end:-1:begin]
    ret_mat = zeros(l, l)
    for i = 1:(l)
        ret_mat[i, max(begin, i + 1 - length(v)):min(i, end)] .= rev_v[max(begin, length(v)-i+1):end]        
    end
    return ret_mat
end

# Evaluate mean number of hospitalisations (using proportion of 0.3)
conv_mat = construct_pmatrix(;) 
Y_mu = 0.3 * conv_mat * X

# Create function to construct Negative binomial with properties matching those in Birell et. al (2021)
function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end

# Draw sample of hospitalisations
Y = rand.(NegativeBinomial3.(Y_mu .+ 1e-3, 10));

# Plot mean hospitalisations over hospitalisations
bar(obstimes, Y, legend=false)
plot!(obstimes, Y_mu, legend=false)

# Store ODE system parameters in a dictionary and write to a file
params = Dict(
    "seed" => seeds_list[seed_idx],
    "Window_size" => Window_size,
    "Data_update_window" => Data_update_window,
    "n_chains" => Threads.nthreads(),
    "samples" => n_samples,
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

# Reinitialise the ODE problem
prob_tvp = ODEProblem(sir_tvp_ode_no_win!,
        u0,
        tspan,
        true_beta);
        
# Define the model taking in the data and the times the beta values changepoints
# Add named args for fixed ODE parameters and for the convolution matrix, the times the beta values change and the specific times for evaluating the ODE
@model function bayes_sir_tvp_init(
    y,
    K,
    ::Type{T}=Float64;
    γ = γ,
    σ = σ,
    N = N,
    conv_mat = conv_mat,
    knots = knots,
    obstimes = obstimes,
) where {T <: Real}
    # Set prior for initial infected
    log_I₀  ~ Normal(-9.0, 0.2)
    I = exp(log_I₀) * N
    u0 = [N-I, 0.0, I, 0.0, 0.0]

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    p = Vector{T}(undef, K+3)
    log_β = Vector{T}(undef, K-2)
    p[1:3] .= [γ, σ, N]
    log_β₀ ~ Normal(log(0.4), β₀σ)
    p[4] = exp(log_β₀)
    for i in 5:K+2
        log_β[i-4] ~ Normal(0.0, βσ)
        p[i] = exp(log(p[i-1]) + log_β[i-4])
    end
    p[K+3] = p[K+2]
    
    # Run model
    ## Remake with new initial conditions and parameter values
    tspan = (0, maximum(obstimes))
    prob = ODEProblem(sir_tvp_ode!, u0, tspan, p)
    
    ## Solve
    sol = solve(prob,
            AutoVern7(Rodas4P()),
            saveat = obstimes,
            maxiters = 1e6,
            d_discontinuities = knots[2:end-1],
            tstops = knots[2:end-1],
            abstol = 1e-8,
            reltol = 1e-5)

    if any(sol.retcode != :Success)
        if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        end
    end
    
    ## Calculate new infections per day, X
    sol_I_tot = [0; Array(sol(obstimes))[5,:]]
    sol_X = (adjdiff(sol_I_tot))
    if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[3,:] .< -1e-3))
        if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            @DynamicPPL.addlogprob! -Inf
            return
        end
    end
    check = minimum(sol_X)
    y_μ = 0.3 * conv_mat * sol_X

    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, 10))
    return (; sol, p, check)
end;

# Create an array to store the window ends to know when to run the model
each_end_time = collect(Window_size:Data_update_window:tmax)
each_end_time = each_end_time[end] ≈ tmax ? each_end_time : vcat(each_end_time, tmax)
each_end_time = Integer.(each_end_time)

algorithm_times = Vector{Float64}(undef, length(each_end_time))

for idx_time in eachindex(each_end_time)

    # Define the parameters for the model given the known window size (i.e. vectors that fit within the window)
    knots_window = collect(0:Δ_βt:each_end_time[idx_time])
    knots_window = knots_window[end] != each_end_time[idx_time] ? vcat(knots_window, each_end_time[idx_time]) : knots_window
    K_window = length(knots_window)
    conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, each_end_time[idx_time])
    obstimes_window = 1.0:1.0:each_end_time[idx_time]
    # Redefine the beta function and sir model given the window size
    #! Better to find some way to generalise this (maybe move into model code?)
    global function betat(p_, t)
        beta = ConstantInterpolation(p_, knots_window)
        return beta(t)
    end;

    global function sir_tvp_ode!(du, u, p_, t)
        (S, E, I, R, I_tot) = u
        (γ, σ, N) = p_[1:3]
        βt = betat(p_[4:end], t)
        infection = βt*S*I/N
        infectious = σ*E
        recovery = γ*I
        @inbounds begin
            du[1] = -infection
            du[2] = infection - infectious
            du[3] = infectious - recovery
            du[4] = infection
            du[5] = infectious
        end
    end;

    # Sample the parameters to construct a "correct order" list of parameters
    ode_nuts = sample(bayes_sir_tvp_init(Y[1:each_end_time[idx_time]], K_window;
        conv_mat = conv_mat_window,
        knots = knots_window,
        obstimes = obstimes_window,
        ), Turing.NUTS(1, 0.65), MCMCThreads(), 1, 1, discard_initial = 0, thinning = 1);

    name_map_correct_order = ode_nuts.name_map.parameters

    # Perform the chosen inference algorithm
    t1_init = time_ns()
    ode_nuts = sample(bayes_sir_tvp_init(Y[1:each_end_time[idx_time]], K_window;
        conv_mat = conv_mat_window,
        knots = knots_window,
        obstimes = obstimes_window,
        ), Turing.NUTS(1000, 0.65), MCMCThreads(), n_samples, Threads.nthreads(), discard_initial = discard_init, thinning = 10);
    t2_init = time_ns()
    algorithm_times[idx_time] = convert(Int64, t2_init-t1_init)

    # Write the results of the chain to a file
    h5open(string(tmpstore,"chn_$seed_idx","_window_$idx_time.h5"), "w") do f
        write(f, ode_nuts)
    end

    ode_nuts = h5open(string(tmpstore,"chn_$seed_idx","_window_$idx_time.h5"), "r") do f
        read(f, Chains)
    end

    # Reorder the parameters read from the file to match with if it had been sampled at runtime
    reorder_params = map(y -> findfirst(x -> x == y, ode_nuts.name_map.parameters), name_map_correct_order)

    ode_nuts

    chain_slices = Vector{Array}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        chain_slices[i] = Array(ode_nuts[:,:,i])[:, reorder_params]
    end

    ode_nuts = Chains(stack(chain_slices, dims = 3), name_map_correct_order)

    # Create a function to take in the chains and evaluate the number of infections and summarise them (at a specific confidence level)
    function generate_confint_infec_init(chn, Y, K, conv_mat, knots, obstimes; cri = 0.95)
        chnm_res = generated_quantities(
            bayes_sir_tvp_init(Y, K;
                conv_mat = conv_mat,
                knots = knots,
                obstimes = obstimes),
            chn)

        infecs = cat(map(x ->Array(x.sol)[3,:], chnm_res)..., dims = 2)
        lowci_inf = mapslices(x -> quantile(x, (1-cri) / 2), infecs, dims = 2)[:,1]
        medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 2)[:, 1]
        uppci_inf = mapslices(x -> quantile(x, cri + (1-cri)/ 2), infecs, dims = 2)[:,1]
        return (; lowci_inf, medci_inf, uppci_inf)
    end

    # Create a function to take in the chains and evaluate the number of recovereds and summarise them (at a specific confidence level)
    function generate_confint_recov_init(chn, Y, K, conv_mat, knots, obstimes; cri = 0.95)
        chnm_res = generated_quantities(bayes_sir_tvp_init(Y, K;
                conv_mat = conv_mat,
                knots = knots,
                obstimes = obstimes),
            chn)

        infecs = cat(map(x ->Array(x.sol)[4,:], chnm_res)..., dims = 2)
        lowci_inf = mapslices(x -> quantile(x,(1-cri) / 2), infecs, dims = 2)[:,1]
        medci_inf = mapslices(x -> quantile(x, 0.5), infecs, dims = 2)[:, 1]
        uppci_inf = mapslices(x -> quantile(x, cri + (1-cri) / 2), infecs, dims = 2)[:,1]
        return (; lowci_inf, medci_inf, uppci_inf)
    end


    I_dat = Array(sol_ode(obstimes))[3,:] # Population of infecteds at times
    R_dat = Array(sol_ode(obstimes))[4,:] # Population of recovereds at times

    # Plot the chain (to understand "mixing" - not correct do not use this)

    plot(ode_nuts)
    savefig(string(outdir,"chn_nuts_window_$idx_time.png"))

    # Get the beta values and calculate the estimated confidence interval and median
    betas = Array(ode_nuts[name_map_correct_order[1:end]])
    beta_idx = [collect(2:K_window); K_window]

    betas[:,2:end] =exp.(cumsum(betas[:,2:end], dims = 2))
    beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
    betas_lci = [quantile(betas[:,i], 0.025) for i in beta_idx]
    betas_uci = [quantile(betas[:,i], 0.975) for i in beta_idx]


    # Plot the betas (and uncertainty)
    plot(obstimes[1:each_end_time[idx_time]],
        betat(beta_μ, obstimes[1:each_end_time[idx_time]]),
        xlabel = "Time",
        ylabel = "β",
        label="Using the NUTS algorithm",
        title="Estimates of β",
        color=:blue)
    plot!(obstimes[1:each_end_time[idx_time]],
        betat(betas_lci, obstimes[1:each_end_time[idx_time]]),
        alpha = 0.3,
        fillrange = betat(betas_uci, obstimes[1:each_end_time[idx_time]]),
        fillalpha = 0.3,
        color=:blue,
        label="95% credible intervals")
    plot!(obstimes[1:each_end_time[idx_time]],
        betat(true_beta, obstimes[1:each_end_time[idx_time]]),
        color=:red,
        label="True β")
    plot!(size = (1200,800))

    savefig(string(outdir,"nuts_betas_window_$idx_time","_$seed_idx.png"))

    # Plot the infecteds
    confint = generate_confint_infec_init(ode_nuts,Y[1:each_end_time[idx_time]],K_window, conv_mat_window, knots_window, obstimes_window)
    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf) , legend = false)
    plot!(I_dat[1:each_end_time[idx_time]], linesize = 3)
    plot!(size = (1200,800))

    savefig(string(outdir,"infections_nuts_window_$idx_time","_$seed_idx.png"))

    # Plot the recovereds
    confint = generate_confint_recov_init(ode_nuts,Y[1:each_end_time[idx_time]],K_window, conv_mat_window, knots_window, obstimes_window)
    plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf)  , legend = false)
    plot!(R_dat[1:each_end_time[idx_time]], linesize = 3)
    plot!(size = (1200,800))

    savefig(string(outdir,"recoveries_nuts_window_$idx_time","_$seed_idx.png"))

end


params = Dict(
    "algorithm_times" => algorithm_times,
    )

open(string(outdir, "timings.toml"), "w") do io
        TOML.print(io, params)
end