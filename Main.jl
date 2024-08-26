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

# Loop over each beta test
# for seed_idx in eachindex(seeds_list)

# Set seed
Random.seed!(seeds_list[seed_idx])

# Set and Create locations to save the plots and Chains
outdir = string("Results/10bp 1e6/Plot attempt $seed_idx/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/")
tmpstore = string("Chains/10bp 1e6/$seed_idx/$Δ_βt","_beta/window_$Window_size/chains_$n_chains/")

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
β₀σ²= 0.15
β₀μ = 0.14
βσ² = 0.15
true_beta = repeat([NaN], Integer(ceil(tmax/ Δ_βt)) + 1)
true_beta[1] = exp(rand(Normal(log(β₀μ), β₀σ²)))
for i in 2:(length(true_beta) - 1)
    # true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,1.0)))
    true_beta[i] = exp(log(true_beta[i-1]) + rand(Normal(0.0,βσ²)))
end 
true_beta[length(true_beta)] = true_beta[length(true_beta)-1]
knots = collect(0.0:Δ_βt:tmax)
knots = knots[end] != tmax ? vcat(knots, tmax) : knots
K = length(knots)

# Construct function to evaluate the ODE model for the SEIR model
function betat_no_win(p_, t)
    beta = ConstantInterpolation(p_, knots)
    return beta(t)
end;

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

prob_ode = ODEProblem(sir_tvp_ode_no_win!, u0, tspan, [p..., true_beta...])
sol_ode = solve(prob_ode,
            AutoVern7(Rodas4()),
            maxiters = 1e6,
            abstol = 1e-8,
            reltol = 1e-5,
            # callback = cb,
            saveat = 1.0,
            tstops = knots[2:end-1],
            d_discontinuities = knots);

plot(stack(map(x -> x[1:4], sol_ode.u))',
    xlabel="Time",
    ylabel="Number",
    labels=["S" "E" "I" "R"])

I_tot = [0; Array(sol_ode(obstimes))[5,:]] # Cumulative cases

function adjdiff(ary)
    ary1 = @view ary[begin:end-1]
    ary2 = @view ary[begin+1:end]
    return ary2 .- ary1
end

X = adjdiff(I_tot)

# Define Gamma distribution by mean and standard deviation
function Gamma_mean_sd_dist(μ, σ)
    α = (μ .* μ) ./ (σ .* σ)
    θ = (σ .* σ) ./ μ
    return Gamma.(α, θ)
end

# Define helpful distributions
incubation_dist = Gamma_mean_sd_dist(4.0, 1.41)
symp_to_hosp = Gamma_mean_sd_dist(9.0, 8.0666667)

# Define approximate convolution of distributions
function approx_convolve_gamma(d1::Gamma, d2::Gamma)
    μ_new = (d1.α * d1.θ) + (d2.α * d2.θ)

    var1 = d1.α * d1.θ * d1.θ
    var2 = d2.α * d2.θ * d2.θ

    σ_new = sqrt(var1 + var2)

    return Gamma_mean_sd_dist(μ_new, σ_new)
end

inf_to_hosp = approx_convolve_gamma(incubation_dist,symp_to_hosp)
inf_to_hosp_array_cdf = cdf(inf_to_hosp,1:160)
inf_to_hosp_array_cdf[2:end] = adjdiff(inf_to_hosp_array_cdf)

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
conv_mat = construct_pmatrix(;) 
Y_mu = 0.3 * conv_mat * X

function NegativeBinomial3(μ, ϕ)
    p = 1 / (1 + ϕ)
    r = μ / ϕ
    return NegativeBinomial(r, p)
end
Y = rand.(NegativeBinomial3.(Y_mu .+ 1e-3, 10));

bar(obstimes, Y, legend=false)
plot!(obstimes, Y_mu, legend=false)

params = Dict(
    "tmax" => tmax,
    "N" => N,
    "I0" => I0,
    "γ" => γ,
    "σ" => σ,
    "β" => true_beta,
    "knots" => knots,
    "inital β mean" => β₀μ,
    "initial β sd" => β₀σ²,
    "β sd" => βσ²,
    "Gamma alpha" => inf_to_hosp.α,
    "Gamma theta" => inf_to_hosp.θ
)

open(string(outdir, "params.toml"), "w") do io
        TOML.print(io, params)
end

prob_tvp = ODEProblem(sir_tvp_ode_no_win!,
        u0,
        tspan,
        true_beta);
        
@model function bayes_sir_tvp_init(
    y,
    # z,
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
    # i₀  ~ Uniform(0, 0.0005)
    I = exp(log_I₀) * N
    u0 = [N-I, 0.0, I, 0.0, 0.0]

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    p = Vector{T}(undef, K+3)
    log_β = Vector{T}(undef, K-2)
    p[1:3] .= [γ, σ, N]
    log_β₀ ~ Normal(log(0.4), 0.2)
    p[4] = exp(log_β₀)
    for i in 5:K+2
        log_β[i-4] ~ Normal(0.0, 0.2)
        p[i] = exp(log(p[i-1]) + log_β[i-4])
    end
    p[K+3] = p[K+2]
    # println(p)
    
    # Run model
    ## Remake with new initial conditions and parameter values
    tspan = (0, maximum(obstimes))
    prob = ODEProblem(sir_tvp_ode!, u0, tspan, p)
    # prob = remake(prob_tvp,
    #     u0=u0,
    #     p=p,
    #     t = (0, maximum(obstimes)),
    #     d_discontinuities = knots)
    
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
    
    ## Calculate cases per day, X
    sol_I_tot = [0; Array(sol(obstimes))[5,:]]
    # sol_I = [I; Array(sol(obstimes))[2,:]]
    sol_X = (adjdiff(sol_I_tot))
    if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[3,:] .< -1e-3))
        if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            # print("hit")
            @DynamicPPL.addlogprob! -Inf
            return
        end
    end
    check = minimum(sol_X)
    y_μ = 0.3 * conv_mat * sol_X
    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, 10))
    # z ~ product_distribution(Normal.(sol_I, 10))
    # l = length(y)
    # for i in 1:l
    #     y[i] ~ Poisson(sol_X[i])
    # end
    return (; sol, p, check)
end;

@model function bayes_sir_tvp(
    y,
    β_prev,
    I₀,
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
    # log_I₀  ~ Normal(-9.0, 0.2)
    # i₀  ~ Uniform(0, 0.0005)
    # I = exp(log_I₀) * N
    u0 = [N-I₀, 0.0, I₀, 0.0, 0.0]

    # Set priors for betas
    ## Note how we clone the endpoint of βt
    p = Vector{T}(undef, K+3)
    log_β = Vector{T}(undef, K-length(β_prev)-1)
    p[1:3] .= [γ, σ, N]
    # log_β₀ ~ Normal(log(0.4), 0.2)
    prev_length=length(β_prev)
    p[4:prev_length+3] .= β_prev
    for i in prev_length+4:K+2
        log_β[i-prev_length-3] ~ Normal(0.0, 0.2)
        p[i] = exp(log(p[i-1]) + log_β[i-prev_length-3])
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
    
    ## Calculate cases per day, X
    sol_I_tot = [0; Array(sol(obstimes))[5,:]]
    # sol_I = [I; Array(sol(obstimes))[2,:]]
    sol_X = (adjdiff(sol_I_tot))
    if (any(sol_X .< -(1e-3)) | any(Array(sol(obstimes))[3,:] .< -1e-3))
        if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
            # print("hit")
            @DynamicPPL.addlogprob! -Inf
            return
        end
    end
    check = minimum(sol_X)
    y_μ = 0.3 * conv_mat * sol_X
    # Assume Poisson distributed counts
    ## Calculate number of timepoints
    y ~ product_distribution(NegativeBinomial3.(y_μ .+ 1e-3, 10))
    # z ~ product_distribution(Normal.(sol_I, 10))
    # l = length(y)
    # for i in 1:l
    #     y[i] ~ Poisson(sol_X[i])
    # end
    return (; sol, p, check)
end;


knots_window = collect(0:Δ_βt:Window_size)
knots_window = knots_window[end] != Window_size ? vcat(knots_window, Window_size) : knots_window
K_window = length(knots_window)
conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, Window_size)
obstimes_window = 1.0:1.0:Window_size

function betat(p_, t)
    beta = ConstantInterpolation(p_, knots_window)
    return beta(t)
end;

function sir_tvp_ode!(du, u, p_, t)
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


ode_nuts = sample(bayes_sir_tvp_init(Y[1:Window_size], K_window;
    conv_mat = conv_mat_window,
    knots = knots_window,
    obstimes = obstimes_window,
    ), Turing.NUTS(1, 0.65), MCMCThreads(), 1, 1, discard_initial = 0, thinning = 1)

name_map_correct_order = ode_nuts.name_map.parameters

ode_nuts = sample(bayes_sir_tvp_init(Y[1:Window_size], K_window;
    conv_mat = conv_mat_window,
    knots = knots_window,
    obstimes = obstimes_window,
    ), Turing.NUTS(1000, 0.65), MCMCThreads(), 1, n_chains, discard_initial = discard_init, thinning = 10)

h5open(string(tmpstore,"chn_1_$seed_idx.h5"), "w") do f
    write(f, ode_nuts)
end

ode_nuts = h5open(string(tmpstore,"chn_1_$seed_idx.h5"), "r") do f
    read(f, Chains)
end

reorder_params = map(y -> findfirst(x -> x == y, ode_nuts.name_map.parameters), name_map_correct_order)

ode_nuts = Chains(Array(ode_nuts)[:,reorder_params], name_map_correct_order)

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

I_dat = Array(sol_ode(obstimes))[3,:] # Cumulative cases

R_dat = Array(sol_ode(obstimes))[4,:] # Cumulative cases

plot(Chains(Array(ode_nuts)[2:2:end,:], name_map_correct_order))

plot(ode_nuts)
savefig(string(outdir,"chn_nuts.png"))

betas = Array(ode_nuts[name_map_correct_order[1:end]])
beta_idx = [collect(2:K_window); K_window]

# betas[:,2] = log.(inverse(Bijectors.Logit(0,1)).(betas[:,2]))]    
betas[:,2:end] =exp.(cumsum(betas[:,2:end], dims = 2))
beta_μ = [quantile(betas[:,i], 0.5) for i in beta_idx]
betas_lci = [quantile(betas[:,i], 0.025) for i in beta_idx]
betas_uci = [quantile(betas[:,i], 0.975) for i in beta_idx]

plot(obstimes[1:Window_size],
    betat(beta_μ, obstimes[1:Window_size]),
    xlabel = "Time",
    ylabel = "β",
    label="Using the NUTS algorithm",
    title="Estimates of β",
    color=:blue)
plot!(obstimes[1:Window_size],
    betat(betas_lci, obstimes[1:Window_size]),
    alpha = 0.3,
    fillrange = betat(betas_uci, obstimes[1:Window_size]),
    fillalpha = 0.3,
    color=:blue,
    label="95% credible intervals")
plot!(obstimes[1:Window_size],
    betat(true_beta, obstimes[1:Window_size]),
    color=:red,
    label="True β")

savefig(string(outdir,"nuts_betas_window_1_$seed_idx.png"))


confint = generate_confint_infec_init(ode_nuts,Y[1:Window_size],K_window, conv_mat_window, knots_window, obstimes_window)
plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf) , legend = false)
plot!(I_dat[1:Window_size], linesize = 3)

savefig(string(outdir,"infections_nuts_window_1_$seed_idx.png"))

confint = generate_confint_recov_init(ode_nuts,Y[1:Window_size],K_window, conv_mat_window, knots_window, obstimes_window)

plot(confint.medci_inf, ribbon = (confint.medci_inf - confint.lowci_inf, confint.uppci_inf - confint.medci_inf)  , legend = false)
plot!(R_dat[1:Window_size], linesize = 3)

savefig(string(outdir,"recoveries_nuts_window_1_$seed_idx.png"))

ode_nuts_arr = Array(ode_nuts)

each_end_time = collect(Window_size:Data_update_window:tmax)
each_end_time = each_end_time[end] ≈ tmax ? each_end_time : vcat(each_end_time, tmax)
each_end_time = Integer.(each_end_time)

init_I₀ = exp.(ode_nuts_arr[:,1]) .* N
list_prev_β = exp.(cumsum(ode_nuts_arr[:, 2:end], dims = 2))

for idx_time in 2:length(each_end_time)
    # Initialise containers for the results (no data race one value per chain to be written)
    chn_list = Vector{Chains}(undef, n_chains)
    list_β = Vector{Vector}(undef, n_chains)

    curr_t = each_end_time[idx_time]
    n_old_betas = Int(floor((curr_t - Window_size) / Δ_βt))
    window_betas = Int(ceil(curr_t / Δ_βt) - n_old_betas)

    log_init_β_params = mapreduce(x -> adjdiff(log.(x)), hcat, eachrow(list_prev_β))'
    log_init_β_params = hcat(log.(list_prev_β)[:,1], log_init_β_params)


    n_prev_betas = length(list_prev_β[1,:])

    global knots_window = collect(0:Δ_βt:curr_t)
    global knots_window = knots_window[end] != curr_t ? vcat(knots_window, curr_t) : knots_window
    global K_window = length(knots_window)
    global conv_mat_window = construct_pmatrix(inf_to_hosp_array_cdf, curr_t)
    global obstimes_window = 1.0:1.0:curr_t

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
    
    if(n_old_betas == 0)
        Threads.@threads for i in 1:n_chains
            I₀_init = init_I₀[i]
            use_params = rand(Normal(0, 0.2), window_betas)
            use_params[1:(n_prev_betas)] .= log_init_β_params[i,1:end]
            chn_list[i] = sample(DynamicPPL.fix(bayes_sir_tvp_init(Y[1:curr_t], K_window;
                conv_mat = conv_mat_window,
                knots = knots_window,
                obstimes = obstimes_window), log_I₀ = log(I₀_init / N)),
                Turing.NUTS(),
                MCMCSerial(), 1, 1;
                discard_initial = discard_init,
                thinning = 1,
                init_parameters = (use_params,))
            new_betas_i = exp.(cumsum(Array(chn_list[i])[1,:]))
            list_β[i] =  new_betas_i
        end

    else
        Threads.@threads for i in 1:n_chains
            I₀_init = init_I₀[i]
            β_hist = list_prev_β[i,1:n_old_betas]
            use_params = rand(Normal(0, 0.2), window_betas)
            if(n_prev_betas > n_old_betas)
                use_params[1:(n_prev_betas - n_old_betas)] .= log_init_β_params[i,(1+n_old_betas):end]
            end
            chn_list[i] = sample(bayes_sir_tvp(Y[1:curr_t], β_hist, I₀_init, K_window;
                conv_mat = conv_mat_window,
                knots = knots_window,
                obstimes = obstimes_window),
                Turing.NUTS(),
                MCMCSerial(), 1, 1;
                discard_initial = discard_init,
                thinning = 1,
                init_parameters = (use_params,))
            new_betas_i = exp.(log(β_hist[end]) .+ cumsum(Array(chn_list[i])[1,:]))
            list_β[i] = vcat(β_hist, new_betas_i)
        end

    end

    list_β_conv = mapreduce(x -> x, hcat, list_β)'

    jldsave(string(tmpstore,"chn_$idx_time","_$seed_idx.jld2"); list_β_conv)

    global list_prev_β = list_β_conv

    generate_quants_params = hcat(ode_nuts_arr[:,1], list_β_conv)
    generate_quants_params[:,3:end] .= log.(generate_quants_params[:,3:end]) .- log.(generate_quants_params[:,2:end - 1])
    generate_quants_params[:,2] .= log.(generate_quants_params[:,2])


    Chain_symb_names =  [Symbol("log_β[$z]") for z in 1:K_window-2]
    Chain_symb_names = vcat(:log_I₀,:log_β₀, Chain_symb_names)

    infs = generate_confint_infec_init(
        Chains(generate_quants_params, Chain_symb_names),
        Y[1:curr_t],
        K_window,
        conv_mat_window,
        knots_window,
        obstimes_window)
    
    plot(infs.medci_inf, ribbon = (infs.medci_inf -infs.lowci_inf, infs.uppci_inf - infs.medci_inf) , legend = false)
    plot!(I_dat[1:curr_t], linesize = 3)
    savefig(string(outdir,"infections_nuts_window_$idx_time","_$seed_idx","_95.png"))

    recovers = generate_confint_recov_init(
        Chains(generate_quants_params, Chain_symb_names),
        Y[1:curr_t],
        K_window,
        conv_mat_window,
        knots_window,
        obstimes_window)
    
    plot(recovers.medci_inf, ribbon = (recovers.medci_inf -recovers.lowci_inf, recovers.uppci_inf - recovers.medci_inf) , legend = false)
    plot!(R_dat[1:curr_t], linesize = 3)
    savefig(string(outdir,"recoveries_nuts_window_$idx_time","_$seed_idx","_95.png"))

    infs = generate_confint_infec_init(
        Chains(generate_quants_params, Chain_symb_names),
        Y[1:curr_t],
        K_window,
        conv_mat_window,
        knots_window,
        obstimes_window; cri = 0.9)
    
    plot(infs.medci_inf, ribbon = (infs.medci_inf -infs.lowci_inf, infs.uppci_inf - infs.medci_inf) , legend = false)
    plot!(I_dat[1:curr_t], linesize = 3)
    savefig(string(outdir,"infections_nuts_window_$idx_time","_$seed_idx","_90.png"))

    recovers = generate_confint_recov_init(
        Chains(generate_quants_params, Chain_symb_names),
        Y[1:curr_t],
        K_window,
        conv_mat_window,
        knots_window,
        obstimes_window; cri = 0.9)
    
    plot(recovers.medci_inf, ribbon = (recovers.medci_inf -recovers.lowci_inf, recovers.uppci_inf - recovers.medci_inf) , legend = false)
    plot!(R_dat[1:curr_t], linesize = 3)
    savefig(string(outdir,"recoveries_nuts_window_$idx_time","_$seed_idx","_90.png"))

end

array_each_window_β = Array{Array}(undef, length(each_end_time))
array_each_window_β[1] = exp.(cumsum(ode_nuts_arr[:, 2:end], dims = 2))

for idx_time in 2:length(each_end_time)
    array_each_window_β[idx_time] = load(string(tmpstore,"chn_$idx_time","_$seed_idx.jld2"))["list_β_conv"]
end


for my_idx in 1:length(each_end_time)
    plot(obstimes[1:Window_size],
        betat(beta_μ, obstimes[1:Window_size]),
        ribbon = (betat(beta_μ, obstimes[1:Window_size]) - betat(betas_lci, obstimes[1:Window_size]), betat(betas_uci, obstimes[1:Window_size]) - betat(beta_μ, obstimes[1:Window_size])), 
        xlabel = "Time",
        ylabel = "β",
        label="Window 1",
        title="Estimates of β")
    if(my_idx > 1)
        for idx_time in 2:my_idx
            beta_idxs = vcat(collect(1:length(0:Δ_βt:each_end_time[idx_time])-1), length(0:Δ_βt:each_end_time[idx_time])-1)
            beta_win_μ = [quantile(array_each_window_β[idx_time][:,i], 0.5) for i in beta_idxs]
            betas_win_lci = [quantile(array_each_window_β[idx_time][:,i], 0.025) for i in beta_idxs]
            betas_win_uci = [quantile(array_each_window_β[idx_time][:,i], 0.975) for i in beta_idxs]
            plot!(obstimes[1:each_end_time[idx_time]],
                betat(beta_win_μ, obstimes[1:each_end_time[idx_time]]),
                ribbon = (betat(beta_win_μ, obstimes[1:each_end_time[idx_time]]) - betat(betas_win_lci, obstimes[1:each_end_time[idx_time]]), betat(betas_win_uci, obstimes[1:each_end_time[idx_time]]) - betat(beta_win_μ, obstimes[1:each_end_time[idx_time]])),
                xlabel = "Time",
                ylabel = "β",
                label="Window $idx_time",
                )
        end
    end 
    plot!(obstimes,
        betat(true_beta, obstimes),
        color=:red,
        label="True β")

    savefig(string(outdir,"recoveries_nuts_window_combined","$my_idx","_$seed_idx","_95.png"))
end

plot(obstimes[1:Window_size],
    betat(beta_μ, obstimes[1:Window_size]),
    ribbon = (betat(beta_μ, obstimes[1:Window_size]) - betat(betas_lci, obstimes[1:Window_size]), betat(betas_uci, obstimes[1:Window_size]) - betat(beta_μ, obstimes[1:Window_size])), 
    xlabel = "Time",
    ylabel = "β",
    label="Window 1",
    title="Estimates of β")
for idx_time in 2:length(each_end_time)
    beta_idxs = vcat(collect(1:length(0:Δ_βt:each_end_time[idx_time])-1), length(0:Δ_βt:each_end_time[idx_time])-1)
    beta_win_μ = [quantile(array_each_window_β[idx_time][:,i], 0.5) for i in beta_idxs]
    betas_win_lci = [quantile(array_each_window_β[idx_time][:,i], 0.05) for i in beta_idxs]
    betas_win_uci = [quantile(array_each_window_β[idx_time][:,i], 0.95) for i in beta_idxs]
    plot!(obstimes[1:each_end_time[idx_time]],
        betat(beta_win_μ, obstimes[1:each_end_time[idx_time]]),
        ribbon = (betat(beta_win_μ, obstimes[1:each_end_time[idx_time]]) - betat(betas_win_lci, obstimes[1:each_end_time[idx_time]]), betat(betas_win_uci, obstimes[1:each_end_time[idx_time]]) - betat(beta_win_μ, obstimes[1:each_end_time[idx_time]])),
        xlabel = "Time",
        ylabel = "β",
        label="Window $idx_time",
        )
end
plot!(obstimes,
    betat(true_beta, obstimes),
    color=:red,
    label="True β")
savefig(string(outdir,"recoveries_nuts_window_combined","_$seed_idx","_90.png"))



# end
