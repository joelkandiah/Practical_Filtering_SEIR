# Imports
using SSMProblems
using LinearAlgebra
using Distributions
using OrdinaryDiffEq
using Tullio
using Turing

function seir_dynamics!(du, u, p_, t)
     @inbounds @views begin
        S = u[1,:,:]
        E1 = u[2,:,:]
        E2 = u[3,:,:]
        I1 = u[4,:,:]
        I2 = u[5,:,:]
    end
    (γ, σ) = p_.params_floats
    a0 = p_.a0
    βt = p_.beta

    # Calculate the force of infection
    @tullio grad = false threads = false  b[i,k] := (I1[j,k] + I2[j,k]) * log(1 - (a0[k] * βt[k] * p_.sus_M[i,j] *  p_.C[j,i])) |> (1 - exp((_)))

    infection = S .* b
    infectious_1 = 0.5 .* σ .* E1
    infectious_2  = 0.5 .* σ .* E2
    recovery_1 = 0.5 .* γ .* I1
    recovery_2 = 0.5 .* γ .* I2
    @inbounds begin
        du[1,:,:] .= .- infection
        du[2,:,:] .= infection .- infectious_1
        du[3,:,:] .= infectious_1 .- infectious_2
        du[4,:,:] .= infectious_2 .- recovery_1
        du[5,:,:] .= recovery_1 .- recovery_2
        du[6,:,:] .= recovery_2
        du[7,:,:] .= infection
    end
end

struct SEIR_params{T_floats <: Real, T_a0 <: Real, T_beta <: Real, T_M <: Real, T_C <: Real}
    params_floats::Vector{T_floats}
    a0::Vector{T_a0}
    beta::Vector{T_beta}
    sus_M::Matrix{T_M}
    C::Matrix{T_C}
end

struct MyDynamics{T<:Real} <: LatentDynamics{T,Vector{T}}
    z::Vector{T}
    NA::Int8
    NR::Int8
    σ²::T
    u0::Array{T,3}
    steplength::T
    #...
end

function SSMProblems.distribution(model::MyDynamics; kwargs...)
    initial_state = zeros(model.NR)
    u0 = model.u0
    return product_distribution(vcat(Dirac.(initial_state), Dirac.(u0)[:]))
end

function SSMProblems.distribution(
    model::MyDynamics{T}, step::Int, prev_state::Vector{T}; kwargs...
) where {T}
    x_prev_vec = prev_state[1:model.NR]
    u_prev = prev_state[model.NR+1:end] |> reshape(...)

    if step == 1
        x_dists = Dirac.(zeros(model.NR))
    else
        x_dists = Normal.(x_prev_vec, model.σ²)
    end
    # Define the ODE problem for the SEIR dynamics
    # Need to construct the parameter struct and remake the problem    
    # Need to rethink the ODE_solver as we need the historic betas? or do we?

    prob = ODEProblem(
        seir_dynamics!,
        u_prev,
        (0.0, model.steplength),
        [x_prev_vec],
    )
    l_dists = solve(
        prob,
        Tsit5(),
        saveat=[model.steplength]
                prob,
        ).u |> stack .|> Dirac

        return product_distribution(vcat(x_dists, l_dists[:]))
end

struct SEIRObservationProcess{T<:Real} <: ObservationProcess{T, Vector{T}}

function SSMProblems.distribution(
    model::SEIRObservationProcess{T}, step::Int, state::Vector{T}; kwargs...
) where {T}
    x_vec = state[1:model.NR]
    l_vec = state[model.NR+1:end] |> reshape(...)

    # Similarly solve l_dists
    prob = ODEProblem(
        seir_dynamics!,
        l_vec,
        (0.0, model.steplength),
        [x_vec],
    )

    l_sol = solve(
        prob,
        Tsit5(),
        saveat=[model.steplength],
    ).u |> stack

    dists_y = map(
        l -> Binomial(model.n[step], l_sol),
        l_sol
    )

    dists_Z = map(
        x -> NegativeBinomial(x_vec, model.ψ),
        x_vec
    )
    
    return product_distribution([dists_y, dists_Z])
end

@model function my_seir_model(
        ode_prob,
        K,
        σ,
        N_pop_R,
        NA = NA,
        NA_N = NA_N,
        NR = NR,
        conv_mat = conv_mat,
        knots = knots,
        C = C,
        sero_sample_sizes = sample_sizes,
        IFR_means:: Vector{Float64} = IFR_means,
        IFR_sds:: Vector{Float64} = IFR_sds,
        trans_unconstrained_IFR = inverse(IFR_bijector)
        ::Type{T_ψ} = Float64,
        ::Type{T_IFR} = Float64,
        ::Type{T_sus_M} = Float64,
        ::Type{T_Seir} = Float64
    ) where {
        T_ψ <: Real,
        T_IFR <: Real,
        T_sus_M <: Real,
        T_Seir <: Real
    }

    # Write the parameter priors here
    # Start with the overdispersion
    η ~ truncated(Gamma(1, 1/0.2), upper = 150)

    # Then define the initial exponential growth parameters
    ψ = Vector{T_ψ}(undef, (NR))
    ψ_EE  ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_LDN ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_MID ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_NEY ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_NW  ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_SE  ~ truncated(Gamma(31.36, 1/224); upper = 0.5)
    ψ_SW  ~ truncated(Gamma(31.36, 1/224); upper = 0.5)

    ψ[1] = ψ_EE
    ψ[2] = ψ_LDN
    ψ[3] = ψ_MID
    ψ[4] = ψ_NEY
    ψ[5] = ψ_NW
    ψ[6] = ψ_SE
    ψ[7] = ψ_SW

    # Then the test sens & spec
    sero_sens  ~ Beta(71.5,29.5)
    sero_spec  ~ Beta(777.5,9.5)

    # Then the duration of infectious period
    d_I  ~ Gamma(1.43,1/0.549)
    inv_γ = d_I + 2
    γ  = 1 / inv_γ
   
    # Make more generic (pass indices)
    sus_M  = Matrix{T_sus_M}(undef, NA, NA)
    m_1 ~ truncated(Gamma(4,1/4), upper = 5.0)
    m_2 ~ truncated(Gamma(4,1/4), upper = 5.0)

    # Write the priors for the IFR
    IFR_vec = Vector{T_IFR}(undef, NA)
    IFR_vec_init = Vector{T_IFR}(undef, NA - 1)
    for i in 1:(NA-1)
        IFR_vec_init[i] ~ Normal(IFR_means[i], IFR_sds[i])
    end

    IFR_vec[2:end] .= trans_unconstrained_IFR.(IFR_vec_init)
    IFR_vec[1] = IFR_vec[2]

    inv_σ = 1 / σ
    R0_num = @. 1 + (ψ * inv_σ * 0.5) 
    R0_denom = @. 1 - (1 / ((1 + (ψ * inv_γ * 0.5))^2))
    a0_num = @. ψ * inv_γ * (R0_num^2) / R0_denom
    
    a0_denom = map(k -> exp(0.0) * inv_γ * real(eigmax(diagm(NA_N[:,k]) * (sus_M .* C))), 1:NR)
    a0 = a0_num ./ a0_denom

    @tullio b0[i,k] := log(1 - (a0[k] * exp(0.0) * sus_M[i,j] *  C[j,i])) |> (1 - exp((_)))
    tot_I₀ = inv_γ * map(k -> sum(b0[:,k] .* NA_N[:,k]) / a0_num[k], 1:NR)
    χ = map(k -> normalize(abs.(eigen(diagm(k) * sus_M .* C, sortby=x -> abs(x)).vectors[:,end]), 1), eachcol(NA_N)) |> stack
    tot_I₀_ages = χ .* transpose(tot_I₀)
    u0 = zeros(T_u0, 7, NA, NR)
    α = exp.(ψ) .- 1
    ϵ = 1
    u0[4,:,:] = tot_I₀_ages .* inv.(transpose(1 .+ ((ϵ .* γ) ./ (α .+ γ))))
    u0[5,:,:] = tot_I₀_ages .- @view u0[4,:,:]
    u0[3,:,:] = transpose((α .+ γ) ./ σ) .* @view(u0[4,:,:])
    u0[2,:,:] = transpose((α .+ σ) ./ σ) .* @view(u0[3,:,:])
    u0[1,:,:] = NA_N .- tot_I₀_ages .- @view(u0[2,:,:]) .- @view u0[3,:,:]

    return StateSpaceModel(dynamics, obs_process)
end

# So far I have that we are passing the previous model states, but... this is an issue with the  generation of observations of deaths... need to think about this 
