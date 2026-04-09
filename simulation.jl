# -----------------------------------------------------------------------------
# Author: Johannes Konrad
# Affiliation: Institute for Digital Communications, Friedrich-Alexander-Universität Erlangen-Nürnberg
# Email: johannes.konrad@fau.de
#
# This code is associated with the manuscript:
# "Spatiotemporal Modeling of Chemotactic cAMP Signaling by Secretion and Degradation"
#
# The code implements the methods and algorithms described in the manuscript.
# Please refer to the paper for a detailed explanation of the methodology, 
# experimental design, and results.
# -----------------------------------------------------------------------------
"""
    Dictyostelium cAMP Signaling Simulation

Stochastic agent-based simulation of cAMP signaling dynamics in Dictyostelium discoideum.
Uses Gillespie/SSA for intracellular reactions coupled with diffusion on a 2D grid.

# Model Components
- Intracellular signaling network (Catalyst.jl reaction network)
- Extracellular cAMP and PDE diffusion
- Agent-based cell representation (Agents.jl)

# Usage
    julia dictyostelium_simulation.jl <output_suffix> <n_agents> [k15_ind] [K]

# Arguments
- `output_suffix`: Suffix for output directory (e.g., "run1" → "output_csv_run1")
- `n_agents`: Number of cells in the simulation
- `k15_ind`: PDE induction rate parameter (default: 1.08)
- `K`: Hill function parameter (default: 4000)

Author: Peter (FAU Erlangen-Nürnberg, LHFT)
"""
module DictyosteliumSimulation

using Agents
using CairoMakie
using Random
using Catalyst
using JumpProcesses
using Distributions
using ProgressMeter
using Printf
using StaticArrays
using CSV
using DataFrames

export run_simulation

# =============================================================================
# Configuration Constants
# =============================================================================

# Grid dimensions
const WIDTH = 500
const HEIGHT = 200
const SPECIES_COUNT = 2

# Species indices
const cAMP_IDX = 1
const PDEe_IDX = 2

# Spatial discretization
const DX = 3.0  # µm per grid unit

# Diffusion coefficients [µm²/s]
const DIFFUSION_RATES = [350.0, 40.0]  # [cAMP, PDE]

# Reaction rates
const DECAY_RATE = 1e5      # cAMP degradation by PDE
const DECAY_PDE = 0.01 / 60 # PDE decay rate

# Simulation parameters
const DEFAULT_STEPS = 100_000
const DEFAULT_DT = 0.002

# Agent placement
const DESIRED_DISTANCE_UM = 40.0
const AGENT_DIST = DESIRED_DISTANCE_UM / DX
const MIN_BORDER_DIST = 50
const STEPSIZE = 10

# Physical constants
const NA = 6.023e23  # Avogadro's number
const V = 3.672e-14  # Cell volume [L]
const MU = 1e-6      # µM conversion factor

# Species mapping for grid access
const SPECIES_MAP = Dict{Symbol,Int}(:cAMPe => cAMP_IDX, :PDEe => PDEe_IDX)

# Nominal initial conditions [molecules or concentration]
const NOMINAL_U0 = Dict(
    :ACA => 0.0,
    :PKA => 0.0,
    :ERK2 => 0.0,
    :RegA => 0.0,
    :cAMPi => 0.0,
    :cAMPe => 700.0,
    :CAR1 => 0.0,
    :PDEe => 0.0,
    :cAMPtracked => 0.0
)

# =============================================================================
# Reaction Network Definition
# =============================================================================

"""
Intracellular signaling network for Dictyostelium cAMP relay.

Includes:
- CAR1 receptor activation
- ACA (adenylyl cyclase) regulation
- PKA/ERK2 feedback loops
- RegA phosphodiesterase
- cAMP synthesis and secretion
- Extracellular PDE production
"""
const REACTION_NETWORK = @reaction_network begin
    # CAR1 → ACA activation
    k1, CAR1 --> ACA + CAR1
    
    # PKA regulation
    k2, ACA + PKA --> PKA
    k3, cAMPi --> PKA + cAMPi
    k4, PKA --> 0
    
    # ERK2 pathway
    k5, CAR1 --> ERK2 + CAR1
    k6, PKA + ERK2 --> PKA
    
    # RegA regulation
    k7, 0 --> RegA
    k8, ERK2 + RegA --> ERK2
    
    # cAMP dynamics
    k9, ACA --> cAMPi + ACA
    k10, RegA + cAMPi --> RegA
    k11, ACA --> cAMPe + ACA + cAMPtracked
    
    # Receptor dynamics
    k13, cAMPe --> CAR1 + cAMPe
    k14, CAR1 --> 0
    
    # Extracellular PDE production (basal + induced)
    k15_basal, 0 --> PDEe
    (k15_ind * (K^2 * CAR1^2) / (K^2 + CAR1^2)^2), 0 --> PDEe
    
    # Extracellular reactions
    decay_rate, cAMPe + PDEe --> PDEe
    decay_pde, PDEe --> 0
end

# =============================================================================
# Agent Definition
# =============================================================================

@agent struct CellAgent(GridAgent{2})
    name::String
    network::ReactionSystem
    integrator::JumpProcesses.SSAIntegrator
    camp_u_idx::Int
    camptrack_u_idx::Int
    pde_u_idx::Int
    tracker::Vector{Int}
end

# =============================================================================
# Agent Stepping Functions
# =============================================================================

"""
Placeholder for Agents.jl agent_step! callback (actual stepping in model_step!).
"""
@inline function step_agent!(agent, model)
    # Stepping handled in model_step! for synchronization
end

"""
    step_agent_internal!(agent, model, dt)

Execute one SSA step for the agent's intracellular network.
Synchronizes grid ↔ agent molecule counts before/after stepping.
"""
@inbounds function step_agent_internal!(agent::CellAgent, model, dt::Float64)
    h, w = agent.pos
    integrator = agent.integrator

    # Sync: Grid → Agent (extracellular species)
    integrator.u[agent.camp_u_idx] = model.data[h, w, SPECIES_MAP[:cAMPe]]
    integrator.u[agent.pde_u_idx] = model.data[h, w, SPECIES_MAP[:PDEe]]
    integrator.u[agent.camptrack_u_idx] = 0

    # Reset SSA aggregator after manual state change
    reset_aggregated_jumps!(integrator)

    # Run SSA for dt
    step!(integrator, dt, true)

    # Sync: Agent → Grid
    agent.tracker[abmtime(model)+1] = integrator.u[agent.camptrack_u_idx]
    model.data[h, w, SPECIES_MAP[:cAMPe]] = integrator.u[agent.camp_u_idx]
    model.data[h, w, SPECIES_MAP[:PDEe]] = integrator.u[agent.pde_u_idx]
end

# =============================================================================
# Diffusion Implementation
# =============================================================================

"""
    draw_fluxes!(buf, n, p)

Draw multinomial fluxes for diffusion using sequential binomials.
Fills `buf` with [N, S, E, W, stay] molecule counts.
"""
@inline function draw_fluxes!(buf, n::Int, p::Float64)
    r = n

    x1 = rand(Binomial(r, p))
    r -= x1

    q2 = p / (1 - p)
    x2 = rand(Binomial(r, q2))
    r -= x2

    q3 = p / (1 - 2p)
    x3 = rand(Binomial(r, q3))
    r -= x3

    q4 = p / (1 - 3p)
    x4 = rand(Binomial(r, q4))
    r -= x4

    buf[1] = x1
    buf[2] = x2
    buf[3] = x3
    buf[4] = x4
    buf[5] = r
    return nothing
end

"""
    apply_diffusion!(data, rates, dt)

Apply stochastic diffusion to all species on the grid.
Uses absorbing boundary conditions.
"""
function apply_diffusion!(data::Array{Int,3}, rates, dt)
    height, width, species_count = size(data)
    delta = zeros(Int, height, width, species_count)

    for s in 1:species_count
        p_jump = rates[s] * dt / (DX^2)

        if p_jump > 0.24
            @warn "Stability risk: p_jump ($p_jump) exceeds 0.24"
        end

        fluxes = MVector{5,Int}(undef)
        
        @inbounds for h in 1:height, w in 1:width
            n_total = data[h, w, s]
            n_total <= 0 && continue

            draw_fluxes!(fluxes, n_total, p_jump)

            # Subtract outgoing molecules
            n_moving = n_total - fluxes[5]
            delta[h, w, s] -= n_moving

            # Add to neighbors (absorbing boundaries)
            h < height && (delta[h+1, w, s] += fluxes[1])  # North
            h > 1 && (delta[h-1, w, s] += fluxes[2])       # South
            w < width && (delta[h, w+1, s] += fluxes[3])   # East
            w > 1 && (delta[h, w-1, s] += fluxes[4])       # West
        end
    end

    data .+= delta
end

# =============================================================================
# Decay Reactions (Extracellular)
# =============================================================================

"""
    decay_tau(n_src, n_kat, c_cAMP, c_PDE, dt)

Tau-leaping approximation for decay reactions (high molecule counts).
"""
@inline function decay_tau(n_src, n_kat, c_cAMP, c_PDE, dt::Float64)
    λ_camp = (c_cAMP * n_src * n_kat) * dt
    λ_pde = (c_PDE * n_kat) * dt

    d_camp = λ_camp > 0 ? rand(Poisson(λ_camp)) : 0
    d_pde = λ_pde > 0 ? rand(Poisson(λ_pde)) : 0

    d_camp = min(d_camp, n_src)
    d_pde = min(d_pde, n_kat)

    return (n_src - d_camp, n_kat - d_pde)
end

"""
    decay_ssa(n_src, n_kat, c_cAMP, c_PDE, endtime)

Exact SSA for decay reactions (low molecule counts).
"""
@inline @inbounds function decay_ssa(n_src, n_kat, c_cAMP, c_PDE, endtime)
    t = 0.0
    
    while n_src > 0 || n_kat > 0
        prob_PDEe = n_kat * c_PDE
        prob_cAMP = (n_src * n_kat) * c_cAMP
        total = prob_cAMP + prob_PDEe

        total <= 0.0 && break

        t += randexp() / total
        t > endtime && break

        if rand() * total < prob_cAMP
            n_src -= 1
        else
            n_kat -= 1
        end
    end
    
    return (n_src, n_kat)
end

"""
    apply_decay!(data, agent_mask, src_idx, kat_idx, endtime)

Apply extracellular decay reactions to grid voxels not occupied by agents.
Uses SSA for low counts, tau-leaping for high counts.
"""
@inline function apply_decay!(data, agent_mask, src_idx::Int, kat_idx::Int, endtime::Float64)
    height = size(data, 1)
    width = size(data, 2)

    c_cAMP = DECAY_RATE / (NA * DX^3)
    c_PDE = DECAY_PDE

    for w in 1:width
        @inbounds for h in 1:height
            agent_mask[w, h] && continue

            n_src = data[h, w, src_idx]
            n_kat = data[h, w, kat_idx]

            (n_src <= 0 || n_kat <= 0) && continue

            if n_src < 50 && n_kat < 50
                n_src, n_kat = decay_ssa(n_src, n_kat, c_cAMP, c_PDE, endtime)
            else
                n_src, n_kat = decay_tau(n_src, n_kat, c_cAMP, c_PDE, endtime)
            end

            data[h, w, src_idx] = n_src
            data[h, w, kat_idx] = n_kat
        end
    end
end

# =============================================================================
# Model Stepping
# =============================================================================

"""
Build boolean mask of grid positions occupied by agents.
"""
@inline function build_agent_mask!(model)
    for agent in allagents(model)
        h, w = agent.pos
        @inbounds model.agent_mask[h, w] = true
    end
end

"""
Model step: execute agent reactions, decay, and diffusion.
"""
function model_step!(model)
    dt = model.dt
    
    # Step all agents
    for agent in allagents(model)
        step_agent_internal!(agent, model, dt)
    end

    # Extracellular decay (outside agent voxels)
    apply_decay!(model.data, model.agent_mask, cAMP_IDX, PDEe_IDX, dt)

    # Diffusion (split-step for stability)
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)
    apply_diffusion!(model.data, DIFFUSION_RATES, dt / 2)

    model.currentTime += dt
end

# =============================================================================
# Agent Initialization
# =============================================================================

"""
Generate perturbed initial conditions for an agent.
"""
function generate_initial_u0(perturbation::Float64, species_list)
    u0_dict = Dict{Symbol,Int}()

    for (spec, nominal) in NOMINAL_U0
        if spec in species_list
            d_i = rand() * 2 - 1
            perturbed_value = nominal * (1 + perturbation * d_i)
            u0_dict[spec] = round(Int, perturbed_value)
        else
            u0_dict[spec] = 0
        end
    end

    return u0_dict
end

"""
Compute reaction rate parameters with proper unit conversions.

# Unit conversions:
- Unimolecular: rate / TIME_SCALE
- Bimolecular: rate / (NA * V * µM * TIME_SCALE)
- Zero-order: rate * NA * V * µM / TIME_SCALE
"""
function compute_reaction_params(k15_ind::Float64, K::Float64)
    TIME_SCALE = 60  # Convert min⁻¹ to s⁻¹

    return Dict(
        # Unimolecular
        :k1 => 1.08,
        :k3 => 2.5 / TIME_SCALE,
        :k4 => 1.5 / TIME_SCALE,
        :k5 => 0.6 / TIME_SCALE,
        :k9 => 0.3 / TIME_SCALE,
        :k11 => 1.0,
        :k13 => 0.71,
        :k14 => 4.5 / TIME_SCALE,

        # Bimolecular
        :k2 => 0.9 / NA / V / MU / TIME_SCALE,
        :k6 => 0.8 / NA / V / MU / TIME_SCALE,
        :k8 => 1.3 / NA / V / MU / TIME_SCALE,
        :k10 => 0.8 / NA / V / MU / TIME_SCALE,

        # Zero-order
        :k7 => 1.0 * NA * V * MU / TIME_SCALE,

        # PDE production
        :k15_basal => 3e-4 * NA * V * MU,
        :k15_ind => k15_ind * NA * V * MU / TIME_SCALE,
        :K => K,

        # Extracellular reactions
        :decay_rate => DECAY_RATE / (NA * DX^3 * 1e-18),
        :decay_pde => DECAY_PDE
    )
end

"""
Add a cell agent to the model at the specified position.
"""
function add_cell_agent!(
    k15_ind::Float64,
    K::Float64,
    model::AgentBasedModel,
    pos::Tuple{Int,Int};
    has_cAMPe::Bool=true,
    name::String="",
    steps::Int=DEFAULT_STEPS,
    dt::Float64=DEFAULT_DT
)
    reaction_params = compute_reaction_params(k15_ind, K)

    species_list = [:PKA, :ERK2, :RegA, :cAMPi, :PDEe]
    if has_cAMPe
        append!(species_list, [:cAMPe, :CAR1, :ACA])
    end

    initial_u0 = generate_initial_u0(0.05, species_list)

    # Initialize grid with agent's cAMPe
    model.data[pos..., SPECIES_MAP[:cAMPe]] = initial_u0[:cAMPe]

    # Find species indices in reaction network
    rn_species_syms = ModelingToolkit.getname.(species(REACTION_NETWORK))
    camp_rn_idx = findfirst(==(:cAMPe), rn_species_syms)
    pde_rn_idx = findfirst(==(:PDEe), rn_species_syms)
    camptrack_idx = findfirst(==(:cAMPtracked), rn_species_syms)

    # Create SSA problem and integrator
    prob = DiscreteProblem(REACTION_NETWORK, initial_u0, (0.0, dt), reaction_params)
    jp = JumpProblem(REACTION_NETWORK, prob, NRM(); save_positions=(false, false))
    integrator = init(jp, SSAStepper())

    add_agent!(
        pos, CellAgent, model;
        network=REACTION_NETWORK,
        integrator=integrator,
        camp_u_idx=camp_rn_idx,
        pde_u_idx=pde_rn_idx,
        camptrack_u_idx=camptrack_idx,
        tracker=zeros(Int, steps),
        name=name
    )
end

"""
Place agents in a line at the vertical center of the grid.
First agent is transmitter (TX), last is receiver (RX), others are relays.
"""
function add_agents_line!(k15_ind::Float64, K::Float64, model, n_agents::Int; steps::Int, dt::Float64)
    n_agents >= 1 || throw(ArgumentError("n_agents must be at least 1"))

    h = cld(HEIGHT, 2)  # Vertical center
    usable_width = WIDTH - 2 * MIN_BORDER_DIST

    usable_width > 0 || throw(ArgumentError("Usable width must be > 0"))

    spacing = usable_width / (n_agents + 1)

    for i in 1:n_agents
        x = clamp(round(Int, MIN_BORDER_DIST + spacing * i), 1, WIDTH)
        pos = (x, h)

        name = if i == 1
            "TX"
        elseif i == n_agents
            "RX"
        else
            "R$(i-1)"
        end

        add_cell_agent!(
            k15_ind, K, model, pos;
            has_cAMPe=(i == 1),
            name=name,
            steps=steps,
            dt=dt
        )
    end
end

# =============================================================================
# Model Initialization
# =============================================================================

"""
    initialize_model(k15_ind, K, n_agents; kwargs...)

Create and initialize the agent-based model.

# Arguments
- `k15_ind`: PDE induction rate parameter
- `K`: Hill function parameter  
- `n_agents`: Number of cells

# Keyword Arguments
- `grid_size`: Tuple (width, height), default (500, 200)
- `seed`: Random seed, default 42
- `steps`: Total simulation steps
- `dt`: Time step size
"""
function initialize_model(
    k15_ind::Float64,
    K::Float64,
    n_agents::Int;
    grid_size::Tuple{Int,Int}=(WIDTH, HEIGHT),
    seed::Int=42,
    steps::Int=DEFAULT_STEPS,
    dt::Float64=DEFAULT_DT
)
    space = GridSpace(grid_size; periodic=false)

    # Initialize grid data
    data = zeros(Int, grid_size[1], grid_size[2], SPECIES_COUNT)
    mask = falses(grid_size[1], grid_size[2])

    properties = Dict(
        :data => data,
        :currentTime => 0.0,
        :agent_mask => mask,
        :dt => dt
    )

    model = StandardABM(
        CellAgent,
        space;
        agent_step!=step_agent!,
        model_step!=model_step!,
        properties=properties,
        rng=Xoshiro(seed)
    )

    add_agents_line!(k15_ind, K, model, n_agents; steps=steps, dt=dt)
    build_agent_mask!(model)

    return model
end

# =============================================================================
# Output Functions
# =============================================================================

"""
Initialize tracking data structure for all agents.
"""
function init_tracking(model)
    tracked = Dict{Int,Dict{Symbol,Vector{Int}}}()

    for a in allagents(model)
        tracked[a.id] = Dict(
            Symbol(s) => Int[] for s in species(a.network)
        )
    end

    return tracked
end

"""
Write simulation results to CSV files.
"""
function write_results(output_dir::String, times::Vector{Float64}, tracked, agents, steps::Int)
    mkpath(output_dir)
    println("Writing simulation data to '$output_dir'...")

    agent_names = Dict(a.id => a.name for a in agents)

    # Write per-agent molecule trajectories
    for (agent_id, data) in tracked
        agent_name = get(agent_names, agent_id, "unknown_$(agent_id)")
        file_path = joinpath(output_dir, "agent_$(agent_name)_molecules.csv")

        open(file_path, "w") do io
            spec_keys = collect(keys(data))
            clean_names = [replace(string(k), "(t)" => "") for k in spec_keys]

            # Header
            print(io, "time")
            for name in clean_names
                print(io, ",", name)
            end
            println(io)

            # Data rows
            for (t_idx, t_val) in enumerate(times)
                @printf(io, "%.4f", t_val)
                for k in spec_keys
                    vals = data[k]
                    val = t_idx <= length(vals) ? vals[t_idx] : 0
                    print(io, ",", val)
                end
                println(io)
            end
        end
    end

    # Write tracker data (cAMP secretion events)
    tracker_path = joinpath(output_dir, "agent_tracker.csv")
    open(tracker_path, "w") do io
        # Header
        println(io, join([a.name for a in agents], ","))

        # Data rows
        for i in 1:steps
            println(io, join([a.tracker[i] for a in agents], ","))
        end
    end

    println("Results saved to '$output_dir'")
end

# =============================================================================
# Main Simulation Function
# =============================================================================

"""
    run_simulation(; kwargs...)

Run the Dictyostelium cAMP signaling simulation.

# Keyword Arguments
- `k15_ind`: PDE induction rate (default: 1.08)
- `K`: Hill function parameter (default: 4000.0)
- `n_agents`: Number of cells (default: 2)
- `steps`: Simulation steps (default: 100_000)
- `dt`: Time step (default: 0.002)
- `output_suffix`: Output directory suffix (default: "run")
- `create_movie`: Generate visualization (default: false)
- `seed`: Random seed (default: 42)
"""
function run_simulation(;
    k15_ind::Float64=1.08,
    K::Float64=4000.0,
    n_agents::Int=2,
    steps::Int=DEFAULT_STEPS,
    dt::Float64=DEFAULT_DT,
    output_suffix::String="run",
    create_movie::Bool=false,
    seed::Int=42
)
    println("Dictyostelium cAMP Simulation")
    println("  k15_ind = $k15_ind, K = $K")
    println("  n_agents = $n_agents, steps = $steps")
    println("  Threads available: $(Threads.nthreads())")

    model = initialize_model(k15_ind, K, n_agents; seed=seed, steps=steps, dt=dt)

    tracked = init_tracking(model)
    times = Float64[]

    progress = Progress(steps; desc="Simulating", showspeed=true)

    if create_movie
        run_with_visualization!(model, tracked, times, steps, progress)
    else
        run_headless!(model, tracked, times, steps, progress)
    end

    output_dir = "output_csv_" * output_suffix
    write_results(output_dir, times, tracked, collect(allagents(model)), steps)

    return model, tracked, times
end

"""
Run simulation without visualization.
"""
function run_headless!(model, tracked, times, steps, progress)
    @inbounds for _ in 1:(steps ÷ STEPSIZE)
        step!(model, STEPSIZE)

        # Record state
        for a in allagents(model)
            u = a.integrator.u
            sp = species(a.network)
            for (i, s) in enumerate(sp)
                push!(tracked[a.id][Symbol(s)], u[i])
            end
        end

        push!(times, model.currentTime)
        next!(progress; step=STEPSIZE)
    end
end

"""
Run simulation with CairoMakie visualization and video output.
"""
function run_with_visualization!(model, tracked, times, steps, progress)
    dt = model.dt
    time_obs = Observable(model.currentTime)

    fig = Figure(resolution=(400 * SPECIES_COUNT, 400))
    observables = [Observable(model.data[:, :, s]) for s in 1:SPECIES_COUNT]
    agent_pos_obs = Observable([Point2f(a.pos) for a in allagents(model)])

    for s in 1:SPECIES_COUNT
        title_obs = @lift("Species $s | Time: $(round($time_obs, digits=2))s")
        ax = Axis(fig[1, s], title=title_obs, aspect=DataAspect())
        hm = heatmap!(ax, observables[s], colormap=:viridis)
        scatter!(ax, agent_pos_obs, color=:white, markersize=8, strokewidth=1, strokecolor=:black)
        Colorbar(fig[2, s], hm, vertical=false, label="Intensity S$s")
    end

    record(fig, "simulation.mp4", 1:(steps ÷ STEPSIZE); framerate=round(Int, 1 / (dt * STEPSIZE))) do _
        step!(model, STEPSIZE)

        time_obs[] = model.currentTime
        agent_pos_obs[] = [Point2f(a.pos) for a in allagents(model)]

        for s in 1:SPECIES_COUNT
            observables[s][] = model.data[:, :, s]
        end

        for a in allagents(model)
            u = a.integrator.u
            sp = species(a.network)
            for (i, s) in enumerate(sp)
                push!(tracked[a.id][Symbol(s)], u[i])
            end
        end

        push!(times, model.currentTime)
        next!(progress; step=STEPSIZE)
    end
end

end # module

# =============================================================================
# Command-Line Interface
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using .DictyosteliumSimulation

    function parse_args()
        output_suffix = length(ARGS) >= 1 ? ARGS[1] : "run"
        n_agents = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
        k15_ind = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.08
        K = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 4000.0

        return (
            output_suffix=output_suffix,
            n_agents=n_agents,
            k15_ind=k15_ind,
            K=K
        )
    end

    function main()
        args = parse_args()

        run_simulation(;
            k15_ind=args.k15_ind,
            K=args.K,
            n_agents=args.n_agents,
            output_suffix=args.output_suffix
        )
    end

    main()
end
