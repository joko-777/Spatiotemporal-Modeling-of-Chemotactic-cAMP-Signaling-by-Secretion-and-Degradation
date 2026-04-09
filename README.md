# Dictyostelium cAMP Signaling Simulation

Stochastic agent-based simulation of cAMP signaling dynamics in *Dictyostelium discoideum* cells.

## Overview

This simulation models the relay of cAMP signals between Dictyostelium cells using:

- **Intracellular signaling**: Gillespie SSA (Stochastic Simulation Algorithm) for reaction networks via [Catalyst.jl](https://github.com/SciML/Catalyst.jl)
- **Spatial dynamics**: 2D grid-based diffusion with absorbing boundaries
- **Agent-based framework**: Cell representation via [Agents.jl](https://github.com/JuliaDynamics/Agents.jl)

### Biological Model

The intracellular network includes:
- CAR1 receptor activation by extracellular cAMP
- ACA (adenylyl cyclase) regulation via PKA feedback
- ERK2/RegA phosphodiesterase pathway
- cAMP synthesis, secretion, and degradation
- Extracellular PDE (phosphodiesterase) production with Hill-type induction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dictyostelium-simulation.git
cd dictyostelium-simulation

# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Dependencies

- Julia ≥ 1.9
- Agents.jl
- Catalyst.jl
- JumpProcesses.jl
- CairoMakie.jl
- Distributions.jl
- CSV.jl, DataFrames.jl

## Usage

### Command Line

```bash
# Basic run with 5 agents
julia dictyostelium_simulation.jl output_name 5

# With custom parameters
julia dictyostelium_simulation.jl run1 5 1.08 4000
```

**Arguments:**
| Position | Name | Description | Default |
|----------|------|-------------|---------|
| 1 | `output_suffix` | Suffix for output directory | `"run"` |
| 2 | `n_agents` | Number of cells | `2` |
| 3 | `k15_ind` | PDE induction rate | `1.08` |
| 4 | `K` | Hill function parameter | `4000.0` |

### Programmatic API

```julia
using .DictyosteliumSimulation

model, tracked, times = run_simulation(
    k15_ind = 1.08,
    K = 4000.0,
    n_agents = 5,
    steps = 100_000,
    dt = 0.002,
    output_suffix = "experiment1",
    create_movie = false,
    seed = 42
)
```

## Output

Results are saved to `output_csv_<suffix>/`:

| File | Description |
|------|-------------|
| `agent_TX_molecules.csv` | Transmitter cell molecule counts over time |
| `agent_RX_molecules.csv` | Receiver cell molecule counts over time |
| `agent_R<n>_molecules.csv` | Relay cell molecule counts |
| `agent_tracker.csv` | cAMP secretion events per timestep |

### Output Format

Each molecule file contains time-series data:
```csv
time,ACA,PKA,ERK2,RegA,cAMPi,cAMPe,CAR1,PDEe,cAMPtracked
0.0200,0,0,0,22,0,700,0,0,0
0.0400,1,0,0,22,0,698,2,0,3
...
```

## Model Parameters

### Reaction Rate Parameters

See `compute_reaction_params()` in the source for full parameter set with unit conversions.

## Algorithm Details

### Operator Splitting

Each timestep combines:
1. **Agent reactions**: SSA via Next Reaction Method (NRM)
2. **Extracellular decay**: Hybrid SSA/tau-leaping based on molecule counts
3. **Diffusion**: Split-step stochastic diffusion (2 × dt/2)

### Stochastic Diffusion

Molecules jump to adjacent voxels with probability:
```
p_jump = D × dt / dx²
```
Absorbing boundary conditions (molecules at edges are removed).

### Decay Reactions

- **Low counts** (< 50 molecules): Exact SSA
- **High counts** (≥ 50 molecules): Tau-leaping approximation

## Visualization

Enable video output with `create_movie = true`:

```julia
run_simulation(create_movie = true, ...)
```

Generates `simulation.mp4` showing:
- Heatmaps of cAMP and PDE concentrations
- Agent positions marked in white

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dictyostelium_simulation,
  author = {Peter},
  title = {Dictyostelium cAMP Signaling Simulation},
  year = {2025},
  institution = {FAU Erlangen-Nürnberg, LHFT}
}
```

## References

- Martiel, J. L., & Goldbeter, A. (1987). A model based on receptor desensitization for cyclic AMP signaling in Dictyostelium cells.
- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions.
