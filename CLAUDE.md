# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a control-guided neural architecture search (NAS) project that analyzes reachability and performance of control systems under various conditions. The project combines Python and Julia code to compute maximum diameters of reachable sets for different control systems.

## Development Commands

### Setup
```bash
# Install dependencies (requires uv package manager)
uv sync --all-extras
```

### Code Quality
```bash
# Format and lint code
uv run ruff format
uv run ruff check
```

### Running Examples
```bash
# Run basic example
uv run python examples/example.py

# Run split computing example
uv run python examples/split_computing_example.py
```

## Architecture

### Core Components
- **`src/control_guided_nas/wrapper.py`**: Main Python interface that provides `get_max_diam()` function
- **`src/control_guided_nas/get_max_diam.jl`**: Julia implementation for reachability analysis using ReachabilityAnalysis.jl
- **`src/control_guided_nas/Models.jl`**: Julia models for different control systems
- **`src/control_guided_nas/Reachability.jl`**: Julia reachability computation utilities

### System Types
- **Linear Systems**: F1, CC (handled by Julia backend)
  - Available benchmark systems in `Models.jl`: RC (resistor-capacitor), F1 (F1-tenth car), DC (DC motor), CS (car suspension), EW (electronic wedge brake), C1 (cruise control 1), CC (cruise control 2), MPC (model predictive control)
- **Non-linear Systems**: CAR (handled by noisyreach Python library)

### Key Dependencies
- **Julia**: Required for reachability analysis (v1.10+)
- **noisyreach**: External Python library for non-linear systems (from GitHub)
- **juliacall**: Python-Julia bridge for calling Julia functions
- **ReachabilityAnalysis.jl**: Julia package for reachability computations

### Data Flow
1. Python wrapper receives latency, errors, and system name
2. For linear systems: calls Julia backend with system parameters
3. For non-linear systems: uses noisyreach library deviation function
4. Returns maximum diameter of reachable set

## Key Functions

### `get_max_diam(latency, errors, sysname)`
Main function that computes maximum diameter of reachable set given:
- `latency`: Control latency in seconds
- `errors`: Sensor/actuator errors (float or list)
- `sysname`: System name ("F1", "CC", "CAR")

## Testing and Validation

The `data/distance_evaluation/` directory contains validation scripts and test results for the reachability computations.
