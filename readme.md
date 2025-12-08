# ME542 Project: PSO Mesh Optimization

This project implements a Particle Swarm Optimization (PSO) algorithm to optimize the location of internal vertices in a tetrahedral mesh. The goal is to minimize the maximum von Mises stress in the simulated part under defined boundary conditions.

## Overview

The workflow consists of:
1.  **Mesh Generation**: `TetGen` is used to generate a tetrahedral mesh from an input surface (`.obj`) and a set of internal vertices.
2.  **Simulation**: Solves a linear elasticity problem on the generated mesh.
    *   **Solvers**: Switchable between `scikit-fem` and `FEniCSx`.
    *   **Material**: BASF Ultrafuse TPU 95A.
    *   **BCs**: Fixed bottom surface, traction load on top surface.
3.  **Optimization**: A PSO algorithm adjusts the positions of the internal vertices to minimize the peak stress.

## Requirements

*   Python 3.x
*   `numpy`
*   `scikit-fem`
*   `meshio`
*   `pyvista`
*   `tetgen` (Python wrapper or executable accessible)

**For FEniCSx Solver:**
*   `fenics-dolfinx`
*   `petsc4py`
*   `mpi4py`

To install the FEniCSx environment:
```bash
conda install -c conda-forge fenics-dolfinx gmsh python-gmsh h5py pyvista tetgen meshio mpich petsc4py
```

## Usage

1.  **Configure**: Open `pso_optimizer.py` and modify the configuration variables at the top of the file:

    ```python
    # --- Configuration ---
    OBJ_PATH = "scan2_volume_v7.obj"
    NUM_INTERNAL_POINTS = 50   # Number of internal vertices to optimize
    NUM_INSTANCES = 2          # Population size (number of design instances)
    MAX_ITER = 1               # Number of iterations
    SOLVER_TYPE = "skfem"      # Options: "skfem", "fenicsx"
    ```

    *   `NUM_INTERNAL_POINTS`: Number of vertices inside the mesh that the PSO will move.
    *   `NUM_INSTANCES`: Number of candidate meshes (particles) in the swarm.
    *   `SOLVER_TYPE`: Choose between `"skfem"` (easier setup) or `"fenicsx"` (robust).

2.  **Run**: Execute the script directly.

    ```bash
    python pso_optimizer.py
    ```

## Output

Results are saved in the `pso_results` directory:

*   **`best_solution_vol.xdmf`**: The final optimized volume mesh.
*   **`best_solution_result.xdmf`** (or `_displacement.bp` / `_vonmises.bp` for FEniCSx): The simulation results for the best mesh, containing displacement and stress fields.
*   **`log/`**: Detailed logs and CSV history (`pso_history.csv`) of the optimization process.

## File Structure

*   `pso_optimizer.py`: Main entry point. Implements the PSO algorithm.
*   `simulation/simulation_skfem.py`: FEA simulation script using `scikit-fem`.
*   `simulation/simulation_xdmf.py`: FEA simulation script using `FEniCSx`.
*   `tetgen_mesh_convert.py`: Helper to interface with TetGen.
*   `initial_vertices.py`: Helper to generate random initial points inside the volume.
