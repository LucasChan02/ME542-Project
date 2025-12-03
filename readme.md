# ME542 Project: PSO Mesh Optimization

This project implements a Particle Swarm Optimization (PSO) algorithm to optimize the location of internal vertices in a tetrahedral mesh. The goal is to minimize the maximum von Mises stress in the simulated part under defined boundary conditions.

## Overview

The workflow consists of:
1.  **Mesh Generation**: `TetGen` is used to generate a tetrahedral mesh from an input surface (`.obj`) and a set of internal vertices.
2.  **Simulation**: `scikit-fem` solves a linear elasticity problem on the generated mesh.
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

```bash
conda install -c conda-forge fenics-dolfinx gmsh python-gmsh h5py pyvista tetgen meshio fenics-dolfinx mpich
pip install tetgen
```

## Usage

Run the optimizer using `pso_optimizer.py`.

```bash
python pso_optimizer.py --points <N> --population <P> --iter <I>
```

### Arguments

*   `--points`: Number of internal vertices to optimize (default: 50).
*   `--population`: Number of particles (candidate meshes) per iteration (default: 2).
*   `--iter`: Number of iterations to run (default: 1).
*   `--obj`: Path to the input OBJ file (default: `scan2_volume_v7.obj`).

### Example

To run an optimization with 400 internal points, a population of 20 particles, and 50 iterations:

```bash
python pso_optimizer.py --points 400 --population 20 --iter 50
```

## Output

Results are saved in the `pso_results` directory:

*   **`best_solution_vol.xdmf`**: The final optimized volume mesh.
*   **`best_solution_result.xdmf`**: The simulation results for the best mesh, containing:
    *   **Displacement**: Vector field of deformation.
    *   **Von Mises Stress**: Scalar field of stress distribution.
*   **`log/`**: Detailed logs for each simulation run.

## File Structure

*   `pso_optimizer.py`: Main entry point. Implements the PSO algorithm.
*   `simulation/simulation_skfem.py`: FEA simulation script using `scikit-fem`.
*   `tetgen_mesh_convert.py`: Helper to interface with TetGen.
*   `initial_vertices.py`: Helper to generate random initial points inside the volume.
