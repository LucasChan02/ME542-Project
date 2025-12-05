import numpy as np
import pyvista as pv
import os
import sys
import shutil

# Import project modules
from initial_vertices import generate_initial_vertices
from tetgen_mesh_convert import generate_tetgen_mesh

# --- Configuration ---
# Modify these variables to configure the optimization
OBJ_PATH = "scan2_volume_v7.obj"
NUM_INTERNAL_POINTS = 1000   # Number of "particles" (internal points) per design instance
NUM_INSTANCES = 2          # Population size (number of design instances)
MAX_ITER = 5               # Number of iterations
W = 0.5                    # Inertia weight
C1 = 1.5                   # Cognitive coefficient
C2 = 1.5                   # Social coefficient
OUTPUT_DIR = "pso_results"
SOLVER_TYPE = "skfem"      # Options: "skfem", "fenicsx"

# Import simulation function based on configuration
if SOLVER_TYPE == "skfem":
    from simulation.simulation_skfem import run_simulation
elif SOLVER_TYPE == "fenicsx":
    try:
        from simulation.simulation_xdmf import run_simulation
    except ImportError:
        print("Error: FEniCSx solver not available. Please install dolfinx.")
        sys.exit(1)
else:
    print(f"Error: Unknown solver type '{SOLVER_TYPE}'")
    sys.exit(1)

class DesignInstance:
    """
    Represents a single design candidate (a mesh configuration).
    Contains a set of internal points (particles) that define the mesh.
    """
    def __init__(self, position, velocity):
        # position: (N, 3) array of internal point coordinates
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float('inf') # Minimizing stress
        self.current_fitness = float('inf')

class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) for Mesh Optimization.
    Optimizes the position of internal vertices (particles) to minimize maximum von Mises stress.
    """
    def __init__(self, obj_path, num_internal_points, num_instances=5, max_iter=10, 
                 w=0.5, c1=1.5, c2=1.5, output_dir="pso_results"):
        self.obj_path = obj_path
        self.num_internal_points = num_internal_points
        self.num_instances = num_instances
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.output_dir = output_dir
        
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.instances = []
        
        # Load mesh for boundary checking
        self.mesh = pv.read(obj_path)
        if not self.mesh.is_all_triangles:
            self.mesh = self.mesh.triangulate()
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not os.path.exists("log"):
            os.makedirs("log")

    def initialize(self):
        print("Initializing swarm...")
        for i in range(self.num_instances):
            print(f"  Initializing instance {i+1}/{self.num_instances}...")
            # Generate random valid positions for internal points (particles)
            pos = generate_initial_vertices(self.obj_path, self.num_internal_points, inward_offset=0.6)
            # Initialize velocity (small random values)
            vel = (np.random.rand(*pos.shape) - 0.5) * 0.1
            
            instance = DesignInstance(pos, vel)
            self.instances.append(instance)
            
        print("Swarm initialized.")

    def check_bounds(self, points):
        """
        Ensures points remain inside the mesh boundary.
        Points outside are pulled towards the mesh center.
        """
        # Check implicit distance
        pts_poly = pv.PolyData(points)
        target = pts_poly.compute_implicit_distance(self.mesh)
        distances = target['implicit_distance']
        
        # Mask for points outside (distance > -offset)
        inward_offset = 0.6
        mask = distances > -inward_offset
        
        if np.any(mask):
            # Move invalid points towards center until they are inside
            center = np.array(self.mesh.center)
            points[mask] = points[mask] * 0.9 + center * 0.1 
            
        return points

    def evaluate_fitness(self, instance, iteration, instance_idx):
        # --- 1. Generate Mesh ---
        run_name = f"iter_{iteration}_inst_{instance_idx}"
        mesh_base = os.path.join(self.output_dir, run_name)
        
        try:
            generate_tetgen_mesh(self.obj_path, mesh_base, instance.position)
        except Exception as e:
            print(f"    Mesh generation failed: {e}")
            return float('inf')

        # --- 2. Run Simulation ---
        mesh_vol = mesh_base + "_vol.xdmf"
        
        if not os.path.exists(mesh_vol):
            print("    Mesh file not found.")
            return float('inf')
            
        try:
            # Redirect stdout to log file
            log_file = os.path.join("log", f"{run_name}.log")
            
            # We need to capture stdout from the imported function
            # This is a bit tricky with direct calls if the function prints to stdout
            # We can redirect sys.stdout temporarily
            
            with open(log_file, "w") as log:
                original_stdout = sys.stdout
                sys.stdout = log
                try:
                    max_stress = run_simulation(mesh_vol)
                finally:
                    sys.stdout = original_stdout
            
            if max_stress is None:
                 return float('inf')
                 
            return max_stress

        except Exception as e:
            print(f"    Simulation failed: {e}")
            return float('inf')

    def optimize(self):
        self.initialize()
        
        for it in range(self.max_iter):
            print(f"\n--- Iteration {it+1}/{self.max_iter} ---")
            
            for i, instance in enumerate(self.instances):
                # Evaluate Fitness
                fitness = self.evaluate_fitness(instance, it, i)
                instance.current_fitness = fitness
                
                print(f"  Instance {i+1}: Stress = {fitness:.2e} Pa")
                
                # Update Personal Best
                if fitness < instance.best_fitness:
                    instance.best_fitness = fitness
                    instance.best_position = instance.position.copy()
                    
                # Update Global Best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = instance.position.copy()
                    print(f"    New Global Best! Stress = {self.global_best_fitness:.2e} Pa")
            
            # Update Instances (Velocity & Position)
            for instance in self.instances:
                if self.global_best_position is None:
                    # Explore randomly if no solution found yet
                    r1 = np.random.rand(*instance.position.shape)
                    instance.velocity = self.w * instance.velocity + r1 * 0.1
                else:
                    r1 = np.random.rand(*instance.position.shape)
                    r2 = np.random.rand(*instance.position.shape)
                    
                    # Standard PSO Velocity Update
                    instance.velocity = (self.w * instance.velocity + 
                                  self.c1 * r1 * (instance.best_position - instance.position) + 
                                  self.c2 * r2 * (self.global_best_position - instance.position))
                
                # Update Position and Check Bounds
                instance.position = instance.position + instance.velocity
                instance.position = self.check_bounds(instance.position)
                
        print("\nOptimization Finished.")
        print(f"Best Stress: {self.global_best_fitness:.2e} Pa")
        
        # --- Save Best Result ---
        if self.global_best_position is not None:
            print("Generating final best mesh...")
            best_mesh_base = os.path.join(self.output_dir, "best_solution")
            generate_tetgen_mesh(self.obj_path, best_mesh_base, self.global_best_position)
            print(f"Best mesh saved to {best_mesh_base}_vol.xdmf")
            
            # Run simulation one last time to generate result file with stress/displacement
            print("Running simulation for best solution to generate output fields...")
            mesh_vol = best_mesh_base + "_vol.xdmf"
            try:
                run_simulation(mesh_vol)
                print(f"Simulation results saved to {best_mesh_base}_result.xdmf")
            except Exception as e:
                print(f"Warning: Failed to run simulation for best solution: {e}")

if __name__ == "__main__":
    print(f"Starting Optimization with:")
    print(f"  OBJ: {OBJ_PATH}")
    print(f"  Internal Points (Particles): {NUM_INTERNAL_POINTS}")
    print(f"  Instances (Population): {NUM_INSTANCES}")
    print(f"  Iterations: {MAX_ITER}")
    
    if not os.path.exists(OBJ_PATH):
        print(f"Error: {OBJ_PATH} not found.")
        sys.exit(1)
        
    optimizer = PSOOptimizer(OBJ_PATH, NUM_INTERNAL_POINTS, NUM_INSTANCES, MAX_ITER, W, C1, C2, OUTPUT_DIR)
    optimizer.optimize()
