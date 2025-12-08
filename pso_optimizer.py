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
NUM_INTERNAL_POINTS = 400   # Number of "particles" (internal points) per design instance
NUM_INSTANCES = 10          # Population size (number of design instances)
MAX_ITER = 50               # Number of iterations
W = 0.4                    # Inertia weight
C1 = 1.8                   # Cognitive coefficient
C2 = 1.5                   # Social coefficient
OUTPUT_DIR = "pso_results"
SOLVER_TYPE = "fenicsx"      # Options: "skfem", "fenicsx"
BOUNDARY_STRATEGY = "reflective" # Options: "soft", "reflective"

# Import simulation function based on configuration
if SOLVER_TYPE == "skfem":
    from simulation.simulation_skfem import run_simulation
elif SOLVER_TYPE == "fenicsx":
    try:
        from simulation.simulation_xdmf import run_simulation
    except ImportError as e:
        print(f"Error: FEniCSx solver not available. Please install dolfinx. Details: {e}")
        sys.exit(1)
else:
    print(f"Error: Unknown solver type '{SOLVER_TYPE}'")
    sys.exit(1)

class SimpleProgressBar:
    """
    A simple progress bar for the console.
    """
    def __init__(self, total, width=40):
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = None
        
    def log(self, message):
        """Prints a message above the progress bar."""
        # Clear line, print message, then reprint bar (handled by next update)
        sys.stdout.write(f"\r\033[K{message}\n")
        self.display()
        
    def update(self, step=1, info=""):
        """Updates progress by step amount."""
        self.current += step
        if self.current > self.total:
            self.current = self.total
        self.display(info)
        
    def display(self, info=""):
        fraction = self.current / self.total
        filled = int(self.width * fraction)
        bar = "=" * filled + "-" * (self.width - filled)
        percent = int(fraction * 100)
        
        # Clear line and print bar
        sys.stdout.write(f"\r\033[K[{bar}] {percent}% {info}")
        sys.stdout.flush()
        
    def finish(self):
        sys.stdout.write("\n")

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
            
        # Initialize CSV logs in log/ directory
        self.history_file = os.path.join("log", "pso_history.csv")
        self.best_history_file = os.path.join("log", "best_candidate_history.csv")
        
        with open(self.history_file, "w") as f:
            f.write("iteration,instance_idx,stress\n")
            
        with open(self.best_history_file, "w") as f:
            f.write("iteration,stress\n")

    def initialize(self):
        print("Initializing swarm...")
        for i in range(self.num_instances):
            print(f"  Initializing instance {i+1}/{self.num_instances}...")
            # Generate random valid positions for internal points (particles)
            pos = generate_initial_vertices(self.obj_path, self.num_internal_points, inward_offset=0.6)
            
            # Initialize velocity scale relative to domain size
            # Using ~5% of bounding box diagonal is a good heuristic
            bounds = self.mesh.bounds
            self.diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
            vel_scale = self.diag * 0.05
            
            # Initialize velocity
            vel = (np.random.rand(*pos.shape) - 0.5) * vel_scale
            
            instance = DesignInstance(pos, vel)
            self.instances.append(instance)
            
        print("Swarm initialized.")

    def check_bounds(self, points, velocities):
        """
        Ensures points remain inside the mesh boundary.
        Implements different strategies for handling boundary collisions.
        Returns updated (points, velocities).
        """
        # --- Hard Reset for Lost Particles ---
        # If points drift exceedingly far (e.g. > 3x diagonal), likely numerical explosion.
        # Reset them completely to random valid positions.
        center = np.array(self.mesh.center)
        dist_from_center = np.linalg.norm(points - center, axis=1)
        reset_mask = dist_from_center > (self.diag * 3.0)
        
        if np.any(reset_mask):
             num_reset = np.sum(reset_mask)
             # print(f"    WARNING: Hard resetting {num_reset} lost particles.")
             # Re-generate valid points (this is expensive inside loop but necessary for recovery)
             # Efficient workaround: Move to center + small random jitter
             points[reset_mask] = center + (np.random.rand(num_reset, 3) - 0.5) * (self.diag * 0.1)
             velocities[reset_mask] = 0.0 # Reset velocity
        
        # --- Normal Boundary Check ---
        # Check implicit distance
        pts_poly = pv.PolyData(points)
        target = pts_poly.compute_implicit_distance(self.mesh)
        distances = target['implicit_distance']
        
        # Mask for points outside (distance > -offset)
        # Using a small offset to keep points cleanly inside
        inward_offset = 0.6
        mask = distances > -inward_offset
        
        if np.any(mask):
            
            if BOUNDARY_STRATEGY == "reflective":
                # Reflective Strategy:
                # 1. Reverse velocity component to "bounce" 
                # (Simple approximation: reverse full velocity vector)
                velocities[mask] *= -1.0
                
                # 2. Push point back inside
                # Calculate vector from center to point
                vec_to_point = points[mask] - center
                # Push back towards center by a random fraction to avoid sticking to surface
                # Push back 10% towards center
                points[mask] = points[mask] - vec_to_point * 0.1
                
            elif BOUNDARY_STRATEGY == "soft":
                # Soft Strategy:
                # 1. Move invalid points towards center until they are inside
                points[mask] = points[mask] * 0.9 + center * 0.1 
                
                # 2. Dampen velocity to simulate energy loss
                velocities[mask] *= 0.5
                
        return points, velocities

    def evaluate_fitness(self, instance, iteration, instance_idx):
        # --- 1. Generate Mesh ---
        run_name = f"iter_{iteration}_inst_{instance_idx}"
        mesh_base = os.path.join(self.output_dir, run_name)
        
        max_stress = float('inf')
        
        try:
            generate_tetgen_mesh(self.obj_path, mesh_base, instance.position)
            
            # --- 2. Run Simulation ---
            mesh_vol = mesh_base + "_vol.xdmf"
            
            if os.path.exists(mesh_vol):
                # Redirect stdout to a single temporary log file or overwrite usually
                # We use a shared log file to avoid clutter, overwriting it each time
                log_file = os.path.join("log", "latest_simulation.log")
                
                with open(log_file, "w") as log:
                    original_stdout = sys.stdout
                    sys.stdout = log
                    try:
                        result = run_simulation(mesh_vol)
                        if result is not None:
                            max_stress = result
                    finally:
                        sys.stdout = original_stdout
                        
        except Exception as e:
            # print(f"    Processing failed: {e}")
            pass
        
        # Robust Logging: Log result even if it is inf
        try:
            with open(self.history_file, "a") as f:
                f.write(f"{iteration},{instance_idx},{max_stress}\n")
        except Exception as e:
            print(f"Error writing to log: {e}")
            
        return max_stress

    def optimize(self):
        self.initialize()
        
        total_steps = self.max_iter * self.num_instances
        bar = SimpleProgressBar(total_steps)
        
        # Velocity Clamping Limit (10% of domain)
        v_max = self.diag * 0.1
        
        print("\nStarting Optimization loop...")
        
        for it in range(self.max_iter):
            # print(f"\n--- Iteration {it+1}/{self.max_iter} ---")
            
            for i, instance in enumerate(self.instances):
                # Evaluate Fitness
                fitness = self.evaluate_fitness(instance, it, i)
                instance.current_fitness = fitness
                
                # print(f"  Instance {i+1}: Stress = {fitness:.2e} Pa")
                
                # Update Personal Best
                if fitness < instance.best_fitness:
                    instance.best_fitness = fitness
                    instance.best_position = instance.position.copy()
                    
                # Update Global Best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = instance.position.copy()
                    bar.log(f"New Global Best! Stress = {self.global_best_fitness:.2e} Pa (Iter {it}, Inst {i})")
                    
                    # Log new global best
                    with open(self.best_history_file, "a") as f:
                        f.write(f"{it},{fitness}\n")
                        
                bar.update(1, f"Iter {it+1}/{self.max_iter} | Best: {self.global_best_fitness:.2e}")
            
            # Update Instances (Velocity & Position)
            for instance in self.instances:
                if self.global_best_position is None:
                    # Explore randomly if no solution found yet
                    r1 = np.random.rand(*instance.position.shape)
                    instance.velocity = self.w * instance.velocity + r1 * 1.0 # Increased random exploration
                else:
                    r1 = np.random.rand(*instance.position.shape)
                    r2 = np.random.rand(*instance.position.shape)
                    
                    # Standard PSO Velocity Update
                    instance.velocity = (self.w * instance.velocity + 
                                  self.c1 * r1 * (instance.best_position - instance.position) + 
                                  self.c2 * r2 * (self.global_best_position - instance.position))
                
                # Velocity Clamping
                # Clip velocity components to avoid explosion
                instance.velocity = np.clip(instance.velocity, -v_max, v_max)
                
                # Update Position
                instance.position = instance.position + instance.velocity
                
                # Check Bounds and Update Position/Velocity
                instance.position, instance.velocity = self.check_bounds(instance.position, instance.velocity)
        
        bar.finish()        
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
