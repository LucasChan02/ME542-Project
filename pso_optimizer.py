import numpy as np
import pyvista as pv
import subprocess
import os
import sys
import re
import shutil

# Import project modules
from initial_vertices import generate_initial_vertices
from tetgen_mesh_convert import generate_tetgen_mesh

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_fitness = float('inf') # Minimizing stress
        self.current_fitness = float('inf')

class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) for Mesh Optimization.
    Optimizes the position of internal vertices to minimize maximum von Mises stress.
    """
    def __init__(self, obj_path, num_internal_points, num_particles=5, max_iter=10, 
                 w=0.5, c1=1.5, c2=1.5, output_dir="pso_results"):
        self.obj_path = obj_path
        self.num_internal_points = num_internal_points
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.output_dir = output_dir
        
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.particles = []
        
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
        for i in range(self.num_particles):
            print(f"  Initializing particle {i+1}/{self.num_particles}...")
            # Generate random valid positions
            pos = generate_initial_vertices(self.obj_path, self.num_internal_points, inward_offset=0.6)
            # Initialize velocity (small random values)
            vel = (np.random.rand(*pos.shape) - 0.5) * 0.1
            
            p = Particle(pos, vel)
            self.particles.append(p)
            
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

    def evaluate_fitness(self, particle, iteration, particle_idx):
        # --- 1. Generate Mesh ---
        run_name = f"iter_{iteration}_part_{particle_idx}"
        mesh_base = os.path.join(self.output_dir, run_name)
        
        try:
            generate_tetgen_mesh(self.obj_path, mesh_base, particle.position)
        except Exception as e:
            print(f"    Mesh generation failed: {e}")
            return float('inf')

        # --- 2. Run Simulation ---
        mesh_vol = mesh_base + "_vol.xdmf"
        mesh_surf = mesh_base + "_surf.xdmf"
        
        if not os.path.exists(mesh_vol):
            print("    Mesh file not found.")
            return float('inf')
            
        sim_script = os.path.join("simulation", "simulation_skfem.py")
        cmd = [sys.executable, sim_script, mesh_vol]
        
        try:
            # Write simulation output to log file
            log_file = os.path.join("log", f"{run_name}.log")
            with open(log_file, "w") as outfile:
                subprocess.run(cmd, stdout=outfile, stderr=subprocess.STDOUT, check=True)
            
            # Read output back for parsing
            with open(log_file, "r") as infile:
                output = infile.read()
                
        except subprocess.CalledProcessError as e:
            print(f"    Simulation failed with return code {e.returncode}")
            return float('inf')

        # --- 3. Parse Result ---
        match = re.search(r"MAX_STRESS:\s*([0-9\.eE\+\-]+)", output)
        if match:
            max_stress = float(match.group(1))
            return max_stress
        else:
            print("    Could not parse max stress from output.")
            return float('inf')

    def optimize(self):
        self.initialize()
        
        for it in range(self.max_iter):
            print(f"\n--- Iteration {it+1}/{self.max_iter} ---")
            
            for i, p in enumerate(self.particles):
                # Evaluate Fitness
                fitness = self.evaluate_fitness(p, it, i)
                p.current_fitness = fitness
                
                print(f"  Particle {i+1}: Stress = {fitness:.2e} Pa")
                
                # Update Personal Best
                if fitness < p.best_fitness:
                    p.best_fitness = fitness
                    p.best_position = p.position.copy()
                    
                # Update Global Best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = p.position.copy()
                    print(f"    New Global Best! Stress = {self.global_best_fitness:.2e} Pa")
            
            # Update Particles (Velocity & Position)
            for p in self.particles:
                if self.global_best_position is None:
                    # Explore randomly if no solution found yet
                    r1 = np.random.rand(*p.position.shape)
                    p.velocity = self.w * p.velocity + r1 * 0.1
                else:
                    r1 = np.random.rand(*p.position.shape)
                    r2 = np.random.rand(*p.position.shape)
                    
                    # Standard PSO Velocity Update
                    p.velocity = (self.w * p.velocity + 
                                  self.c1 * r1 * (p.best_position - p.position) + 
                                  self.c2 * r2 * (self.global_best_position - p.position))
                
                # Update Position and Check Bounds
                p.position = p.position + p.velocity
                p.position = self.check_bounds(p.position)
                
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
            sim_script = os.path.join("simulation", "simulation_skfem.py")
            cmd = [sys.executable, sim_script, mesh_vol]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Simulation results saved to {best_mesh_base}_result.xdmf")
            except subprocess.CalledProcessError:
                print("Warning: Failed to run simulation for best solution.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PSO Mesh Optimizer")
    parser.add_argument("--obj", type=str, default="scan2_volume_v7.obj", help="Path to input OBJ file")
    parser.add_argument("--points", type=int, default=50, help="Number of internal vertices")
    parser.add_argument("--population", type=int, default=2, help="Population size (number of particles/meshes per iteration)")
    parser.add_argument("--iter", type=int, default=1, help="Number of iterations")
    
    args = parser.parse_args()
    
    OBJ_PATH = args.obj
    NUM_INTERNAL_POINTS = args.points
    NUM_PARTICLES = args.population
    MAX_ITER = args.iter
    
    print(f"Starting Optimization with:")
    print(f"  OBJ: {OBJ_PATH}")
    print(f"  Internal Points: {NUM_INTERNAL_POINTS}")
    print(f"  Population Size: {NUM_PARTICLES}")
    print(f"  Iterations: {MAX_ITER}")
    
    if not os.path.exists(OBJ_PATH):
        print(f"Error: {OBJ_PATH} not found.")
        sys.exit(1)
        
    optimizer = PSOOptimizer(OBJ_PATH, NUM_INTERNAL_POINTS, NUM_PARTICLES, MAX_ITER)
    optimizer.optimize()
