import numpy as np
from tetgen_mesh_convert import generate_tetgen_mesh

# Create dummy distant points to simulate exploding PSO particles
# 1000 points at huge coordinates
huge_points = np.ones((1000, 3)) * 1e12

print("Attempting mesh generation with huge internal points...")
try:
    generate_tetgen_mesh("scan2_volume_v7.obj", "test_fail_case", huge_points)
except Exception as e:
    print(f"\nCaught Expected Exception: {e}")
