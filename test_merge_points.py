import pyvista as pv
import numpy as np
import tetgen

print("Testing point merging strategy...")

# Create a sphere (all triangles)
sphere = pv.Sphere().triangulate()
print(f"Original: n_points={sphere.n_points}, n_cells={sphere.n_cells}, is_all_triangles={sphere.is_all_triangles}")

# Internal points
internal_points = np.array([[0,0,0], [0.1,0.1,0.1]])

# Strategy: Append points, keep faces
# faces array in PyVista is [n_nodes, i0, i1, i2, n_nodes, j0, j1, j2, ...]
# Since we append points at the end, original indices are valid.

new_points = np.vstack((sphere.points, internal_points))
new_mesh = pv.PolyData(new_points, sphere.faces)

print(f"New mesh: n_points={new_mesh.n_points}, n_cells={new_mesh.n_cells}")
print(f"is_all_triangles: {new_mesh.is_all_triangles}")

try:
    tet = tetgen.TetGen(new_mesh)
    print("TetGen initialized successfully.")
    tet.tetrahedralize(switches="pq1.4")
    print(f"Tetrahedralized: {tet.grid.n_points} points")
except Exception as e:
    print(f"TetGen failed: {e}")
