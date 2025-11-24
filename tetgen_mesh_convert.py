import pyvista as pv
import tetgen
import numpy as np
import meshio
import sys
import os

def generate_tetgen_mesh(obj_path, output_filename):
    """
    Generates a tetrahedral mesh from an OBJ file using TetGen and exports:
    1. A volume mesh XDMF ({output_filename}.xdmf)
    2. A surface mesh XDMF with tags ({output_filename}_facets.xdmf)
    
    Args:
        obj_path (str): Path to the input .obj file.
        output_filename (str): Base name for the output file (without extension).
    """
    print(f"Reading surface from {obj_path}...")
    # 1. Load and clean surface
    surface = pv.read(obj_path)
    if not surface.is_all_triangles:
        surface = surface.triangulate()
    
    print("Cleaning surface mesh...")
    surface = surface.clean()
    
    # 2. Run TetGen
    print("Running TetGen...")
    tet = tetgen.TetGen(surface)
    
    # Tetrahedralize: p=PLC, q=Quality (1.4)
    tet.tetrahedralize(switches="pq1.4")
    
    grid = tet.grid
    print(f"Generated mesh with {grid.n_points} points and {grid.n_cells} cells.")
    
    # 3. Extract Surface (for mixed mesh output)
    # Use pass_pointid to map surface points back to volume grid indices.
    grid_surf = grid.extract_surface(pass_pointid=True)
    
    # 4. Prepare Data for MeshIO
    points = grid.points
    
    # Extract Tetrahedral Cells (reshape flat array [n_pts, p0...p3])
    if not np.all(grid.celltypes == 10): # 10 is VTK_TETRA
        print("Warning: Generated mesh contains non-tetrahedral cells.")
    
    # Skip first column (n_pts=4)
    tet_cells = grid.cells.reshape(-1, 5)[:, 1:]
    
    # Extract Triangular Surface Cells
    tri_cells_local = grid_surf.faces.reshape(-1, 4)[:, 1:]
    
    # Map surface indices to global grid indices
    if "vtkOriginalPointIds" in grid_surf.point_data:
        orig_ids = grid_surf.point_data["vtkOriginalPointIds"]
        tri_cells = orig_ids[tri_cells_local]
    else:
        print("Error: Could not map surface points to volume points.")
        return

    # 5. Tagging Surface Facets
    # Tag 2: Bottom surface (min Z)
    # Tag 1: Top/Curved surface (everything else)
    
    # Calculate cell centers for surface triangles to determine their position
    # We can use the points of the surface mesh directly.
    # A triangle is on the bottom if all its vertices are close to min Z.
    
    min_z = points[:, 2].min()
    tolerance = 1e-3 * (points[:, 2].max() - min_z) # 0.1% of height tolerance
    
    # Get Z coordinates of vertices for each triangle
    # tri_cells is (N, 3) array of point indices
    tri_z = points[tri_cells, 2] # (N, 3) array of Z coords
    
    # Check if all vertices of a triangle are close to min_z
    is_bottom = np.all(np.abs(tri_z - min_z) < tolerance, axis=1)
    
    # Create tags array
    facet_tags = np.ones(len(tri_cells), dtype=np.int32) # Default to 1 (Top)
    facet_tags[is_bottom] = 2 # Set Bottom to 2
    
    print(f"Tagged {np.sum(is_bottom)} facets as Bottom (2) and {np.sum(~is_bottom)} as Top (1).")

    # 6. Create MeshIO Meshes
    
    # Volume Mesh
    mesh_vol = meshio.Mesh(
        points=points,
        cells=[("tetra", tet_cells)]
    )
    
    # Surface Mesh (Facets)
    mesh_surf = meshio.Mesh(
        points=points,
        cells=[("triangle", tri_cells)],
        cell_data={"Grid": [facet_tags]} # "Grid" is the name used in simulation script
    )
    
    # 7. Write to XDMF
    out_path_vol = f"{output_filename}_vol.xdmf"
    out_path_surf = f"{output_filename}_surf.xdmf"
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path_vol)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    meshio.write(out_path_vol, mesh_vol)
    meshio.write(out_path_surf, mesh_surf)
    
    print(f"Volume mesh written to {out_path_vol}")
    print(f"Surface mesh written to {out_path_surf}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 2:
        generate_tetgen_mesh(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python tetgen_mesh_convert.py <input_obj> <output_base>")