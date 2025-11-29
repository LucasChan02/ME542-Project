import pyvista as pv
import numpy as np
import os

def generate_initial_points(mesh_path, num_vertices, inward_offset=0.6):
    """
    Generates random vertices inside a given volume mesh with an inward offset constraint.
    
    Args:
        mesh_path (str): Path to the .obj or .stl file.
        num_vertices (int): Number of points to generate.
        inward_offset (float): Minimum distance from the surface (inwards). Default is 6.0 mm.
        
    Returns:
        np.ndarray: Array of shape (num_vertices, 3) containing the points.
    """
    save_coordinates_list = False  
    # Handle .step extension if passed by mistake, preferring .obj if it exists
    if mesh_path.lower().endswith('.step') or mesh_path.lower().endswith('.stp'):
        base = os.path.splitext(mesh_path)[0]
        obj_candidate = base + ".obj"
        if os.path.exists(obj_candidate):
            
            print(f"Warning: '{mesh_path}' is a STEP file. Using '{obj_candidate}' for point generation instead.")
            mesh_path = obj_candidate
        else:
            # Fallback: If no OBJ exists, we can't easily proceed with PyVista alone for a STEP file
            # without converting it first.
            raise ValueError(f"Cannot generate points from STEP file '{mesh_path}' directly with PyVista. Please provide a surface mesh (.obj, .stl) or ensure '{obj_candidate}' exists.")

    mesh = pv.read(mesh_path)
    
    # Ensure mesh is triangulated
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
        
    bounds = mesh.bounds
    # bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    internal_points = np.empty((0, 3))
    
    # Generate points in batches until we have enough
    # We generate more than needed because some will be outside or within the offset
    batch_size = max(num_vertices * 5, 1000)
    
    print(f"Generating {num_vertices} internal points from {mesh_path} with inward offset {inward_offset}...")
    
    while len(internal_points) < num_vertices:
        # Generate random points within bounding box
        pts = np.random.rand(batch_size, 3)
        pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
        pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min
        pts[:, 2] = pts[:, 2] * (z_max - z_min) + z_min
        
        # Convert to PolyData for PyVista check
        pts_poly = pv.PolyData(pts)
        
        # Compute implicit distance from points to the mesh
        # Positive values are outside, negative values are inside.
        # We want points that are inside and at least 'inward_offset' away from the surface.
        # So distance <= -inward_offset
        target = pts_poly.compute_implicit_distance(mesh)
        distances = target['implicit_distance']
        
        mask = distances <= -inward_offset
        
        inside_pts = pts[mask]
        
        if len(inside_pts) > 0:
            internal_points = np.vstack((internal_points, inside_pts))
            
        print(f"  Batch generated {len(inside_pts)} valid points. Total: {len(internal_points)}/{num_vertices}")
            
    result = internal_points[:num_vertices]
    print(f"{len(result)} Initial points generated.")
    
    if save_coordinates_list:
        # Save to CSV for debugging
        csv_path = os.path.join(os.path.dirname(mesh_path), "internal_points.csv")
        np.savetxt(csv_path, result, delimiter=",", header="x,y,z", comments="")
    
    return result
