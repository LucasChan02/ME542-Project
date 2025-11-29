import sys
import argparse
# from gmsh_mesh_convert import generate_gmsh_fenics
from tetgen_mesh_convert import generate_tetgen_mesh
from initial_points import generate_initial_points

import os

def main():
    parser = argparse.ArgumentParser(description="Test mesh generation methods.")
    parser.add_argument("method", choices=["gmsh", "tetgen"], help="Mesh generation method to use")
    args = parser.parse_args()

    obj_path = "scan2_volume_v7.obj"
    
    print(f"Generating initial vertices from {obj_path}...")
    # Generate points using the OBJ mesh
    internal_points = generate_initial_points(obj_path, 100, inward_offset=0.6)

    # if args.method == "gmsh":
    #     print("Running Gmsh generation...")
    #     gmsh_mesh = generate_gmsh_fenics(obj_path, internal_points)
    #     print("Gmsh generation complete.")

    if args.method == "tetgen":
        print("Running TetGen generation...")
        # Construct output filename from input filename
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        output_base = os.path.join("gen_mesh", f"{base_name}_tetgen")
        
        generate_tetgen_mesh(obj_path, output_base, internal_points=internal_points)
        print("TetGen generation complete.")

if __name__ == "__main__":
    main()

