import sys
import argparse
# from gmsh_mesh_convert import generate_gmsh_fenics
from tetgen_mesh_convert import generate_tetgen_mesh
from initial_vertices import generate_initial_vertices

def main():
    parser = argparse.ArgumentParser(description="Test mesh generation methods.")
    parser.add_argument("method", choices=["gmsh", "tetgen"], help="Mesh generation method to use")
    args = parser.parse_args()

    obj_path = "scan2_volume_v7.obj"
    
    print(f"Generating initial vertices from {obj_path}...")
    # Generate points using the OBJ mesh
    internal_points = generate_initial_vertices(obj_path, 100)

    # if args.method == "gmsh":
    #     print("Running Gmsh generation...")
    #     gmsh_mesh = generate_gmsh_fenics(obj_path, internal_points)
    #     print("Gmsh generation complete.")

    if args.method == "tetgen":
        print("Running TetGen generation...")
        generate_tetgen_mesh(obj_path, "gen_mesh/tetgen_output")
        print("TetGen generation complete.")

if __name__ == "__main__":
    main()

