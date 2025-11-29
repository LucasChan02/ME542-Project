import tetgen
import pyvista as pv
import numpy as np

print("Inspecting tetgen.TetGen...")
try:
    # Create a dummy surface
    sphere = pv.Sphere()
    tet = tetgen.TetGen(sphere.triangulate())
    print("Attributes/Methods of tet object:")
    print(dir(tet))
    
    print("\nHelp on tetrahedralize:")
    print(help(tet.tetrahedralize))
except Exception as e:
    print(e)
