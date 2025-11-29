import tetgen
import pyvista as pv
import numpy as np

print("Testing tetgen arguments...")
try:
    sphere = pv.Sphere()
    tet = tetgen.TetGen(sphere.triangulate())
    
    points = np.array([[0,0,0], [0.1,0.1,0.1]])
    
    print("Trying with points argument...")
    try:
        tet.tetrahedralize(switches="pq1.4i", points=points)
        print("Success with 'points' argument!")
    except TypeError as e:
        print(f"Failed with 'points': {e}")
    except Exception as e:
        print(f"Error with 'points': {e}")

except Exception as e:
    print(f"Setup error: {e}")
