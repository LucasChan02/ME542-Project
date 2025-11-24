import gmsh
import math
import h5py
from mpi4py import MPI
from dolfinx.io import gmshio

def generate_gmsh_fenics(obj_path, internal_points):
    """
    Generates a distributed FEniCSx mesh from OBJ with embedded points.
    """
    gmsh.initialize()
    gmsh.model.add("VolumeMesh")

    # 1. Merge OBJ and Reparametrize
    gmsh.merge(obj_path)
    # Angle=40 deg for sharp edges. forceParametrizable=True for FEA.
    gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, True, True)
    gmsh.model.mesh.createGeometry()

    # 2. Create Volume
    surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    sl = gmsh.model.geo.addSurfaceLoop(surfaces)
    vol = gmsh.model.geo.addVolume([sl])

    # 3. Embed Internal Points
    p_tags = []
    for pt in internal_points:
        # addPoint(x, y, z, mesh_size)
        t = gmsh.model.geo.addPoint(pt, pt[1], pt[2], 0.05) 
        p_tags.append(t)

    gmsh.model.geo.synchronize()
    
    # Embed points (dim=0) into volume (dim=3)
    gmsh.model.mesh.embed(0, p_tags, 3, vol)

    # 4. Add Physical Groups (Mandatory for FEniCSx)
    # FEniCSx needs a Physical Volume to read the mesh cells
    gmsh.model.addPhysicalGroup(3, [vol], tag=1, name="MainVolume")

    # 5. Generate Mesh
    gmsh.model.mesh.generate(3)

    # 6. Import directly into FEniCSx
    # partitioner=None uses default METIS/SCOTCH
    dolfin_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=3
    )
    
    gmsh.finalize()
    return dolfin_mesh