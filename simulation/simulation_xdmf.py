from mpi4py import MPI
import numpy as np
import ufl
import os
import sys

from dolfinx import fem, io
from dolfinx.io import XDMFFile
from dolfinx.fem import (
    functionspace,
    Function,
    Constant,
    Expression,
    dirichletbc,
    locate_dofs_topological,
    form,
)
from dolfinx.fem.petsc import LinearProblem

def run_simulation(mesh_path):
    # Derive facets path from mesh path
    # Assuming mesh is "name_vol.xdmf" and facets is "name_surf.xdmf"
    base_name = mesh_path.replace("_vol.xdmf", "")
    facets_path = f"{base_name}_surf.xdmf"
    
    comm = MPI.COMM_WORLD
    
    # Check if files exist
    if not os.path.exists(mesh_path):
        if comm.rank == 0:
            print(f"Error: Mesh file '{mesh_path}' not found.")
        return None
    
    if not os.path.exists(facets_path):
        if comm.rank == 0:
            print(f"Error: Facets file '{facets_path}' not found.")
        return None

    # Reading mesh and tags from XDMF
    if comm.rank == 0:
        print(f"Reading mesh from {mesh_path}...")
        
    with XDMFFile(comm, mesh_path, "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        # cell_tags = xdmf.read_meshtags(msh, name="Grid") # Optional if we need volume tags
        
    # Create connectivity between facets and cells
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    if comm.rank == 0:
        print(f"Reading facets from {facets_path}...")
        
    with XDMFFile(comm, facets_path, "r") as xdmf:
        # This lines up the surface tags with the existing 3D mesh
        facet_tags = xdmf.read_meshtags(msh, name="Grid")

    gdim = msh.geometry.dim

    # Function Space
    V = functionspace(msh, ("Lagrange", 1, (gdim,)))
    v = ufl.TestFunction(V)      # test function

    # Material properties
    E = 44.2e6       # Young's Modulus - Pa
    nu = 0.38       # Poisson's ratio
    mu = E / (2.0 * (1.0 + nu))    # Shear Modulus - Pa
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))    # Lame's first parameter - Pa

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim)

    # Boundary conditions
    top_tag = 1  # Top curved surface (Foot contour)
    bottom_tag = 2  # Bottom flat surface

    # Bottom fixed (u = 0)
    # Ensure tags exist
    if len(facet_tags.find(bottom_tag)) == 0:
        if comm.rank == 0:
            print(f"Warning: No facets found with bottom_tag={bottom_tag}")
            
    bottom_facets = facet_tags.find(bottom_tag)
    bottom_dofs = locate_dofs_topological(
        V, entity_dim=msh.topology.dim - 1, entities=bottom_facets
    )

    # Zero displacement 
    zero_disp = np.zeros(gdim, dtype=np.float64)
    bc_bottom = dirichletbc(zero_disp, bottom_dofs, V)

    # We apply uniform traction in -z direction on the top surface
    traction_mag = -1.0e5  # Pa
    t = Constant(msh, np.array([0.0, 0.0, traction_mag], dtype=np.float64))

    dx = ufl.Measure("dx", domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    uh = ufl.TrialFunction(V)  # trial displacement for bilinear form

    # Using eps for symmetric weak form
    a = ufl.inner(sigma(uh), eps(v)) * dx
    L = ufl.inner(t, v) * ds(top_tag)

    # Linear problem wrapper
    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "gamg",
    }

    problem = LinearProblem(
        a,
        L,
        bcs=[bc_bottom],
        petsc_options=petsc_opts,
        petsc_options_prefix="elasticity_"  
    )

    if comm.rank == 0:
        print("Solving linear problem...")
        
    uh = problem.solve()
    uh.name = "displacement"

    # Post-processing: von Mises stress
    sig = sigma(uh)
    dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(gdim)
    sigma_vm = ufl.sqrt(1.5 * ufl.inner(dev, dev))

    W = functionspace(msh, ("Discontinuous Lagrange", 0))
    sigma_vm_h = Function(W)
    sigma_vm_h.name = "von_mises"

    sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points)
    sigma_vm_h.interpolate(sigma_vm_expr)

    # Maximum von mises
    max_vm_local = sigma_vm_h.x.array.max()
    max_vm_global = comm.allreduce(max_vm_local, op=MPI.MAX)

    if comm.rank == 0:
        print(f"Maximum von Mises stress: {max_vm_global:.2e} Pa")
        print(f"MAX_STRESS: {max_vm_global}")

    # Output (Use software like paraview to open the created folder)
    from dolfinx.io import VTXWriter

    # Determine output paths
    # Use the mesh filename as the base for the results
    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
    
    # Create output directory
    results_dir = os.path.dirname(mesh_path) # Save in same dir as mesh
        
    out_disp = os.path.join(results_dir, f"{mesh_basename}_displacement.bp")
    out_vm = os.path.join(results_dir, f"{mesh_basename}_vonmises.bp")

    # Write displacement
    with VTXWriter(comm, out_disp, [uh]) as vtx:
        vtx.write(0.0)

    # Write von Mises stress
    with VTXWriter(comm, out_vm, [sigma_vm_h]) as vtx:
        vtx.write(0.0)

    if comm.rank == 0:
        print(f"DONE open {out_disp} and {out_vm} in ParaView")
        
    return max_vm_global

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_simulation(sys.argv[1])
    else:
        print("Usage: python simulation_xdmf.py <mesh_vol.xdmf>")
