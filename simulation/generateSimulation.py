from mpi4py import MPI
import numpy as np
import ufl

from dolfinx import fem, io
from dolfinx.io import gmsh as gmshio
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

comm = MPI.COMM_WORLD

import os
#reading mesh file
script_dir = os.path.dirname(os.path.abspath(__file__))
gmsh_file = os.path.join(script_dir, "cube_with_void.msh")

meshdata = gmshio.read_from_msh(
    gmsh_file,
    comm=comm,
    rank=0,   # root rank that reads the file
    gdim=3,
)

msh = meshdata.mesh
cell_tags = meshdata.cell_tags
facet_tags = meshdata.facet_tags

gdim = msh.geometry.dim


V = functionspace(msh, ("Lagrange", 1, (gdim,)))

v = ufl.TestFunction(V)      # test function

#Material properties
E = 5.0e7       # Young's Modulus - Pa
nu = 0.30       # Poisson's ratio
mu = E / (2.0 * (1.0 + nu))    # Shear Modulus - Pa
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))    # Lame's first parameter - Pa

def eps(w):
    return ufl.sym(ufl.grad(w))

def sigma(w):
    return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim)

#Boundaryy conditions
top_tag = 1  #Foot contour
bottom_tag = 2  #Bottom flat surface

# Bottom fixed (u = 0)
bottom_facets = facet_tags.find(bottom_tag)
bottom_dofs = locate_dofs_topological(
    V, entity_dim=msh.topology.dim - 1, entities=bottom_facets
)

# Zero displacement 
zero_disp = np.zeros(gdim, dtype=np.float64)
bc_bottom = dirichletbc(zero_disp, bottom_dofs, V)


# We apply uniform traction in -z direction
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

uh = problem.solve()
uh.name = "displacement"


sig = sigma(uh)
dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(gdim)
sigma_vm = ufl.sqrt(1.5 * ufl.inner(dev, dev))

W = functionspace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_h = Function(W)
sigma_vm_h.name = "von_mises"

sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points)
sigma_vm_h.interpolate(sigma_vm_expr)

#MAximum von mises
max_vm_local = sigma_vm_h.x.array.max()
max_vm_global = comm.allreduce(max_vm_local, op=MPI.MAX)

if comm.rank == 0:
    print(f"Maximum von Mises stress: {max_vm_global:.2e} Pa")

#Output (Use software like paraview to open the created folder)
from dolfinx.io import VTXWriter

# Write displacement
with VTXWriter(comm, "results_displacement.bp", [uh]) as vtx:
    vtx.write(0.0)

# Write von Mises stress
with VTXWriter(comm, "results_vonmises.bp", [sigma_vm_h]) as vtx:
    vtx.write(0.0)

if comm.rank == 0:
    print("DONE open results_displacement.bp and results_vonmises.bp in ParaView")