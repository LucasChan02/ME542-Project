
import sys
import os
import numpy as np
import meshio
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot, grad, sym_grad, trace, eye, identity, ddot

def run_simulation(mesh_path):
    # --- 1. Load Mesh ---
    try:
        mesh = MeshTet.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return

    # --- 2. Material Properties (BASF Ultrafuse TPU 95A) ---
    E = 26.0e6      # Young's Modulus (Pa)
    nu = 0.40       # Poisson's ratio
    lam, mu = lame_parameters(E, nu)

    # --- 3. Identify Boundaries ---
    # Bottom: Fixed (Dirichlet)
    # Top: Traction (Neumann)
    p = mesh.p
    z = p[2, :]
    z_min = z.min()
    height = z.max() - z_min
    tol = 1e-3 * height
    
    # --- 4. FEM Definition ---
    e = ElementVector(ElementTetP1())
    basis = Basis(mesh, e, intorder=1)
    
    # Manual stiffness definition
    # @BilinearForm
    # def stiffness(u, v, w):
    #     def C(e):
    #         return 2.0 * mu * e + lam * trace(e) * eye(e, 3)
    #     return ddot(C(sym_grad(u)), sym_grad(v))
    
    # Use built-in linear_elasticity
    stiffness = linear_elasticity(lam, mu)

    traction_mag = -1.0e5
    @LinearForm
    def traction(v, w):
        return traction_mag * v[2]

    # Identify facets for BCs
    bottom_facets = mesh.facets_satisfying(lambda x: x[2] < z_min + tol)
    all_boundary_facets = mesh.boundary_facets()
    top_facets = np.setdiff1d(all_boundary_facets, bottom_facets)
    
    # Boundary Basis for Traction
    basis_top = BoundaryFacetBasis(mesh, e, facets=top_facets, intorder=1)
    
    # --- 5. Assembly and Solve ---
    K = asm(stiffness, basis)
    f = traction.assemble(basis_top)
    
    # Apply Dirichlet BCs (Bottom fixed)
    bottom_dofs = basis.get_dofs(lambda x: x[2] < z_min + tol).flatten()
    
    # Solve system
    x = solve(*condense(K, f, D=bottom_dofs))
    
    # --- 6. Post-Processing: Von Mises Stress ---
    # Calculate stress at element centroids (DG0)
    basis_centroid = Basis(mesh, e, intorder=0)
    
    # Interpolate solution and calculate gradients
    out = basis_centroid.interpolate(x)
    du = out.grad
    print(f"Gradient shape: {du.shape}")
    
    # Symetric gradient (Strain)
    # eps = 0.5 * (du + du.T)
    # du is (3, 3, n_elems) after squeezing
    # Gradient shape: (3, 3, n_elems, n_quad)
    # e.g. (3, 3, 3317, 1)
    
    if du.ndim == 4:
        if du.shape[3] == 1:
            du = du.squeeze(3)
        else:
            du = np.mean(du, axis=3)
            
    if du.ndim != 3:
        print(f"Unexpected gradient shape after processing: {du.shape}")
        
    # Calculate Strain (Symmetric Gradient)
    eps = 0.5 * (du + np.transpose(du, (1, 0, 2)))
    
    # Calculate Stress (Sigma) using Hooke's Law
    tr_eps = eps[0, 0, :] + eps[1, 1, :] + eps[2, 2, :]
    sigma = 2.0 * mu * eps
    sigma[0, 0, :] += lam * tr_eps
    sigma[1, 1, :] += lam * tr_eps
    sigma[2, 2, :] += lam * tr_eps
    
    # Calculate Von Mises Stress
    tr_sigma = sigma[0, 0, :] + sigma[1, 1, :] + sigma[2, 2, :]
    dev = sigma.copy()
    dev[0, 0, :] -= tr_sigma / 3.0
    dev[1, 1, :] -= tr_sigma / 3.0
    dev[2, 2, :] -= tr_sigma / 3.0
    
    dev_dot_dev = np.sum(dev * dev, axis=(0, 1))
    vm_stress = np.sqrt(1.5 * dev_dot_dev)
    
    max_vm = np.max(vm_stress)
    
    print(f"Maximum von Mises stress: {max_vm:.2e} Pa")
    print(f"MAX_STRESS: {max_vm}")
    
    # --- 7. Save Results ---
    # Save displacement and stress to XDMF for visualization
    print(f"vm_stress shape: {vm_stress.shape}")
    
    base = os.path.splitext(mesh_path)[0]
    out_path = f"{base}_result.xdmf"
    
    # Map solution vector 'x' to nodal displacements (N, 3)
    u_vec = np.zeros((mesh.p.shape[1], 3))
    u_vec[:, 0] = x[basis.nodal_dofs[0]]
    u_vec[:, 1] = x[basis.nodal_dofs[1]]
    u_vec[:, 2] = x[basis.nodal_dofs[2]]
    
    meshio.write(
        out_path,
        meshio.Mesh(
            points=mesh.p.T,
            cells={"tetra": mesh.t.T},
            point_data={"displacement": u_vec},
            cell_data={"von_mises": [vm_stress]}
        )
    )
    print(f"Result written to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_simulation(sys.argv[1])
    else:
        print("Usage: python simulation_skfem.py <mesh_vol.xdmf>")
