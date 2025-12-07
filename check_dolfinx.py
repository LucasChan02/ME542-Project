import sys
print("Checking imports...")
try:
    import dolfinx
    print("dolfinx imported successfully")
except ImportError as e:
    print(f"Failed to import dolfinx: {e}")

try:
    from mpi4py import MPI
    print("mpi4py imported successfully")
except ImportError as e:
    print(f"Failed to import mpi4py: {e}")
