print("Starting mpi4py check...")
try:
    from mpi4py import MPI
    print("mpi4py imported successfully")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"MPI Rank: {rank}")
except Exception as e:
    print(f"Error importing mpi4py: {e}")
print("Finished mpi4py check.")
