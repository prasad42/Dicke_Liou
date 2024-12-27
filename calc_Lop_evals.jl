using SparseArrays
using MatrixMarket
using Arpack

# Load the sparse matrix from the Matrix Market file
sparse_matrix = MatrixMarket.mmread(f"evals_par_Lop/evals_j={j}_M={M}_ω={ω}_ω0={ω0}_gc={np.round(np.sqrt(ω/ω0*(γ**2+ω**2))/2,2)}_γ={γ}_g={g}.mtx")

# Ensure it is a SparseMatrixCSC (compressed sparse column format)
sparse_matrix = SparseMatrixCSC(sparse_matrix)

# Compute only eigenvalues using Arpack's `eigs` function
eigenvalues = eigs(sparse_matrix; nev=3, which=:LM, vec=false)  # `vec=false` skips eigenvectors

println("Computed eigenvalues: ", eigenvalues.values)