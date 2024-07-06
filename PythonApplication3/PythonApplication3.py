import numpy as np

def custom_svd(A):
    # Step 1: Compute A^T A and A A^T
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    # Step 2: Eigen decomposition of A^T A and A A^T
    eigvals_V, eigvecs_V = np.linalg.eig(ATA)
    eigvals_U, eigvecs_U = np.linalg.eig(AAT)

    # Filter out negative eigenvalues and small numerical errors
    eigvals_V = np.clip(eigvals_V, 0, None)
    eigvals_U = np.clip(eigvals_U, 0, None)

    # Sort eigenvalues and eigenvectors
    sorted_indices_V = np.argsort(eigvals_V)[::-1]
    sorted_indices_U = np.argsort(eigvals_U)[::-1]

    eigvals_V = eigvals_V[sorted_indices_V]
    eigvecs_V = eigvecs_V[:, sorted_indices_V]

    eigvals_U = eigvals_U[sorted_indices_U]
    eigvecs_U = eigvecs_U[:, sorted_indices_U]

    # Step 3: Form the Σ matrix
    singular_values = np.sqrt(eigvals_U)
    Σ = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    np.fill_diagonal(Σ, singular_values[:min(A.shape)])


    # Step 4: Form the U and V^T matrices
    U = eigvecs_U
    V_T = eigvecs_V.T

    return U, Σ, V_T

# Example usage
A = np.array([[1, 2], [3, 4]], dtype=float)
U, Σ, V_T = custom_svd(A)

# Verify the decomposition
A_reconstructed = np.dot(U, np.dot(Σ, V_T))

print("Original matrix A:\n", A)
print("Reconstructed matrix A:\n", A_reconstructed)
print("U matrix:\n", U)
print("Σ matrix:\n", Σ)
print("V^T matrix:\n", V_T)

# Checking if the reconstruction is close to the original matrix
print("Reconstruction is close to the original: ", np.allclose(A, A_reconstructed))
