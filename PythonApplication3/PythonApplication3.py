import numpy as np

def custom_svd(A):
    m, n = A.shape

    # Compute A^T A and A A^T
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    # A^T A and A A^T eidvects and eigvals
    eigval_AAT, eigvect_AAT = np.linalg.eig(AAT)
    eigval_ATA, eigvect_ATA = np.linalg.eig(ATA)

    # Sort eigenvalues and eigenvectors
    sorted_indices_AAT = np.argsort(eigval_AAT)[::-1]
    sorted_indices_ATA = np.argsort(eigval_ATA)[::-1]

    eigval_AAT = eigval_AAT[sorted_indices_AAT]
    eigvect_AAT = eigvect_AAT[:, sorted_indices_AAT]

    eigval_ATA = eigval_ATA[sorted_indices_ATA]
    eigvect_ATA = eigvect_ATA[:, sorted_indices_ATA]

    # Form the Σ
    singular_values = np.sqrt(np.maximum(eigval_ATA, 0))
    min_dim = min(m, n)
    Σ = np.zeros((min_dim, min_dim))
    np.fill_diagonal(Σ, singular_values[:min_dim])

    U = eigvect_AAT[:, :min_dim]
    V = eigvect_ATA[:, :min_dim]

    for i in range(min_dim):
        if singular_values[i] > 1e-10:  # avoid division by zero for very small singular values
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]

    return U, Σ, V.T


A = np.array([[25, 42, 12], [37, 41, 69]])
U, Σ, V_T = custom_svd(A)

A_reconstructed = np.dot(U, np.dot(Σ, V_T)).round(1)

print("Original matrix A:\n", A)
print("U matrix:\n", np.round(U, decimals=1))
print("Σ matrix:\n", np.round(Σ, decimals=1))
print("V^T matrix:\n", np.round(V_T, decimals=1))
print("Reconstructed matrix: \n", A_reconstructed)

print("Reconstruction is close to the original: ", np.allclose(A, A_reconstructed))
