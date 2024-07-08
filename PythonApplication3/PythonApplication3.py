import numpy as np

def custom_svd(A):
    # Step 1: Compute A^T A and A A^T
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    eigval_AAT, eigvect_AAT = np.linalg.eig(np.dot(A, A.T))
    U = eigvect_AAT[:, np.argsort(eigval_AAT)[::-1]]

    eigval_ATA, eigvect_ATA = np.linalg.eig(np.dot(A.T, A))
    V = eigvect_ATA[:, np.argsort(eigval_ATA)[::-1]]

    singular_values = np.sqrt(np.maximum(eigval_ATA, 0))
    Σ = np.zeros(A.shape)
    Σ[:min(A.shape), :min(A.shape)] = np.diag(singular_values)

    for i in range(len(singular_values)):
        U[:, i] = np.dot(A, V[:, i]) / singular_values[i]

    return U, Σ, V

# Example usage
A = np.array([[25, 42], [37, 41]])
U, Σ, V = custom_svd(A)

# Verify the decomposition
A_reconstructed = np.dot(U, np.dot(Σ, V.T)).round(1)

print("Original matrix A:\n", A)
print("U matrix:\n", np.round(U, decimals=1))
print("Σ matrix:\n", np.round(Σ, decimals=1))
print("V^T matrix:\n", np.round(V.T, decimals=1))
print("Reconstructed matrix: \n", A_reconstructed)


# Checking if the reconstruction is close
print("Reconstruction is close to the original: ", np.allclose(A, A_reconstructed))
