import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# ���������� CSV �����
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

# ��������� ������� ��������
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Գ�������� ������������, �� ������� ����� 200 ������, �� ������ � ����� 100 ��������
ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

# ���������� ������� ������� ������� ��������� (2.5)
ratings_matrix_filled = ratings_matrix.fillna(2.5)

# ������������ �� ����� NumPy
R = ratings_matrix_filled.values

# �������� ������������ ���������� ������ ������������
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# ��������� SVD
U, sigma, Vt = svds(R_demeaned, k=3)

# ³��������� ������������ � ����������� �������
def plot_users(U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U[:, 0], U[:, 1], U[:, 2], c='r', marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title("Users")
    plt.show()

# ³��������� ������ � ����������� �������
def plot_movies(Vt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Vt[0, :], Vt[1, :], Vt[2, :], c='b', marker='^')
    plt.title("Movies")
    plt.show()

# �������� ����� 20 ������������ �� ������
plot_users(U[:20, :])
plot_movies(Vt[:, :20])

# ³��������� ��������� �������
sigma = np.diag(sigma)
R_approx = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# �������� ��������� ������� � ����������� (��������� ������)
print("Original Matrix (limited):\n", ratings_matrix.values)
print("Approximated Matrix:\n", R_approx)

# �������� �� ��������� ��������� ������� �� ����������
print("Reconstruction is close to the original: ", np.allclose(ratings_matrix.values, R_approx, atol=1))
