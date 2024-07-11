import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Зчитування CSV файлу
ratings_path = pd.read_csv('ratings.csv')
movies_path = pd.read_csv('movies.csv')

ratings_matrix = ratings_path.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix1 = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix2 = ratings_matrix.dropna(thresh=100, axis=1)


def svd_ratings(ratings_matrix, k):
    ratings_matrix_filled = ratings_matrix.fillna(2.5)
    R = ratings_matrix_filled.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=k)
    return U, sigma, Vt, user_ratings_mean

U, sigma, Vt, user_ratings_mean = svd_ratings(ratings_matrix2, 3)

# Візуалізація користувачів у тривимірному просторі
def plot_users(U):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(U[:, 0], U[:, 1], U[:, 2], c='r', marker='o')
    plt.title("Users")
    plt.show()

# Візуалізація фільмів у тривимірному просторі
def plot_movies(Vt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Vt[0, :], Vt[1, :], Vt[2, :], c='b', marker='^')
    plt.title("Movies")
    plt.show()

# Показуємо перші 20 користувачів та фільмів
plot_users(U[:20, :])
plot_movies(Vt[:, :20])


# 2 part
def perform_pred(ratings_matrix, k):
    U, sigma, Vt, user_ratings_mean = svd_ratings(ratings_matrix, k)
    all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean.reshape(-1, 1)    
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
    return preds_df


ratings_matrix3 = ratings_matrix.dropna(thresh=20, axis=0)
ratings_matrix4 = ratings_matrix.dropna(thresh=20, axis=1)
U1, sigma1, Vt1, user_ratings_mean1 = svd_ratings(ratings_matrix4, 3)

K = [3, 5, 10, 25, 50, 100]
errors = []
# варіації для розних к
for k in K:
    preds = perform_pred(ratings_matrix, k)
    start_ratings = ratings_matrix.values
    predict_ratings = preds.values
    mask = ~np.isnan(start_ratings)
    mse = mean_squared_error(start_ratings[mask], predict_ratings[mask])
    errors.append(mse)
    print(f'k = {k}, Mean Squared Error: {mse}')
    
# побудова графіку
plt.plot(K, errors, marker='o')
plt.xlabel('k (Number of latent factors)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of k on SVD Prediction Accuracy')
plt.show()

optimal_k = K[np.argmin(errors)]
final_preds_df = perform_pred(ratings_matrix4, optimal_k)
preds_only_df = final_preds_df.mask(~ratings_matrix.isna())
print("Predicted ratings generally:")
print(final_preds_df)
print("Predicted ratings only:")
print(preds_only_df)

# таблиця рекомендацій
def recom_movies_for_user(preds_only_df, user_id, num_recommend=10):
    user_row = preds_only_df.loc[user_id].dropna().sort_values(ascending=False).head(num_recommend)
    titles = movies_path.set_index('movieId').loc[user_row.index]
    recommend_table = pd.DataFrame({
        'Movie Title': titles['title'],
        'Genres': titles['genres']
    })
    return recommend_table

user_id = 18
recommendations = recom_movies_for_user(preds_only_df, user_id)
print(f"Top 10 movies recommend for user {user_id}:")
print(recommendations)
