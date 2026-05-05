
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_movielens(ratings_path, movies_path):
    ratings = pd.read_csv(
        ratings_path, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    movies = pd.read_csv(
        movies_path, sep="|", encoding="latin-1",
        usecols=[0, 1], names=["item_id", "title"],
        engine="python",
    )
    return ratings, movies


def build_user_item_matrix(ratings):
    return ratings.pivot_table(
        index="user_id", columns="item_id",
        values="rating", fill_value=0,
    )


def compute_item_similarity(user_item_matrix):
    item_matrix = user_item_matrix.T.values
    sim_matrix = cosine_similarity(item_matrix)
    item_ids = user_item_matrix.columns
    return pd.DataFrame(sim_matrix, index=item_ids, columns=item_ids)


def predict_score(user_id, item_id, user_item_matrix, item_similarity, k=20):
    if user_id not in user_item_matrix.index:
        return 0.0
    if item_id not in item_similarity.index:
        return 0.0
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    rated_items = [i for i in rated_items if i in item_similarity.index]
    if not rated_items:
        return 0.0
    sims = item_similarity.loc[item_id, rated_items]
    top_k_sims = sims[sims > 0].nlargest(k)
    if top_k_sims.empty:
        return 0.0
    numerator   = sum(top_k_sims[j] * user_ratings[j] for j in top_k_sims.index)
    denominator = top_k_sims.abs().sum()
    return round(numerator / denominator, 4) if denominator else 0.0


def get_top_n_recommendations(user_id, user_item_matrix, item_similarity, movies, n=10, k=20):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=["item_id", "title", "predicted_score"])
    user_ratings  = user_item_matrix.loc[user_id]
    unseen_items  = user_ratings[user_ratings == 0].index.tolist()
    predictions   = []
    for item_id in unseen_items:
        score = predict_score(user_id, item_id, user_item_matrix, item_similarity, k=k)
        if score > 0:
            predictions.append({"item_id": item_id, "predicted_score": score})
    if not predictions:
        return pd.DataFrame(columns=["item_id", "title", "predicted_score"])
    results = pd.DataFrame(predictions)
    results = results.sort_values("predicted_score", ascending=False).head(n)
    results = results.merge(movies, on="item_id", how="left")
    results = results[["item_id", "title", "predicted_score"]].reset_index(drop=True)
    results.index += 1
    return results


def get_similar_items(item_id, item_similarity, movies, n=10):
    if item_id not in item_similarity.index:
        return pd.DataFrame()
    sims = item_similarity.loc[item_id].drop(index=item_id)
    top  = sims.nlargest(n).reset_index()
    top.columns = ["item_id", "similarity"]
    top = top.merge(movies, on="item_id", how="left")
    top = top[["item_id", "title", "similarity"]].reset_index(drop=True)
    top.index += 1
    return top


def evaluate_rmse(test_ratings, user_item_matrix, item_similarity, k=20, sample_size=500):
    sample = test_ratings.sample(min(sample_size, len(test_ratings)), random_state=42)
    errors = []
    for _, row in sample.iterrows():
        pred = predict_score(int(row["user_id"]), int(row["item_id"]),
                             user_item_matrix, item_similarity, k=k)
        if pred > 0:
            errors.append((pred - row["rating"]) ** 2)
    return round(np.sqrt(np.mean(errors)), 4) if errors else float("nan")
