"""
Core recommendation logic — identical to the original notebook (finall.ipynb).
Kept free of any Streamlit imports so it can be unit-tested independently.
"""

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

RATING_THRESHOLD = 4

AGE_MAP = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}

OCCUPATION_MAP = {
    0: "other / not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
    19: "unemployed", 20: "writer",
}


def load_data(data_dir="data"):
    movies = pd.read_csv(
        f"{data_dir}/movies.dat", sep="::", engine="python",
        names=["MovieID", "Title", "Genres"], encoding="latin-1",
    )
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat", sep="::", engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin-1",
    )
    users = pd.read_csv(
        f"{data_dir}/users.dat", sep="::", engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="latin-1",
    )
    ratings["DateTime"] = pd.to_datetime(ratings["Timestamp"], unit="s")
    return movies, ratings, users


def train_apriori_model(movies, ratings, min_support=0.05, min_lift=1.2):
    positive_ratings = ratings[ratings["Rating"] >= RATING_THRESHOLD]
    ratings_with_titles = pd.merge(positive_ratings, movies, on="MovieID")
    transactions = ratings_with_titles.groupby("UserID")["Title"].apply(list).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=2)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules, len(transactions), df_encoded.shape[1]


def build_rule_index(rules_df):
    index = defaultdict(list)
    for _, row in rules_df.iterrows():
        antecedents = frozenset(row["antecedents"])
        index[antecedents].append((tuple(row["consequents"]), float(row["lift"])))
    return index


def get_top_n_recommendations(rules_df, user_id, ratings_df, movies_df, n=5):
    user_history = ratings_df[(ratings_df["UserID"] == user_id) & (ratings_df["Rating"] >= RATING_THRESHOLD)]
    liked_titles = set(pd.merge(user_history, movies_df, on="MovieID")["Title"])

    if not liked_titles:
        return [], liked_titles

    recommendation_map = defaultdict(float)
    for _, row in rules_df.iterrows():
        antecedents = set(row["antecedents"])
        if antecedents.issubset(liked_titles):
            for candidate in row["consequents"]:
                if candidate not in liked_titles:
                    recommendation_map[candidate] += row["lift"]

    sorted_predictions = sorted(recommendation_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictions[:n], liked_titles


def cold_start_recommendations(rules_df, selected_titles, n=5):
    selected_titles = set(selected_titles)
    recommendation_map = defaultdict(float)
    for _, row in rules_df.iterrows():
        antecedents = set(row["antecedents"])
        if antecedents.issubset(selected_titles):
            for candidate in row["consequents"]:
                if candidate not in selected_titles:
                    recommendation_map[candidate] += row["lift"]
    return sorted(recommendation_map.items(), key=lambda x: x[1], reverse=True)[:n]


def make_train_test_split(ratings_df, test_size_ratio=0.2, seed=42):
    np.random.seed(seed)
    test_indices = []
    for user_id, group in ratings_df.groupby("UserID"):
        n_test = max(1, int(len(group) * test_size_ratio))
        if len(group) > n_test:
            test_idx = group.sample(n=n_test, random_state=seed).index
            test_indices.extend(test_idx)
    test_ratings = ratings_df.loc[test_indices].copy()
    train_ratings = ratings_df.drop(test_indices).copy()
    return train_ratings, test_ratings


def evaluate_apriori_recommender(rule_index, train_ratings_df, test_ratings_df, movies_df,
                                  sample_size=300, k=5, progress_cb=None):
    test_users = list(test_ratings_df["UserID"].unique())
    if sample_size:
        test_users = test_users[:sample_size]

    results = []
    total = len(test_users)
    for idx, user_id in enumerate(test_users, 1):
        if progress_cb and total and idx % max(1, total // 20) == 0:
            progress_cb(idx / total)

        user_test = test_ratings_df[test_ratings_df["UserID"] == user_id]
        ground_truth_movies = set(user_test[user_test["Rating"] >= RATING_THRESHOLD]["MovieID"].tolist())
        if len(ground_truth_movies) == 0:
            continue

        user_history = train_ratings_df[
            (train_ratings_df["UserID"] == user_id) & (train_ratings_df["Rating"] >= RATING_THRESHOLD)
        ]
        liked_titles = set(pd.merge(user_history, movies_df, on="MovieID")["Title"])

        recommendation_map = defaultdict(float)
        candidate_antecedents = [frozenset([title]) for title in liked_titles]
        if len(liked_titles) > 1:
            candidate_antecedents += [frozenset(pair) for pair in itertools.combinations(liked_titles, 2)]
        for antecedent in candidate_antecedents:
            for consequents, lift in rule_index.get(antecedent, []):
                for candidate in consequents:
                    if candidate not in liked_titles:
                        recommendation_map[candidate] += lift

        sorted_predictions = sorted(recommendation_map.items(), key=lambda x: x[1], reverse=True)[:k]
        recommended_titles = set(title for title, _ in sorted_predictions)

        recommended_movies = set()
        for title in recommended_titles:
            movie_rows = movies_df[movies_df["Title"] == title]["MovieID"].values
            if len(movie_rows) > 0:
                recommended_movies.add(movie_rows[0])

        if len(recommended_movies) == 0:
            precision, recall = 0.0, 0.0
        else:
            precision = len(ground_truth_movies & recommended_movies) / len(recommended_movies)
            recall = len(ground_truth_movies & recommended_movies) / len(ground_truth_movies)

        results.append({
            "UserID": user_id, "Precision": precision, "Recall": recall,
            "GT_Size": len(ground_truth_movies), "Rec_Size": len(recommended_movies),
        })

    if progress_cb:
        progress_cb(1.0)

    return pd.DataFrame(results)
