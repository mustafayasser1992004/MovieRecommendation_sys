"""
Movie Recommendation System — Live Application
Market Basket Analysis with the Apriori Algorithm (MovieLens 1M)
"""

import time

import pandas as pd
import streamlit as st
import plotly.express as px

import core

st.set_page_config(page_title="Movie Recommender — Apriori", page_icon="🎬", layout="wide")

AGE_MAP = core.AGE_MAP
OCCUPATION_MAP = core.OCCUPATION_MAP
RATING_THRESHOLD = core.RATING_THRESHOLD


@st.cache_data(show_spinner=False)
def cached_load_data():
    return core.load_data(data_dir="data")


@st.cache_resource(show_spinner=False)
def cached_train_model(min_support: float, min_lift: float):
    movies, ratings, _ = cached_load_data()
    freq, rules, n_tx, n_items = core.train_apriori_model(movies, ratings, min_support, min_lift)
    return {"frequent_itemsets": freq, "rules": rules, "n_transactions": n_tx, "n_items": n_items}


@st.cache_resource(show_spinner=False)
def cached_rule_index(_rules_df, cache_key):
    return core.build_rule_index(_rules_df)


@st.cache_data(show_spinner=False)
def cached_train_test_split(_ratings_df, cache_key):
    return core.make_train_test_split(_ratings_df)


st.sidebar.title("🎬 Movie Recommender")
st.sidebar.caption("Apriori-based Market Basket Analysis · MovieLens 1M")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview & EDA", "👤 Recommend for a User", "✨ New Session (Cold Start)",
     "🔗 Association Rules Explorer", "📈 Model Evaluation"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
min_support = st.sidebar.slider("Min support", 0.02, 0.15, 0.05, 0.01,
                                 help="Minimum fraction of users an itemset must appear in.")
min_lift = st.sidebar.slider("Min lift", 1.0, 3.0, 1.2, 0.1,
                              help="Minimum lift threshold for keeping a rule.")

movies, ratings, users = cached_load_data()

with st.spinner("Training the Apriori model (runs once, then cached)…"):
    model = cached_train_model(min_support, min_lift)

rules = model["rules"]
rule_index = cached_rule_index(rules, cache_key=f"{min_support}-{min_lift}")

st.sidebar.markdown("---")
st.sidebar.metric("Frequent itemsets", f"{len(model['frequent_itemsets']):,}")
st.sidebar.metric("Association rules", f"{len(rules):,}")
st.sidebar.metric("User transactions", f"{model['n_transactions']:,}")


if page == "🏠 Overview & EDA":
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Market Basket Analysis with the Apriori Algorithm")
    st.markdown(
        "This live app mines association rules from the **MovieLens 1M** dataset "
        "(6,040 users, 3,883 movies, 1,000,209 ratings) to generate personalized "
        "and cold-start movie recommendations."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("Users", f"{len(users):,}")
    c3.metric("Ratings", f"{len(ratings):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        rating_counts = ratings["Rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"}, title="Distribution of Movie Ratings",
            color_discrete_sequence=["#458588"],
        )
        st.plotly_chart(fig, width='stretch')

        gender_counts = users["Gender"].value_counts()
        fig2 = px.pie(
            values=gender_counts.values,
            names=["Male" if g == "M" else "Female" for g in gender_counts.index],
            title="Gender Distribution of Users", color_discrete_sequence=["#458588", "#b16286"],
        )
        st.plotly_chart(fig2, width='stretch')

    with col2:
        users_age_df = users.copy()
        users_age_df["AgeGroup"] = users_age_df["Age"].map(AGE_MAP)
        age_order = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
        age_counts = users_age_df["AgeGroup"].value_counts().reindex(age_order)
        fig3 = px.bar(
            x=age_counts.index, y=age_counts.values,
            labels={"x": "Age Group", "y": "Count"}, title="Age Distribution of Users",
            color_discrete_sequence=["#d65d0e"],
        )
        st.plotly_chart(fig3, width='stretch')

        genre_series = movies["Genres"].str.split("|").explode()
        top_genres = genre_series.value_counts().head(10)
        fig4 = px.bar(
            x=top_genres.values, y=top_genres.index, orientation="h",
            labels={"x": "Number of Movies", "y": "Genre"}, title="Top 10 Genres",
            color_discrete_sequence=["#689d6a"],
        )
        fig4.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig4, width='stretch')


elif page == "👤 Recommend for a User":
    st.title("👤 Personalized Recommendations")
    st.markdown("Pick an existing `UserID` from the dataset to see their profile and generate recommendations.")

    col1, col2 = st.columns([1, 2])
    with col1:
        user_id = st.number_input(
            "User ID", min_value=int(users["UserID"].min()), max_value=int(users["UserID"].max()),
            value=1, step=1,
        )
        n_recs = st.slider("Number of recommendations", 3, 15, 5)
        run = st.button("🎯 Get Recommendations", type="primary", width='stretch')

    with col2:
        user_row = users[users["UserID"] == user_id]
        if not user_row.empty:
            u = user_row.iloc[0]
            st.markdown("**User profile**")
            uc1, uc2, uc3 = st.columns(3)
            uc1.write(f"**Gender:** {'Male' if u['Gender']=='M' else 'Female'}")
            uc2.write(f"**Age group:** {AGE_MAP.get(u['Age'], u['Age'])}")
            uc3.write(f"**Occupation:** {OCCUPATION_MAP.get(u['Occupation'], 'unknown')}")

    if run:
        recs, liked_titles = core.get_top_n_recommendations(rules, user_id, ratings, movies, n=n_recs)

        st.markdown("---")
        st.subheader(f"⭐ Movies User #{user_id} already liked (rated ≥ {RATING_THRESHOLD})")
        if liked_titles:
            liked_list = sorted(liked_titles)
            st.write(", ".join(liked_list[:15]) + (" ..." if len(liked_list) > 15 else ""))
        else:
            st.info("This user has no ratings ≥ 4, so no rule-based recommendation can be generated.")

        st.subheader("🎬 Recommended Movies")
        if recs:
            rec_df = pd.DataFrame(recs, columns=["Title", "Recommendation Score (Σ lift)"])
            rec_df.index += 1
            st.dataframe(rec_df, width='stretch')
        else:
            st.warning("No recommendations found for this user with the current model settings. "
                       "Try lowering the min support / min lift in the sidebar.")


elif page == "✨ New Session (Cold Start)":
    st.title("✨ New User Session (Cold Start)")
    st.markdown(
        "Simulate a brand-new visitor with no rating history. Pick a few movies they "
        "just said they like, and the engine instantly returns correlated recommendations "
        "— no retraining needed."
    )

    all_titles = sorted(movies["Title"].unique())
    default_titles = [t for t in [
        "Star Wars: Episode IV - A New Hope (1977)",
        "Jurassic Park (1993)",
        "Matrix, The (1999)",
    ] if t in all_titles]

    selected = st.multiselect("Movies this session likes:", all_titles, default=default_titles)
    n_recs = st.slider("Number of recommendations", 3, 15, 5, key="cold_n")

    if st.button("⚡ Get Instant Recommendations", type="primary"):
        if not selected:
            st.warning("Pick at least one movie first.")
        else:
            recs = core.cold_start_recommendations(rules, selected, n=n_recs)
            st.subheader("🎬 Recommended for this session")
            if recs:
                rec_df = pd.DataFrame(recs, columns=["Title", "Recommendation Score (Σ lift)"])
                rec_df.index += 1
                st.dataframe(rec_df, width='stretch')
            else:
                st.warning("No rules matched this combination. Try different or fewer movies, "
                           "or lower the min support / min lift in the sidebar.")


elif page == "🔗 Association Rules Explorer":
    st.title("🔗 Association Rules Explorer")
    st.markdown("Browse the mined rules directly, ranked by lift.")

    search = st.text_input("Filter by movie title (in antecedents or consequents)")
    top_n = st.slider("Rows to show", 10, 200, 25)

    display_rules = rules.copy()
    display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(x))
    display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(x))

    if search:
        mask = (
            display_rules["antecedents"].str.contains(search, case=False, na=False)
            | display_rules["consequents"].str.contains(search, case=False, na=False)
        )
        display_rules = display_rules[mask]

    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    st.dataframe(display_rules[cols].head(top_n), width='stretch')
    st.caption(f"Showing {min(top_n, len(display_rules))} of {len(display_rules):,} matching rules.")


elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation — Precision & Recall")
    st.markdown(
        "Runs the same train/test evaluation as the notebook: 80/20 split per user, "
        "then checks whether the top-K recommendations show up in each user's held-out "
        "positive ratings."
    )

    col1, col2 = st.columns(2)
    sample_size = col1.slider("Number of test users to evaluate", 50, 2000, 300, 50)
    k = col2.slider("Top-K recommendations", 3, 15, 5)

    if st.button("▶️ Run Evaluation", type="primary"):
        train_ratings, test_ratings = cached_train_test_split(ratings, cache_key="split_v1")
        progress = st.progress(0.0)

        t0 = time.time()
        results_df = core.evaluate_apriori_recommender(
            rule_index, train_ratings, test_ratings, movies,
            sample_size=sample_size, k=k, progress_cb=progress.progress,
        )
        elapsed = time.time() - t0

        if results_df.empty:
            st.warning("No valid test users found (try a different sample size or model settings).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Precision", f"{results_df['Precision'].mean():.4f}")
            c2.metric("Average Recall", f"{results_df['Recall'].mean():.4f}")
            c3.metric("Users evaluated", f"{len(results_df):,}")
            st.caption(f"Evaluation completed in {elapsed:.1f}s")

            fig = px.histogram(
                results_df, x="Precision", nbins=20, title="Precision Distribution Across Users",
                color_discrete_sequence=["#458588"],
            )
            st.plotly_chart(fig, width='stretch')

            st.dataframe(results_df.head(50), width='stretch')
