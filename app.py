import streamlit as st
import pandas as pd
import core  # ربط مباشر مع ملف الـ core.py الفول فيتشرز بتاعك

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System (Market Basket Analysis)")
st.markdown("This app uses the **Apriori Algorithm** to recommend movies based on viewing patterns and co-occurrences.")

# 1. تحميل البيانات وتدريب الموديل مرة واحدة في الذاكرة (كاش)
@st.cache_resource
def initialize_core_system():
    # استدعاء الدوال الأصلية من ملفك
    movies, ratings, users = core.load_data(data_dir="data")
    # تدريب الموديل الأصلي بتاعك
    frequent_itemsets, rules, num_transactions, num_items = core.train_apriori_model(
        movies, ratings, min_support=0.05, min_lift=1.2
    )
    # بناء الـ Rule Index المستعمل في الـ Evaluation
    rule_index = core.build_rule_index(rules)
    return movies, ratings, users, rules, rule_index, num_transactions, num_items

with st.spinner("🚀 Loading system and training Apriori model from core.py..."):
    try:
        movies_df, ratings_df, users_df, rules_df, rule_index, n_trans, n_items = initialize_core_system()
        system_ready = True
    except Exception as e:
        st.error(f"Error loading core system: {e}")
        system_ready = False

if system_ready:
    # القائمة الجانبية لعرض إحصائيات الموديل من الـ core بتاعك
    st.sidebar.header("📊 Model Statistics")
    st.sidebar.write(f"**Total Transactions:** {n_trans}")
    st.sidebar.write(f"**Total Unique Movies:** {n_items}")
    st.sidebar.write(f"**Generated Association Rules:** {len(rules_df)}")

    # تصميم التابات لعرض كل الفيتشرز
    tab1, tab2, tab3 = st.tabs([
        "🎯 User-Based Recommendations", 
        "🆕 Cold Start (New Session)", 
        "📈 Model Evaluation"
    ])

    # -------------------------------------------------------------
    # الفيتشر الأولى: التوصية لمستخدم حالي بناءً على الـ ID والـ History
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Get Recommendations for an Existing User")
        
        # اختيار User ID من المتاحين في الداتا عندك
        available_users = sorted(ratings_df["UserID"].unique())
        selected_user = st.selectbox("Select User ID:", available_users)
        
        if st.button("Generate User Recommendations"):
            with st.spinner("Fetching history and rules..."):
                # استدعاء دالتك الأصلية للحصول على التوصيات والـ history
                recs, liked_titles = core.get_top_n_recommendations(
                    rules_df, selected_user, ratings_df, movies_df, n=5
                )
                
                # عرض الـ History الخاص بالمستخدم (الفيتشر الأصلية)
                st.markdown("### 📜 User Watching History (Rated >= 4):")
                if liked_titles:
                    st.write(", ".join(list(liked_titles)[:15]) + ("..." if len(liked_titles) > 15 else ""))
                else:
                    st.write("No high-rated movies found in history.")
                
                # عرض التوصيات المعتمدة على الـ lift
                st.markdown("### 🔮 Top Recommended Movies for this User:")
                if recs:
                    for rank, (movie, lift_score) in enumerate(recs, 1):
                        st.write(f"**{rank}.** {movie} *(Combined Lift: {lift_score:.2f})*")
                else:
                    st.warning("No explicit rules matched this user's history. Showing popular movies instead!")
                    popular_ids = ratings_df.groupby('MovieID').size().sort_values(ascending=False).head(5).index
                    popular_recs = movies_df[movies_df['MovieID'].isin(popular_ids)]['Title'].tolist()
                    for rank, title in enumerate(popular_recs, 1):
                        st.write(f"**{rank}.** {title}")

    # -------------------------------------------------------------
    # الفيتشر الثانية: الـ Cold Start التفاعلي (سلة الأفلام)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Simulate a New User Session (Cold Start)")
        st.info("Select multiple movies to simulate an incoming active user session:")
        
        movie_list = sorted(movies_df['Title'].unique())
        user_basket = st.multiselect("Select movies you have watched:", movie_list)
        
        if st.button("Generate Context Recommendations"):
            if not user_basket:
                st.warning("Please select at least one movie first.")
            else:
                with st.spinner("Running cold start logic..."):
                    # استدعاء دالة الـ cold start اللي جوه ملفك بالظبط
                    cold_recs = core.cold_start_recommendations(rules_df, user_basket, n=5)
                    
                    if cold_recs:
                        st.success("Recommendations based on your current session basket:")
                        for rank, (movie, lift_score) in enumerate(cold_recs, 1):
                            st.write(f"**{rank}.** {movie} *(Lift: {lift_score:.2f})*")
                    else:
                        st.warning("No association rules found for this selection. Try adding more or different movies!")

    # -------------------------------------------------------------
    # الفيتشر الثالثة: تقييم النموذج (Evaluation Metrics)
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Model Evaluation Metrics")
        st.markdown("Run evaluation using Train/Test split logic from `core.py`:")
        
        sample_size = st.slider("Select User Sample Size for Evaluation:", 50, 500, 100)
        
        if st.button("Run Evaluation"):
            with st.spinner("Splitting data and calculating Precision/Recall..."):
                train_ratings, test_ratings = core.make_train_test_split(ratings_df)
                
                # استدعاء دالة التقييم الأصلية من ملفك
                eval_df = core.evaluate_apriori_recommender(
                    rule_index, train_ratings, test_ratings, movies_df, sample_size=sample_size, k=5
                )
                
                if not eval_df.empty:
                    avg_precision = eval_df["Precision"].mean()
                    avg_recall = eval_df["Recall"].mean()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("🎯 Average Precision @ K", f"{avg_precision:.4f}")
                    col2.metric("📈 Average Recall @ K", f"{avg_recall:.4f}")
                    
                    st.dataframe(eval_df.head(10))
                else:
                    st.error("Evaluation returned empty results. Ensure the sample size contains users with valid test data.")
