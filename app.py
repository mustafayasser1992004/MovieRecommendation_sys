import streamlit as st
import pandas as pd
import core  # استدعاء ملف الأكواد الذكي الخاص بك

st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("This app links directly with your **core.py** Apriori engine to serve real-time recommendations.")

# 1. تحميل البيانات وتدريب النموذج مرة واحدة فقط (باستخدام الكاش لتجنب الشاشة البيضاء والتهنيج)
@st.cache_resource
def init_system():
    # تحميل الداتا من الفولدر بتاعك
    movies, ratings, users = core.load_data(data_dir="data")
    
    # تدريب موديل الـ Apriori بناءً على دالتك (مع وضع حد أدنى للـ support ليكون خفيف وسريع)
    frequent_itemsets, rules, _, _ = core.train_apriori_model(movies, ratings, min_support=0.07, min_lift=1.1)
    return movies, ratings, rules

with st.spinner("Initializing Apriori Engine & Loading Datasets... Please wait."):
    try:
        movies_df, ratings_df, rules_df = init_system()
        movie_list = sorted(movies_df['Title'].unique())
        system_ready = True
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        system_ready = False

if system_ready:
    # تقسيم الواجهة إلى تابات ذكية للـ Cold Start
    tab1, tab2 = st.tabs(["🔥 New User (Popular Movies)", "🎯 Cold Start Session Simulation"])
    
    # -------------------------------------------------------------
    # التاب الأول: المستخدم الجديد تماماً (Popular)
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Popular Movies Right Now")
        st.caption("Since you are a new user, here are the top trending movies based on user interactions:")
        
        # حساب أشهر الأفلام بناءً على داتا الـ ratings الفعلي الخاص بك
        popular_ids = ratings_df.groupby('MovieID').size().sort_values(ascending=False).head(5).index
        popular_titles = movies_df[movies_df['MovieID'].isin(popular_ids)]['Title'].tolist()
        
        cols = st.columns(5)
        for i, title in enumerate(popular_titles):
            with cols[i]:
                st.info(f"🎬 {title}")
                
    # -------------------------------------------------------------
    # التاب الثاني: محاكاة جلسة مستخدم (Cold Start Session) باستخدام دالتك
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Simulate an Active User Session")
        st.info("Select movies you liked right now to find associations:")
        
        user_basket = st.multiselect("Select watched movies:", movie_list)
        
        if st.button("Generate Recommendations"):
            if not user_basket:
                st.warning("Please select at least one movie first.")
            else:
                with st.spinner("Finding association rules from core.py..."):
                    # استدعاء دالة الـ Cold Start الموجودة في ملفك بالحرف!
                    recs = core.cold_start_recommendations(rules_df, user_basket, n=5)
                    
                    if recs:
                        st.success("Recommendations based on your current session basket:")
                        for rank, (movie, lift_score) in enumerate(recs, 1):
                            st.write(f"**{rank}.** {movie} *(Lift: {lift_score:.2f})*")
                    else:
                        st.warning("No clear association rules found for this specific combination. Try adding more popular movies!")
