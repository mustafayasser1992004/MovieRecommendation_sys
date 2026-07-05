# 🎬 Movie Recommendation System — Live App

نظام توصية أفلام باستخدام **Apriori Algorithm** (association rule mining) على بيانات **MovieLens 1M**، اتحول من الـ notebook (`finall.ipynb`) لتطبيق Streamlit تفاعلي شغّال Live.

## 📁 الملفات
- `app.py` — واجهة Streamlit (الصفحات والتفاعل فقط)
- `core.py` — منطق الموديل الحقيقي (نفس كود النوتبوك بالظبط: تحميل البيانات، بناء الـ transactions، Apriori، التوصيات، التقييم) — مفصول عشان يكون قابل للاختبار المستقل
- `data/` — ملفات `movies.dat`, `ratings.dat`, `users.dat`
- `requirements.txt` — المكتبات المطلوبة

## ▶️ إزاي تشغّله على جهازك

```bash
# 1. ادخل مجلد المشروع
cd movie_app

# 2. ثبّت المكتبات (يفضل جوه virtual environment)
pip install -r requirements.txt

# 3. شغّل التطبيق
streamlit run app.py
```

هيفتحلك المتصفح تلقائيًا على `http://localhost:8501`.

> أول تشغيل بياخد حوالي 30 ثانية لأن الموديل (Apriori) بيتدرب من الصفر، وبعدها بيتخزن في الـ cache (`st.cache_resource`) فمش هيتكرر تاني إلا لو غيّرت min_support أو min_lift من الشريط الجانبي.

## 🧭 صفحات التطبيق
1. **Overview & EDA** — إحصائيات ورسومات عن البيانات (توزيع التقييمات، الجنس، العمر، أهم الأنواع)
2. **Recommend for a User** — تدخل UserID موجود، يعرضلك بروفايله وتاريخه، ويطلعلك توصيات مبنية على association rules
3. **New Session (Cold Start)** — تختار كذا فيلم "حاليًا بيحبهم مستخدم جديد" من غير أي تاريخ تقييمات، ويطلعلك توصيات فورية
4. **Association Rules Explorer** — تصفح كل الـ rules اللي اتكشفت، مرتبة بالـ Lift، مع بحث باسم الفيلم
5. **Model Evaluation** — يشغّل نفس تقييم الـ train/test split من النوتبوك (Precision & Recall) على عدد مستخدمين تحدده

## 🚀 لو عايز تنشره Live على النت (مجانًا)
أسهل طريقة هي **Streamlit Community Cloud**:
1. ارفع المجلد ده على GitHub repository
2. روح على share.streamlit.io وسجّل دخول بحساب GitHub
3. اختار الريبو والملف `app.py`
4. Deploy — هيديك رابط عام تقدر تحطه في الـ CV أو البورتفوليو بتاعك

## ⚙️ الإعدادات القابلة للتعديل (من الشريط الجانبي)
- **Min support**: أقل نسبة ظهور للفيلم بين المستخدمين عشان يتحسب "متكرر" (افتراضي 0.05 زي النوتبوك)
- **Min lift**: أقل قيمة Lift عشان القاعدة تتحسب "قوية" (افتراضي 1.2 زي النوتبوك)
