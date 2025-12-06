import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ==================== RuleFitLite Class ====================
class RuleFitLite:
    def __init__(self, n_estimators=200, max_depth=5, min_samples_leaf=1,
                 random_state=42, alpha=1.0):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.alpha = float(alpha)
        self.lr = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0/max(self.alpha, 1e-8),
            max_iter=3000,
            multi_class="auto",
        )
        self.classes_ = None

    def _rules_matrix(self, X):
        mats = []
        for est in self.rf.estimators_:
            M = est.decision_path(X)
            if M.shape[1] > 1:
                M = M[:, 1:]
            mats.append(M)
        if len(mats) == 1:
            return mats[0].tocsr()
        return sp.hstack(mats, format="csr")

    def fit(self, X, y):
        X = np.asarray(X)
        self.rf.fit(X, y)
        self.classes_ = np.unique(y)
        R = self._rules_matrix(X)
        self.lr.fit(R, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        R = self._rules_matrix(X)
        return self.lr.predict_proba(R)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return np.array([self.lr.classes_[i] for i in idx])

# ==================== Question Mapping ====================
QUESTION_MAP = {
    "Database Fundamentals": "Seberapa mahir Anda dalam mengelola dan merancang database?",
    "Computer Architecture": "Seberapa paham Anda tentang struktur dan cara kerja komputer?",
    "Distributed Computing Systems": "Seberapa familiar Anda dengan sistem komputasi terdistribusi?",
    "Cyber Security": "Seberapa kuat kemampuan Anda dalam keamanan siber?",
    "Networking": "Seberapa ahli Anda dalam jaringan komputer?",
    "Software Development": "Seberapa berpengalaman Anda dalam pengembangan perangkat lunak?",
    "Programming Skills": "Seberapa mahir Anda dalam pemrograman?",
    "Project Management": "Seberapa terampil Anda dalam mengelola proyek?",
    "Computer Forensics Fundamentals": "Seberapa paham Anda tentang forensik komputer?",
    "Technical Communication": "Seberapa baik Anda dalam berkomunikasi secara teknis?",
    "AI ML": "Seberapa mahir Anda dalam AI dan Machine Learning?",
    "Software Engineering": "Seberapa kuat pemahaman Anda tentang rekayasa perangkat lunak?",
    "Business Analysis": "Seberapa terampil Anda dalam menganalisis bisnis?",
    "Communication skills": "Seberapa baik kemampuan komunikasi Anda?",
    "Data Science": "Seberapa mahir Anda dalam ilmu data?",
    "Troubleshooting skills": "Seberapa cepat Anda dalam memecahkan masalah teknis?",
    "Graphics Designing": "Seberapa kreatif Anda dalam desain grafis?",
    "Openness": "Seberapa terbuka Anda terhadap pengalaman dan ide baru?",
    "Conscientousness": "Seberapa teliti dan terorganisir Anda dalam bekerja?",
    "Extraversion": "Seberapa aktif Anda dalam berinteraksi sosial?",
    "Agreeableness": "Seberapa mudah Anda bekerja sama dengan orang lain?",
    "Emotional_Range": "Seberapa stabil emosi Anda dalam situasi sulit?",
    "Conversation": "Seberapa nyaman Anda dalam percakapan mendalam?",
    "Openness to Change": "Seberapa fleksibel Anda menerima perubahan?",
    "Hedonism": "Seberapa penting kesenangan pribadi dalam hidup Anda?",
    "Self-enhancement": "Seberapa penting kesuksesan pribadi bagi Anda?",
    "Self-transcendence": "Seberapa peduli Anda terhadap kesejahteraan orang lain?"
}

# ==================== Load Model ====================
# ==================== Load Model ====================
@st.cache_resource
def load_model():
    ARTIFACTS_DIR = Path("artifacts")
    
    # Load metadata untuk info hard/soft skills dan weights
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
        hard_skill_cols = metadata["hard_skills"]
        soft_skill_cols = metadata["soft_skills"]
        weights = metadata["ensemble_weights"]
    
    # Load model hard & soft skills
    rf_lite_hard = joblib.load(ARTIFACTS_DIR / "rf_hard_normalized.pth")
    rf_lite_soft = joblib.load(ARTIFACTS_DIR / "rf_soft_normalized.pth")
    preprocess_hard = joblib.load(ARTIFACTS_DIR / "preprocess_hard_normalized.pth")
    preprocess_soft = joblib.load(ARTIFACTS_DIR / "preprocess_soft_normalized.pth")
    
    # Feature columns (gabungan hard + soft)
    feature_columns = hard_skill_cols + soft_skill_cols
    
    return rf_lite_hard, rf_lite_soft, preprocess_hard, preprocess_soft, feature_columns, hard_skill_cols, soft_skill_cols, weights

# ==================== Prediction Function ====================
def predict_jobs(user_data: Dict[str, float], rf_lite_hard, rf_lite_soft, 
                 preprocess_hard, preprocess_soft, feature_columns, 
                 hard_skill_cols, soft_skill_cols, weights) -> pd.DataFrame:
    
    # Pisahkan data hard & soft skills
    user_data_hard = {k: user_data.get(k, np.nan) for k in hard_skill_cols}
    user_data_soft = {k: user_data.get(k, np.nan) for k in soft_skill_cols}
    
    # Create DataFrame untuk hard & soft
    user_df_hard = pd.DataFrame([user_data_hard])
    user_df_soft = pd.DataFrame([user_data_soft])
    
    # Preprocess
    user_df_pre_hard = preprocess_hard.transform(user_df_hard)
    user_df_pre_soft = preprocess_soft.transform(user_df_soft)
    
    # Predict probabilities dari masing-masing model
    proba_hard = rf_lite_hard.predict_proba(user_df_pre_hard)[0]
    proba_soft = rf_lite_soft.predict_proba(user_df_pre_soft)[0]
    
    # Ensemble dengan weighted average (60:40)
    weight_hard = weights["hard"]
    weight_soft = weights["soft"]
    proba = weight_hard * proba_hard + weight_soft * proba_soft
    
    classes = rf_lite_hard.lr.classes_
    
    # Create leaderboard
    results = []
    for i, job in enumerate(classes):
        results.append({
            "Rank": i + 1,
            "Job Role": job,
            "Match Score": f"{proba[i]*100:.2f}%",
            "Probability": proba[i]
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Probability", ascending=False).reset_index(drop=True)
    df_results["Rank"] = range(1, len(df_results) + 1)
    
    return df_results[["Rank", "Job Role", "Match Score"]]

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(
        page_title="Job Recommendation System",
        page_icon="💼",
        layout="wide"
    )
    
    # Load model
    rf_lite_hard, rf_lite_soft, preprocess_hard, preprocess_soft, feature_columns, hard_skill_cols, soft_skill_cols, weights = load_model()
    
    # Header
    st.title("💼 AI-Powered Job Recommendation System")
    st.markdown("### Temukan pekerjaan yang paling cocok untuk Anda!")
    st.markdown("---")
    
    # Session state initialization
    if 'page' not in st.session_state:
        st.session_state.page = 'email'
    if 'email' not in st.session_state:
        st.session_state.email = ''
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    
    # ==================== EMAIL PAGE ====================
    if st.session_state.page == 'email':
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("## 📧 Selamat Datang!")
            st.markdown("Silakan masukkan email Anda untuk memulai")
            
            email = st.text_input(
                "Email Address",
                placeholder="contoh@email.com",
                key="email_input"
            )
            
            if st.button("Mulai Assessment ➡️", use_container_width=True):
                if email and "@" in email:
                    st.session_state.email = email
                    st.session_state.responses = {}  # Reset responses
                    st.session_state.page = 'questions'
                    st.rerun()
                else:
                    st.error("❌ Mohon masukkan email yang valid!")
    
    # ==================== QUESTIONS PAGE ====================
    elif st.session_state.page == 'questions':
        st.markdown(f"### 👤 Assessment untuk: `{st.session_state.email}`")
        st.markdown("---")
        
        # Questions form
        with st.form("assessment_form"):
            st.markdown("### 📝 Jawab pertanyaan berikut (Skala 1-5)")
            
            # Legend
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <b>Panduan Rating:</b> 
            1 = Sangat Lemah/Tidak Setuju | 
            2 = Lemah | 
            3 = Cukup | 
            4 = Kuat | 
            5 = Sangat Kuat/Sangat Setuju
            </div>
            """, unsafe_allow_html=True)
            
            # Group questions by category
            categories = {
                "💻 Technical Skills": feature_columns[:17],
                "🧠 Personality Traits": feature_columns[17:]
            }
            
            responses = {}
            
            for category, features in categories.items():
                st.markdown(f"#### {category}")
                
                for feature in features:
                    question = QUESTION_MAP.get(feature, f"Rate your {feature}")
                    
                    # Get previous value or None
                    previous_value = st.session_state.responses.get(feature, None)
                    
                    # Radio button horizontal
                    col_q, col_r = st.columns([3, 2])
                    
                    with col_q:
                        st.markdown(f"**{question}**")
                    
                    with col_r:
                        # Gunakan index sesuai previous value
                        if previous_value is None:
                            selected = st.radio(
                                f"radio_{feature}",
                                options=[1, 2, 3, 4, 5],
                                index=None,
                                horizontal=True,
                                key=f"radio_{feature}",
                                label_visibility="collapsed"
                            )
                        else:
                            selected = st.radio(
                                f"radio_{feature}",
                                options=[1, 2, 3, 4, 5],
                                index=previous_value - 1,
                                horizontal=True,
                                key=f"radio_{feature}",
                                label_visibility="collapsed"
                            )
                        
                        responses[feature] = selected
                
                st.markdown("")
            
            # Buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.form_submit_button("⬅️ Kembali", use_container_width=True):
                    st.session_state.page = 'email'
                    st.rerun()
            
            with col3:
                # Submit tanpa disabled
                if st.form_submit_button("Lihat Hasil 🎯", use_container_width=True):
                    # Check jika semua sudah dijawab
                    if all(v is not None for v in responses.values()):
                        st.session_state.responses = responses
                        st.session_state.page = 'results'
                        st.rerun()
                    else:
                        st.error("❌ Mohon jawab semua pertanyaan terlebih dahulu!")

    
    # ==================== RESULTS PAGE ====================
    elif st.session_state.page == 'results':
        st.markdown(f"### 🎉 Hasil Rekomendasi untuk: `{st.session_state.email}`")
        st.markdown("---")
        
        # Calculate average skill score
        avg_score = np.mean(list(st.session_state.responses.values()))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Average Skill Score", f"{avg_score:.2f}/5")
        
        with col2:
            st.metric("✅ Questions Completed", f"{len(st.session_state.responses)}/{len(feature_columns)}")
        
        with col3:
            rating = "⭐" * int(avg_score)
            st.metric("⭐ Rating", rating if rating else "⭐")
        
        st.markdown("---")
        
        # Predict and display leaderboard
        st.markdown("### 🏆 Job Recommendation Leaderboard")
        
        with st.spinner("Menganalisis profil Anda..."):
            results_df = predict_jobs(
                st.session_state.responses,
                rf_lite_hard,
                rf_lite_soft,
                preprocess_hard,
                preprocess_soft,
                feature_columns,
                hard_skill_cols,
                soft_skill_cols,
                weights
            )
        
        # Highlight top 3
        def highlight_top3(row):
            if row['Rank'] == 1:
                return ['background-color: #FFD700'] * len(row)  # Gold
            elif row['Rank'] == 2:
                return ['background-color: #C0C0C0'] * len(row)  # Silver
            elif row['Rank'] == 3:
                return ['background-color: #CD7F32'] * len(row)  # Bronze
            return [''] * len(row)
        
        styled_df = results_df.style.apply(highlight_top3, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Top recommendation
        top_job = results_df.iloc[0]
        st.success(f"### 🎯 Rekomendasi Terbaik: **{top_job['Job Role']}** ({top_job['Match Score']})")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Isi Ulang", use_container_width=True):
                st.session_state.responses = {}
                st.session_state.page = 'questions'
                st.rerun()
        
        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"job_recommendations_{st.session_state.email.split('@')[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🏠 Mulai Baru", use_container_width=True):
                st.session_state.page = 'email'
                st.session_state.responses = {}
                st.session_state.email = ''
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by RuleFit-Lite Model | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()