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
from streamlit_lottie import st_lottie

# ==================== Page Config ====================
st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Custom CSS ====================
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Force light mode */
    .stApp {
        background: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Force all text to be dark */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #1e293b !important;
    }
    
    /* Landing Page Styles */
    .brand-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .brand-subtitle {
        font-size: 1.2rem;
        color: #64748b !important;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.7;
    }
    
    .input-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        gap: 0.75rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: white;
        padding: 0.6rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        color: #64748b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Assessment Styles */
    .assessment-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .assessment-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
    }
    
    .assessment-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    /* Progress bar */
    .progress-wrapper {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #374151;
    }
    
    .progress-track {
        background: #e5e7eb;
        border-radius: 100px;
        height: 8px;
        overflow: hidden;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        height: 100%;
        border-radius: 100px;
        transition: width 0.3s ease;
    }
    
    /* Question card */
    .question-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .category-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid;
    }
    
    .category-tech {
        color: #6366f1;
        border-bottom-color: #6366f1;
    }
    
    .category-soft {
        color: #8b5cf6;
        border-bottom-color: #8b5cf6;
    }
    
    .question-text {
        font-size: 0.9rem;
        color: #374151;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Rating legend */
    .rating-legend {
        background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%);
        border: 1px solid #e0e7ff;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .legend-title {
        font-weight: 600;
        color: #4338ca;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    
    .legend-items {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.8rem;
        color: #6366f1;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .legend-num {
        background: #6366f1;
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    /* Results */
    .results-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .results-title {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    
    .top-job-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .top-job-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.4rem;
    }
    
    .top-job-name {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .top-job-score {
        background: rgba(255,255,255,0.2);
        display: inline-block;
        padding: 0.4rem 1.25rem;
        border-radius: 50px;
        font-weight: 600;
    }
    
    /* Leaderboard */
    .leaderboard-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .leaderboard-header {
        background: #1e1b4b;
        color: white;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .leaderboard-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .leaderboard-item:last-child {
        border-bottom: none;
    }
    
    .leaderboard-rank {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        margin-right: 0.75rem;
    }
    
    .rank-gold {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1f2937;
    }
    
    .rank-silver {
        background: linear-gradient(135deg, #d1d5db 0%, #9ca3af 100%);
        color: #1f2937;
    }
    
    .rank-bronze {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white;
    }
    
    .rank-default {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .leaderboard-job {
        flex: 1;
        font-weight: 500;
        color: #1f2937;
        font-size: 0.9rem;
    }
    
    .leaderboard-score {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 0.3rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    }
    
    /* Input */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Lottie container - force white/transparent background */
    .lottie-container {
        background: #f8fafc;
        border-radius: 20px;
        padding: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Force Lottie iframe/canvas to have white background */
    iframe[title="lottie"] {
        background: white !important;
        border-radius: 20px;
    }
    
    [data-testid="stLottie"] > div {
        background: white !important;
        border-radius: 20px;
    }
    
    /* Override Streamlit dark elements */
    .stApp [data-testid="stForm"] {
        background: transparent;
    }
    
    [data-testid="stTextInput"] {
        background: transparent;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== Load Lottie ====================
@st.cache_data
def load_lottie_file(filepath: str):
    """Load lottie animation from local file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

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
@st.cache_resource
def load_model():
    ARTIFACTS_DIR = Path("artifacts")
    
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
        hard_skill_cols = metadata["hard_skills"]
        soft_skill_cols = metadata["soft_skills"]
        weights = metadata["ensemble_weights"]
    
    rf_lite_hard = joblib.load(ARTIFACTS_DIR / "rf_hard_normalized.pth")
    rf_lite_soft = joblib.load(ARTIFACTS_DIR / "rf_soft_normalized.pth")
    preprocess_hard = joblib.load(ARTIFACTS_DIR / "preprocess_hard_normalized.pth")
    preprocess_soft = joblib.load(ARTIFACTS_DIR / "preprocess_soft_normalized.pth")
    
    feature_columns = hard_skill_cols + soft_skill_cols
    
    return rf_lite_hard, rf_lite_soft, preprocess_hard, preprocess_soft, feature_columns, hard_skill_cols, soft_skill_cols, weights

# ==================== Prediction Function ====================
def predict_jobs(user_data: Dict[str, float], rf_lite_hard, rf_lite_soft, 
                 preprocess_hard, preprocess_soft, feature_columns, 
                 hard_skill_cols, soft_skill_cols, weights) -> pd.DataFrame:
    
    user_data_hard = {k: user_data.get(k, np.nan) for k in hard_skill_cols}
    user_data_soft = {k: user_data.get(k, np.nan) for k in soft_skill_cols}
    
    user_df_hard = pd.DataFrame([user_data_hard])
    user_df_soft = pd.DataFrame([user_data_soft])
    
    user_df_pre_hard = preprocess_hard.transform(user_df_hard)
    user_df_pre_soft = preprocess_soft.transform(user_df_soft)
    
    proba_hard = rf_lite_hard.predict_proba(user_df_pre_hard)[0]
    proba_soft = rf_lite_soft.predict_proba(user_df_pre_soft)[0]
    
    weight_hard = weights["hard"]
    weight_soft = weights["soft"]
    proba = weight_hard * proba_hard + weight_soft * proba_soft
    
    classes = rf_lite_hard.lr.classes_
    
    results = []
    for i, job in enumerate(classes):
        results.append({
            "Rank": i + 1,
            "Job Role": job,
            "Match Score": f"{proba[i]*100:.1f}%",
            "Probability": proba[i]
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Probability", ascending=False).reset_index(drop=True)
    df_results["Rank"] = range(1, len(df_results) + 1)
    
    return df_results[["Rank", "Job Role", "Match Score"]]

# ==================== Main App ====================
def main():
    apply_custom_css()
    
    # Load model
    rf_lite_hard, rf_lite_soft, preprocess_hard, preprocess_soft, feature_columns, hard_skill_cols, soft_skill_cols, weights = load_model()
    
    # Session state
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    if 'email' not in st.session_state:
        st.session_state.email = ''
    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    # ==================== LANDING PAGE ====================
    if st.session_state.page == 'landing':
        lottie_animation = load_lottie_file("Manwithtasklist.json")

        # Add vertical spacing to center content
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="brand-title">CareerMatch AI</div>', unsafe_allow_html=True)
            st.markdown('''
            <div class="brand-subtitle">
                Temukan karir yang tepat untuk Anda dengan teknologi Artificial Intelligence. 
                Analisis mendalam berdasarkan skill teknis dan kepribadian Anda.
            </div>
            ''', unsafe_allow_html=True)

            st.markdown('<span class="input-label">📧 Masukkan Email Anda untuk Memulai</span>', unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="nama@email.com", key="email_input", label_visibility="collapsed")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            if st.button("🚀 Mulai Assessment", use_container_width=True):
                if email and "@" in email:
                    st.session_state.email = email
                    st.session_state.responses = {}
                    st.session_state.page = 'assessment'
                    st.rerun()
                else:
                    st.error("❌ Masukkan email yang valid")

            st.markdown('''
            <div class="badge-container">
                <div class="badge"><span>🤖</span><span>Powered by AI</span></div>
                <div class="badge"><span>🎯</span><span>Akurasi Tinggi</span></div>
                <div class="badge"><span>⚡</span><span>Hasil Instan</span></div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            if lottie_animation:
                st_lottie(lottie_animation, height=400, key="lottie_main", loop=True, quality="high")
            else:
                st.warning("Animasi tidak dapat dimuat. Pastikan file Manwithtasklist.json ada di folder Deployment.")

    # ==================== ASSESSMENT PAGE ====================
    elif st.session_state.page == 'assessment':
        st.markdown(f'''
        <div class="assessment-header">
            <div class="assessment-title">📝 Skill Assessment</div>
            <div class="assessment-subtitle">Halo {st.session_state.email}! Jawab semua pertanyaan di bawah.</div>
        </div>
        ''', unsafe_allow_html=True)
        
        answered = sum(1 for v in st.session_state.responses.values() if v is not None)
        total = len(feature_columns)
        progress = (answered / total) * 100
        
        st.markdown(f'''
        <div class="progress-wrapper">
            <div class="progress-header">
                <span>Progress</span>
                <span>{answered}/{total} pertanyaan</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: {progress}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="rating-legend">
            <div class="legend-title">📖 Panduan Rating</div>
            <div class="legend-items">
                <div class="legend-item"><span class="legend-num">1</span> Sangat Lemah</div>
                <div class="legend-item"><span class="legend-num">2</span> Lemah</div>
                <div class="legend-item"><span class="legend-num">3</span> Cukup</div>
                <div class="legend-item"><span class="legend-num">4</span> Kuat</div>
                <div class="legend-item"><span class="legend-num">5</span> Sangat Kuat</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        with st.form("assessment_form"):
            responses = {}
            
            # Technical Skills Section
            st.markdown("### 💻 Technical Skills")
            
            for i, feature in enumerate(feature_columns[:17]):
                question = QUESTION_MAP.get(feature, f"Rate your {feature}")
                prev_val = st.session_state.responses.get(feature, None)
                
                st.markdown(f"**{i+1}. {question}**")
                
                selected = st.radio(
                    f"q_{feature}",
                    options=[1, 2, 3, 4, 5],
                    index=prev_val - 1 if prev_val else None,
                    horizontal=True,
                    key=f"radio_{feature}",
                    label_visibility="collapsed"
                )
                responses[feature] = selected
            
            st.markdown("---")
            
            # Soft Skills Section
            st.markdown("### 🧠 Personality & Soft Skills")
            
            for i, feature in enumerate(feature_columns[17:]):
                question = QUESTION_MAP.get(feature, f"Rate your {feature}")
                prev_val = st.session_state.responses.get(feature, None)
                
                st.markdown(f"**{i+18}. {question}**")
                
                selected = st.radio(
                    f"q_{feature}",
                    options=[1, 2, 3, 4, 5],
                    index=prev_val - 1 if prev_val else None,
                    horizontal=True,
                    key=f"radio_{feature}",
                    label_visibility="collapsed"
                )
                responses[feature] = selected
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                back = st.form_submit_button("⬅️ Kembali", use_container_width=True)
            
            with col3:
                submit = st.form_submit_button("🎯 Lihat Hasil", use_container_width=True)
            
            if back:
                st.session_state.page = 'landing'
                st.rerun()
            
            if submit:
                if all(v is not None for v in responses.values()):
                    st.session_state.responses = responses
                    st.session_state.page = 'results'
                    st.rerun()
                else:
                    st.error("❌ Mohon jawab semua pertanyaan!")

    # ==================== RESULTS PAGE ====================
    elif st.session_state.page == 'results':
        st.markdown(f'''
        <div class="results-header">
            <div class="results-title">🎉 Hasil Rekomendasi Karir</div>
            <div style="opacity: 0.9; margin-top: 0.3rem; font-size: 0.95rem;">untuk {st.session_state.email}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        avg_score = np.mean(list(st.session_state.responses.values()))
        tech_score = np.mean([st.session_state.responses[k] for k in feature_columns[:17]])
        soft_score = np.mean([st.session_state.responses[k] for k in feature_columns[17:]])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.1f}/5</div><div class="metric-label">Skor Rata-rata</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{tech_score:.1f}/5</div><div class="metric-label">Technical Skills</div></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{soft_score:.1f}/5</div><div class="metric-label">Soft Skills</div></div>', unsafe_allow_html=True)
        
        with st.spinner("🔮 Menganalisis profil Anda..."):
            results_df = predict_jobs(
                st.session_state.responses,
                rf_lite_hard, rf_lite_soft,
                preprocess_hard, preprocess_soft,
                feature_columns, hard_skill_cols, soft_skill_cols, weights
            )
        
        top_job = results_df.iloc[0]
        st.markdown(f'''
        <div class="top-job-card">
            <div class="top-job-label">🎯 Rekomendasi #1 untuk Anda</div>
            <div class="top-job-name">{top_job['Job Role']}</div>
            <div class="top-job-score">Match: {top_job['Match Score']}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="leaderboard-card"><div class="leaderboard-header">🏆 Semua Rekomendasi Karir</div>', unsafe_allow_html=True)
        
        for _, row in results_df.iterrows():
            rank = row['Rank']
            if rank == 1:
                rank_class = "rank-gold"
                medal = "🥇"
            elif rank == 2:
                rank_class = "rank-silver"
                medal = "🥈"
            elif rank == 3:
                rank_class = "rank-bronze"
                medal = "🥉"
            else:
                rank_class = "rank-default"
                medal = ""
            
            st.markdown(f'''
            <div class="leaderboard-item">
                <div style="display: flex; align-items: center;">
                    <div class="leaderboard-rank {rank_class}">{rank}</div>
                    <div class="leaderboard-job">{medal} {row['Job Role']}</div>
                </div>
                <div class="leaderboard-score">{row['Match Score']}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Isi Ulang", use_container_width=True):
                st.session_state.responses = {}
                st.session_state.page = 'assessment'
                st.rerun()
        
        with col2:
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"career_{st.session_state.email.split('@')[0]}.csv", "text/csv", use_container_width=True)
        
        with col3:
            if st.button("🏠 Beranda", use_container_width=True):
                st.session_state.page = 'landing'
                st.session_state.responses = {}
                st.session_state.email = ''
                st.rerun()
        
        st.markdown('<div class="footer"><div>🚀 CareerMatch AI</div><div style="margin-top: 0.3rem;">Powered by Machine Learning • Built with Streamlit</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
