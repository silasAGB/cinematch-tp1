import os, urllib.request, zipfile, random
import pandas as pd
import streamlit as st
from recommender import (
    build_user_item_matrix, compute_item_similarity,
    get_top_n_recommendations, get_similar_items, evaluate_rmse, load_movielens,
)

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { color: #e94560; font-size: 2.8rem; margin: 0; font-weight: 700; }
.hero p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1.1rem; }

.film-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
    transition: transform 0.2s;
    border: 1px solid #2a2a4a;
    height: 100%;
}
.film-poster {
    width: 100%;
    aspect-ratio: 2/3;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    font-weight: 700;
    color: white;
    position: relative;
    overflow: hidden;
}
.film-title {
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.6rem 0.7rem 0.2rem;
    line-height: 1.3;
    min-height: 2.4rem;
}
.film-year {
    color: #718096;
    font-size: 0.75rem;
    padding: 0 0.7rem 0.4rem;
}
.stars-row {
    padding: 0.3rem 0.5rem 0.7rem;
    display: flex;
    gap: 2px;
}
.star-btn {
    background: none !important;
    border: none !important;
    font-size: 1.4rem;
    cursor: pointer;
    padding: 0 2px;
    line-height: 1;
    transition: transform 0.1s;
}
.star-btn:hover { transform: scale(1.2); }

.reco-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.reco-card:hover { border-color: #e94560; }
.reco-poster {
    width: 56px;
    height: 80px;
    border-radius: 6px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
}
.reco-info { flex: 1; }
.reco-title { color: #e2e8f0; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
.reco-score { color: #e94560; font-size: 0.85rem; font-weight: 500; }
.score-bar-bg { background: #2a2a4a; border-radius: 4px; height: 6px; margin-top: 6px; }
.score-bar { background: linear-gradient(90deg, #e94560, #f6c90e); border-radius: 4px; height: 6px; }

.stat-box {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-num { color: #e94560; font-size: 1.8rem; font-weight: 700; }
.stat-lbl { color: #718096; font-size: 0.8rem; margin-top: 2px; }

[data-testid="stApp"] { background: #0d0d1a; }
section[data-testid="stSidebar"] { background: #12122a !important; }
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c13350) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
h2, h3 { color: #e2e8f0 !important; }
p, li { color: #a8b2d8 !important; }
label { color: #a8b2d8 !important; }
.stSlider > div > div { background: #e94560 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Couleurs pour les affiches placeholder ───
POSTER_COLORS = [
    ("1a1a4e","4a90e2"),("2d1b4e","9b59b6"),("1a3a2e","27ae60"),
    ("4e1a1a","e74c3c"),("3a2d1a","e67e22"),("1a3a4e","2980b9"),
    ("2e1a3a","8e44ad"),("1a4e3a","16a085"),("4e3a1a","d35400"),
    ("1a2e4e","2471a3"),
]

def poster_color(item_id):
    return POSTER_COLORS[item_id % len(POSTER_COLORS)]

def poster_letter(title):
    return title[0].upper() if title else "?"

def poster_html(item_id, title, size=None):
    bg, accent = poster_color(item_id)
    letter = poster_letter(title)
    h = size or "100%"
    w = size or "100%"
    return f"""
    <div style="width:{w};height:{h};background:#{bg};display:flex;
        align-items:center;justify-content:center;
        font-size:2.2rem;font-weight:700;color:#{accent};
        border-radius:8px;border:2px solid #{accent}33;">
        {letter}
    </div>"""

# ─── Chargement & Cache ───
DATA_DIR = "data/ml-100k"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_DIR):
        with st.spinner("📥 Téléchargement de MovieLens 100K..."):
            urllib.request.urlretrieve(URL, "data/ml-100k.zip")
            with zipfile.ZipFile("data/ml-100k.zip", "r") as z:
                z.extractall("data")
            os.remove("data/ml-100k.zip")

@st.cache_data(show_spinner="Chargement des films...")
def load_data():
    download_data()
    return load_movielens(f"{DATA_DIR}/u.data", f"{DATA_DIR}/u.item")

@st.cache_data(show_spinner="Construction de la matrice...")
def build_matrix(_ratings):
    return build_user_item_matrix(_ratings)

@st.cache_data(show_spinner="Calcul des similarités (~30s)...")
def build_similarity(_matrix):
    return compute_item_similarity(_matrix)

# ─── Init session state ───
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}
if "page" not in st.session_state:
    st.session_state.page = "noter"
if "film_sample" not in st.session_state:
    st.session_state.film_sample = None

# ─── Load data ───
ratings, movies = load_data()
train_ratings    = ratings.sort_values("timestamp").iloc[:int(len(ratings)*0.8)]
test_ratings     = ratings.sort_values("timestamp").iloc[int(len(ratings)*0.8):]
user_item_matrix = build_matrix(train_ratings)
item_similarity  = build_similarity(user_item_matrix)

# Sélection d'un échantillon de films populaires à noter
if st.session_state.film_sample is None:
    popular = train_ratings["item_id"].value_counts().head(100).index.tolist()
    sample_ids = random.sample(popular, min(30, len(popular)))
    st.session_state.film_sample = movies[movies["item_id"].isin(sample_ids)].reset_index(drop=True)

film_sample = st.session_state.film_sample

# ─── HERO ───
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Note des films, reçois des recommandations personnalisées</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    nb_rated = len(st.session_state.user_ratings)

    if st.button("⭐ Noter des films"):
        st.session_state.page = "noter"
    if st.button("🚀 Mes recommandations"):
        st.session_state.page = "recommandations"
    if st.button("🔍 Films similaires"):
        st.session_state.page = "similaires"
    if st.button("📊 Évaluation"):
        st.session_state.page = "evaluation"

    st.markdown("---")
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-num">{nb_rated}</div>
        <div class="stat-lbl">films notés</div>
    </div>
    """, unsafe_allow_html=True)

    if nb_rated > 0:
        st.markdown("**Tes notes :**")
        for iid, r in list(st.session_state.user_ratings.items())[-5:]:
            title = movies[movies["item_id"]==iid]["title"].values
            t = title[0][:22] if len(title) > 0 else str(iid)
            st.markdown(f"<span style='color:#a8b2d8;font-size:0.8rem'>{'⭐'*r} {t}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres")
    n_reco = st.slider("Top-N", 5, 20, 10)
    k_neighbors = st.slider("Voisins K", 5, 50, 20)
    st.markdown("---")
    if st.button("🗑️ Réinitialiser mes notes"):
        st.session_state.user_ratings = {}
        st.rerun()

# ═══════════════════════════════════════════
# PAGE 1 — NOTER DES FILMS
# ═══════════════════════════════════════════
if st.session_state.page == "noter":
    st.markdown("## ⭐ Note ces films")
    st.markdown("<p>Clique sur les étoiles pour noter. Plus tu notes, meilleures sont tes recommandations !</p>", unsafe_allow_html=True)

    if st.button("🔀 Nouveaux films à noter"):
        popular = train_ratings["item_id"].value_counts().head(100).index.tolist()
        sample_ids = random.sample(popular, min(30, len(popular)))
        st.session_state.film_sample = movies[movies["item_id"].isin(sample_ids)].reset_index(drop=True)
        st.rerun()

    cols_per_row = 5
    for row_start in range(0, len(film_sample), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            film_idx = row_start + col_idx
            if film_idx >= len(film_sample):
                break
            film = film_sample.iloc[film_idx]
            item_id = int(film["item_id"])
            title   = film["title"]
            current_rating = st.session_state.user_ratings.get(item_id, 0)

            with col:
                # Affiche
                bg, accent = poster_color(item_id)
                letter = poster_letter(title)
                st.markdown(f"""
                <div style="background:#{bg};border-radius:10px;aspect-ratio:2/3;
                    display:flex;align-items:center;justify-content:center;
                    font-size:2.5rem;font-weight:700;color:#{accent};
                    border:2px solid #{accent}55;margin-bottom:6px;">
                    {letter}
                </div>
                <div style="color:#e2e8f0;font-size:0.78rem;font-weight:600;
                    line-height:1.3;min-height:2.2rem;margin-bottom:4px;">
                    {title[:28]}{'...' if len(title)>28 else ''}
                </div>
                """, unsafe_allow_html=True)

                # Étoiles interactives avec radio buttons stylés
                stars = st.radio(
                    label=f"Note_{item_id}",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: "⭐" * x,
                    index=current_rating - 1 if current_rating > 0 else 0,
                    horizontal=True,
                    key=f"star_{item_id}",
                    label_visibility="collapsed",
                )

                col_rate, col_skip = st.columns([2, 1])
                with col_rate:
                    if st.button("Noter", key=f"btn_{item_id}"):
                        st.session_state.user_ratings[item_id] = stars
                        st.rerun()
                with col_skip:
                    if current_rating > 0:
                        st.markdown(f"<div style='color:#e94560;font-size:0.8rem;padding-top:6px;'>{'★'*current_rating}</div>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2a4a;margin:1rem 0'>", unsafe_allow_html=True)

    st.markdown("---")
    nb = len(st.session_state.user_ratings)
    if nb >= 5:
        st.success(f"✅ Tu as noté {nb} films ! Va dans **Mes recommandations** pour voir tes suggestions.")
    else:
        st.info(f"ℹ️ Note au moins 5 films pour obtenir des recommandations. ({nb}/5 notés)")


# ═══════════════════════════════════════════
# PAGE 2 — RECOMMANDATIONS
# ═══════════════════════════════════════════
elif st.session_state.page == "recommandations":
    st.markdown("## 🚀 Tes recommandations personnalisées")

    nb = len(st.session_state.user_ratings)
    if nb < 3:
        st.warning("⚠️ Note au moins 3 films d'abord ! Va sur **⭐ Noter des films**.")
    else:
        st.markdown(f"<p>Basé sur tes <b style='color:#e94560'>{nb} notes</b>, voici ce que tu devrais regarder :</p>", unsafe_allow_html=True)

        with st.spinner("🧠 Calcul des recommandations..."):
            # Construire le vecteur de notes du nouvel utilisateur
            new_user_id = 99999
            new_ratings_rows = [
                {"user_id": new_user_id, "item_id": iid, "rating": r, "timestamp": 999999999}
                for iid, r in st.session_state.user_ratings.items()
            ]
            new_df = pd.DataFrame(new_ratings_rows)
            combined = pd.concat([train_ratings, new_df], ignore_index=True)

            # Reconstruire la matrice avec le nouvel utilisateur
            new_matrix = build_user_item_matrix(combined)

            reco = get_top_n_recommendations(
                new_user_id, new_matrix, item_similarity,
                movies, n=n_reco, k=k_neighbors
            )

        if reco.empty:
            st.warning("Pas assez de données pour générer des recommandations. Note plus de films !")
        else:
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("### 🏆 Top recommandations")
                for _, row in reco.iterrows():
                    iid   = int(row["item_id"])
                    title = str(row["title"])
                    score = float(row["predicted_score"])
                    bg, accent = poster_color(iid)
                    letter = poster_letter(title)
                    pct = int(score / 5 * 100)

                    st.markdown(f"""
                    <div class="reco-card">
                        <div class="reco-poster" style="background:#{bg};border:2px solid #{accent}55;">
                            <span style="color:#{accent}">{letter}</span>
                        </div>
                        <div class="reco-info">
                            <div class="reco-title">{title}</div>
                            <div class="reco-score">Score prédit : {score:.2f} / 5.0</div>
                            <div class="score-bar-bg">
                                <div class="score-bar" style="width:{pct}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_right:
                st.markdown("### 📊 Pourquoi ces films ?")
                st.markdown("""
                <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:1.2rem;">
                <p style="color:#a8b2d8;font-size:0.9rem;line-height:1.7;">
                Le système analyse les <b style="color:#e94560">patterns de notation</b> de milliers d'utilisateurs.<br><br>
                Si tu as aimé les mêmes films qu'un autre groupe d'utilisateurs,
                le système suppose que tu aimeras aussi les films qu'ils ont bien notés mais que tu n'as pas encore vus.<br><br>
                <b style="color:#e94560">Formule :</b>
                </p>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"\hat{r}_{u,i} = \frac{\sum_{j} sim(i,j) \cdot r_{u,j}}{\sum_{j} |sim(i,j)|}")

                st.markdown("### 🎬 Tes notes")
                for iid, r in st.session_state.user_ratings.items():
                    t = movies[movies["item_id"]==iid]["title"].values
                    title = t[0][:30] if len(t) > 0 else str(iid)
                    st.markdown(f"<div style='color:#a8b2d8;font-size:0.82rem;margin:3px 0'>{'⭐'*r} {title}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 3 — FILMS SIMILAIRES
# ═══════════════════════════════════════════
elif st.session_state.page == "similaires":
    st.markdown("## 🔍 Films similaires")
    st.markdown("<p>Cherche un film et découvre les films les plus similaires selon les goûts des utilisateurs.</p>", unsafe_allow_html=True)

    search = st.text_input("🔎 Rechercher un film", placeholder="Ex: Star Wars, Toy Story...")

    if search:
        matches = movies[movies["title"].str.contains(search, case=False, na=False)]
        if matches.empty:
            st.warning("Aucun film trouvé.")
        else:
            sel = st.selectbox("Sélectionne le film", matches["title"].tolist())
            sel_id = int(matches[matches["title"]==sel]["item_id"].iloc[0])

            similar = get_similar_items(sel_id, item_similarity, movies, n=n_reco)

            if similar.empty:
                st.warning("Pas assez de données pour ce film.")
            else:
                bg0, ac0 = poster_color(sel_id)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1rem;background:#1a1a2e;
                    border:1px solid #e9456055;border-radius:12px;padding:1rem;margin:1rem 0">
                    <div style="width:50px;height:70px;background:#{bg0};border-radius:6px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.5rem;font-weight:700;color:#{ac0};border:2px solid #{ac0}55;">
                        {poster_letter(sel)}
                    </div>
                    <div>
                        <div style="color:#e94560;font-weight:700;font-size:1.1rem">{sel}</div>
                        <div style="color:#718096;font-size:0.85rem">Film de référence</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"### Films similaires à *{sel}*")
                cols = st.columns(5)
                for i, (_, row) in enumerate(similar.iterrows()):
                    if i >= 10: break
                    iid   = int(row["item_id"])
                    title = str(row["title"])
                    sim   = float(row["similarity"])
                    bg, accent = poster_color(iid)
                    letter = poster_letter(title)

                    with cols[i % 5]:
                        pct = int(sim * 100)
                        st.markdown(f"""
                        <div style="background:#{bg};border-radius:8px;aspect-ratio:2/3;
                            display:flex;align-items:center;justify-content:center;
                            font-size:2rem;font-weight:700;color:#{accent};
                            border:2px solid #{accent}55;margin-bottom:6px;">
                            {letter}
                        </div>
                        <div style="color:#e2e8f0;font-size:0.75rem;font-weight:600;min-height:2rem;line-height:1.3;">
                            {title[:25]}{'...' if len(title)>25 else ''}
                        </div>
                        <div style="background:#2a2a4a;border-radius:4px;height:4px;margin-top:4px;">
                            <div style="background:#e94560;border-radius:4px;height:4px;width:{pct}%;"></div>
                        </div>
                        <div style="color:#718096;font-size:0.72rem;margin-top:2px;">{sim:.3f}</div>
                        """, unsafe_allow_html=True)

        if (i + 1) % 5 == 0 and i < 9:
            st.markdown("")


# ═══════════════════════════════════════════
# PAGE 4 — ÉVALUATION
# ═══════════════════════════════════════════
elif st.session_state.page == "evaluation":
    st.markdown("## 📊 Évaluation du modèle")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{ratings["user_id"].nunique()}</div>
            <div class="stat-lbl">utilisateurs</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{ratings["item_id"].nunique()}</div>
            <div class="stat-lbl">films</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{len(ratings):,}</div>
            <div class="stat-lbl">notes au total</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Calcul du RMSE")
    st.markdown("<p>Le RMSE mesure l'écart moyen entre les notes prédites et les vraies notes. Plus c'est proche de 0, mieux c'est.</p>", unsafe_allow_html=True)

    sample_size = st.slider("Nombre d'exemples à évaluer", 100, 1000, 300, 100)

    if st.button("📊 Calculer le RMSE", type="primary"):
        with st.spinner("Évaluation en cours..."):
            rmse = evaluate_rmse(test_ratings, user_item_matrix, item_similarity,
                                 k=k_neighbors, sample_size=sample_size)

        st.markdown(f"""
        <div style="background:#1a1a2e;border:2px solid #e94560;border-radius:12px;
            padding:1.5rem;text-align:center;margin:1rem 0">
            <div style="color:#718096;font-size:0.9rem;margin-bottom:0.5rem">RMSE sur {sample_size} exemples</div>
            <div style="color:#e94560;font-size:3rem;font-weight:700">{rmse:.4f}</div>
            <div style="color:#a8b2d8;font-size:0.85rem;margin-top:0.5rem">
                {'✅ Excellent (< 0.8)' if rmse < 0.8 else '🟡 Correct (< 1.0)' if rmse < 1.0 else '🔴 À améliorer'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Split train / test")
    c1, c2 = st.columns(2)
    c1.metric("Train (80%)", f"{len(train_ratings):,} notes")
    c2.metric("Test  (20%)", f"{len(test_ratings):,} notes")
    st.caption("Split temporel : les notes les plus récentes forment le jeu de test.")
