import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Netflix theme colors ---
NETFLIX_RED = "#E50914"
NETFLIX_DARK = "#141414"
NETFLIX_LIGHT_GRAY = "#333333"
NETFLIX_TEXT = "#FFFFFF"

st.set_page_config(page_title="Netflix Movie Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

# Enhanced Netflix-style CSS with fixed white text
st.markdown(f"""
    <style>
    /* Main app background */
    .stApp {{
        background-color: {NETFLIX_DARK};
        color: {NETFLIX_TEXT};
    }}
    
    /* SIDEBAR STYLING */
    .css-1d391kg,
    .css-1lcbmhc,
    .css-17eq0hr,
    section[data-testid="stSidebar"] {{
        background-color: #000000 !important;
        border-right: 2px solid {NETFLIX_RED} !important;
    }}
    
    /* Force all sidebar text to be WHITE */
    .css-1d391kg *,
    .css-1lcbmhc *,
    .css-17eq0hr *,
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: {NETFLIX_TEXT} !important;
        font-weight: bold !important;
    }}
    
    /* Sidebar input fields - WHITE TEXT */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
        border: 2px solid {NETFLIX_RED} !important;
    }}
    
    /* Multiselect dropdown options - WHITE TEXT */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
    }}
    
    /* Multiselect dropdown menu items - WHITE TEXT */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="menu"] {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
    }}
    
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="menu"] li {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
    }}
    
    /* Selected genre tags - WHITE TEXT */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
        background-color: {NETFLIX_RED} !important;
        color: {NETFLIX_TEXT} !important;
    }}
    
    /* Placeholder text - WHITE */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="input"] {{
        color: {NETFLIX_TEXT} !important;
    }}
    
    /* Slider styling in sidebar */
    section[data-testid="stSidebar"] .stSlider {{
        color: {NETFLIX_TEXT} !important;
    }}
    
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
        background-color: {NETFLIX_RED} !important;
    }}
    
    /* All text elements */
    .stMarkdown, .stText, p, span, div {{
        color: {NETFLIX_TEXT} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {NETFLIX_TEXT} !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }}
    
    /* Main title */
    .main-title {{
        color: {NETFLIX_RED} !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Subtitle */
    .subtitle {{
        color: #cccccc !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {NETFLIX_RED};
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: #b8070f;
    }}
    
    /* Main content select boxes */
    .stSelectbox > div > div > div {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
        border: 1px solid #555 !important;
    }}
    
    .stSelectbox label {{
        color: {NETFLIX_TEXT} !important;
        font-weight: bold !important;
    }}
    
    /* Main content multiselect */
    .stMultiSelect > div > div > div {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
        color: {NETFLIX_TEXT} !important;
        border: 1px solid #555 !important;
    }}
    
    .stMultiSelect label {{
        color: {NETFLIX_TEXT} !important;
        font-weight: bold !important;
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        background-color: {NETFLIX_LIGHT_GRAY};
    }}
    
    /* Section headers */
    .section-header {{
        color: {NETFLIX_TEXT} !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        margin: 2rem 0 1rem 0 !important;
        border-bottom: 2px solid {NETFLIX_RED};
        padding-bottom: 0.5rem;
    }}
    
    /* Movie recommendation section */
    .recommendation-section {{
        background-color: {NETFLIX_LIGHT_GRAY};
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #555;
    }}
    
    /* Charts container */
    .chart-container {{
        background-color: {NETFLIX_LIGHT_GRAY};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #555;
    }}
    
    /* Table styling */
    .stTable {{
        background-color: {NETFLIX_LIGHT_GRAY} !important;
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('../data/data.csv', nrows=10000)
    df.dropna(subset=['title', 'genres', 'imdbAverageRating', 'releaseYear'], inplace=True)
    df['imdbAverageRating'] = pd.to_numeric(df['imdbAverageRating'], errors='coerce')
    df['releaseYear'] = pd.to_numeric(df['releaseYear'], errors='coerce')
    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    df.dropna(subset=['imdbAverageRating', 'releaseYear'], inplace=True)
    return df.reset_index(drop=True)

@st.cache_data
def build_recommender(df):
    df['content'] = df['genres'].apply(lambda x: ' '.join(x)) + ' ' + df['type'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim, top_n=10):
    idx_list = df.index[df['title'] == title].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude the selected movie itself
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genres', 'imdbAverageRating', 'releaseYear']]

# Load data
df = load_data()

# Main header
st.markdown(f'<h1 class="main-title">🎬 Netflix Movie Dashboard</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Explore, analyze, and get movie recommendations with a Netflix vibe!</p>', unsafe_allow_html=True)

# Sidebar filters - REMOVED DEBUG INFO
st.sidebar.markdown(f'<h2 style="color: {NETFLIX_TEXT}; font-weight: bold;">Filter Movies</h2>', unsafe_allow_html=True)

# Extract individual genres (properly handle multiple genres per movie)
all_genres = set()
for genre_list in df['genres']:
    if isinstance(genre_list, list):
        for genre in genre_list:
            if isinstance(genre, str) and genre.strip():
                all_genres.add(genre.strip())

genres_all = sorted(list(all_genres))
selected_genres = st.sidebar.multiselect(
    "Select Genres", 
    genres_all, 
    default=['Drama', 'Comedy'] if 'Drama' in genres_all and 'Comedy' in genres_all else []
)

min_rating, max_rating = st.sidebar.slider("Rating Range", 0.0, 10.0, (5.0, 9.0), 0.1)
year_min, year_max = int(df['releaseYear'].min()), int(df['releaseYear'].max())
selected_years = st.sidebar.slider("Release Year Range", year_min, year_max, (2000, year_max))

# Filter dataframe based on sidebar selections
if selected_genres:
    filtered_df = df[
        (df['imdbAverageRating'] >= min_rating) &
        (df['imdbAverageRating'] <= max_rating) &
        (df['releaseYear'] >= selected_years[0]) &
        (df['releaseYear'] <= selected_years[1]) &
        (df['genres'].apply(lambda gs: any(g in gs for g in selected_genres)))
    ]
else:
    filtered_df = df[
        (df['imdbAverageRating'] >= min_rating) &
        (df['imdbAverageRating'] <= max_rating) &
        (df['releaseYear'] >= selected_years[0]) &
        (df['releaseYear'] <= selected_years[1])
    ]

# Top Movies Section
st.markdown(f'<h2 class="section-header">Top Movies ({len(filtered_df)} found)</h2>', unsafe_allow_html=True)

top_n = st.slider("How many top movies to show?", 5, 30, 10)
top_movies = filtered_df.sort_values('imdbAverageRating', ascending=False).head(top_n)

# Format genres for display
top_movies_display = top_movies.copy()
top_movies_display['genres'] = top_movies_display['genres'].apply(lambda x: ', '.join(x))

st.dataframe(
    top_movies_display[['title', 'genres', 'imdbAverageRating', 'releaseYear']],
    use_container_width=True,
    hide_index=True
)

# Movie recommendation section
st.markdown(f'<h2 class="section-header">🔍 Get Movie Recommendations</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="recommendation-section">', unsafe_allow_html=True)
    
    movie_titles = df['title'].unique().tolist()
    selected_movie = st.selectbox("Choose a movie you like:", movie_titles)
    
    if selected_movie:
        cosine_sim = build_recommender(df)
        recommendations = get_recommendations(selected_movie, df, cosine_sim, top_n=10)
        
        if not recommendations.empty:
            recommendations_display = recommendations.copy()
            recommendations_display['genres'] = recommendations_display['genres'].apply(lambda x: ', '.join(x))
            
            st.write(f"**Movies similar to {selected_movie}:**")
            st.table(recommendations_display)
        else:
            st.write("No recommendations found for this movie.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations
st.markdown(f'<h2 class="section-header">📊 Visual Exploration</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("**Rating Distribution**")
    
    # Interactive histogram with Plotly
    fig_hist = px.histogram(
        df, 
        x='imdbAverageRating', 
        nbins=20,
        title="",
        color_discrete_sequence=[NETFLIX_RED]
    )
    
    fig_hist.update_layout(
        plot_bgcolor=NETFLIX_DARK,
        paper_bgcolor=NETFLIX_DARK,
        font_color=NETFLIX_TEXT,
        xaxis_title="IMDb Average Rating",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    fig_hist.update_xaxes(gridcolor='#444444')
    fig_hist.update_yaxes(gridcolor='#444444')
    
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("**Genre Popularity**")
    
    # Create genre counts
    genre_counts = pd.Series([g for sublist in df['genres'] for g in sublist]).value_counts()
    
    # Interactive bar chart with Plotly
    fig_bar = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="",
        color=genre_counts.values,
        color_continuous_scale=[[0, '#b8070f'], [1, NETFLIX_RED]]
    )
    
    fig_bar.update_layout(
        plot_bgcolor=NETFLIX_DARK,
        paper_bgcolor=NETFLIX_DARK,
        font_color=NETFLIX_TEXT,
        xaxis_title="Number of Movies",
        yaxis_title="Genre",
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    fig_bar.update_xaxes(gridcolor='#444444')
    fig_bar.update_yaxes(gridcolor='#444444')
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f'<p style="text-align: center; color: {NETFLIX_TEXT}; font-size: 1.1rem;">Made with ❤️ to feel like Netflix!</p>', unsafe_allow_html=True)