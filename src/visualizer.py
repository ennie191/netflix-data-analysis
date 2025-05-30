import plotly.express as px
import pandas as pd
from collections import Counter

def plot_genre_distribution(df):
    all_genres = df["genres"].explode()
    genre_counts = all_genres.value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]

    fig = px.bar(genre_counts.head(10), x="Genre", y="Count", color="Genre",
                 title="Top Genres", template="plotly_dark")
    return fig

def plot_release_trends(df):
    release_counts = df["releaseYear"].value_counts().sort_index().reset_index()
    release_counts.columns = ["Year", "Count"]

    fig = px.line(release_counts, x="Year", y="Count", title="Movies Released Over Years",
                  markers=True, template="plotly_dark")
    return fig
