import pandas as pd

def load_and_filter_data(df, genres=None, year_range=(2000, 2020), rating_range=(6.0, 9.0)):
    df = df.copy()

    # Drop rows with missing critical values
    df.dropna(subset=["title", "genres", "releaseYear", "imdbAverageRating"], inplace=True)

    # Convert genres to list
    df["genres"] = df["genres"].apply(lambda x: x.split(", "))

    # Filter by genres
    if genres:
        df = df[df["genres"].apply(lambda x: any(g in x for g in genres))]

    # Filter by year
    df = df[(df["releaseYear"] >= year_range[0]) & (df["releaseYear"] <= year_range[1])]

    # Filter by rating
    df = df[(df["imdbAverageRating"] >= rating_range[0]) & (df["imdbAverageRating"] <= rating_range[1])]

    return df
