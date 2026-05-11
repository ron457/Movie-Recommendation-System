# =========================
# 1. Mount Google Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# 2. Imports
# =========================
import pandas as pd
import numpy as np
import os

# =========================
# 3. Set dataset folder path
# Example: /content/drive/MyDrive/movie_dataset
# =========================
data_path = "/content/drive/MyDrive/movie_dataset"

# Optional: verify files
print("Files in folder:", os.listdir(data_path))

# =========================
# 4. Load datasets
# =========================
movies = pd.read_csv(f"{data_path}/movies.csv")
ratings = pd.read_csv(f"{data_path}/ratings.csv")
tags = pd.read_csv(f"{data_path}/tags.csv")

# =========================
# 5. Show basic info
# =========================
print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)
print("Tags shape:", tags.shape)

print("\nMovies columns:", movies.columns.tolist())
print("Ratings columns:", ratings.columns.tolist())
print("Tags columns:", tags.columns.tolist())

# =========================
# 6. Remove duplicates
# =========================
movies = movies.drop_duplicates()
ratings = ratings.drop_duplicates()
tags = tags.drop_duplicates()

# =========================
# 7. Clean movies data
# =========================
movies["title"] = movies["title"].astype(str).str.strip()

# Extract release year from title like "Toy Story (1995)"
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

# Clean genres
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].replace("(no genres listed)", "", regex=False)
movies["genres_list"] = movies["genres"].apply(lambda x: x.split("|") if x else [])

# =========================
# 8. Clean tags data
# =========================
tags["tag"] = tags["tag"].fillna("").astype(str).str.strip().str.lower()

tag_grouped = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"tag": "all_tags"})
)

# =========================
# 9. Aggregate ratings
# =========================
ratings_summary = (
    ratings.groupby("movieId")
    .agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    )
    .reset_index()
)

# Optional: user-level stats too
user_summary = (
    ratings.groupby("userId")
    .agg(
        ratings_given=("rating", "count"),
        user_avg_rating=("rating", "mean")
    )
    .reset_index()
)

# =========================
# 10. Merge into one movie table
# =========================
movie_data = movies.merge(ratings_summary, on="movieId", how="left")
movie_data = movie_data.merge(tag_grouped, on="movieId", how="left")

# Fill missing values after merge
movie_data["avg_rating"] = movie_data["avg_rating"].fillna(0)
movie_data["num_ratings"] = movie_data["num_ratings"].fillna(0).astype(int)
movie_data["all_tags"] = movie_data["all_tags"].fillna("")

# =========================
# 11. Create metadata column
# Useful for TF-IDF / content-based model later
# =========================
movie_data["metadata"] = (
    movie_data["title"].fillna("") + " " +
    movie_data["genres"].fillna("") + " " +
    movie_data["all_tags"].fillna("")
).str.lower()

# =========================
# 12. Optional popularity filtering
# =========================
popular_movies = movie_data[movie_data["num_ratings"] >= 20].copy()

# =========================
# 13. Save processed files back to Drive
# =========================
output_path = f"{data_path}/processed_output"
os.makedirs(output_path, exist_ok=True)

movie_data.to_csv(f"{output_path}/processed_movies.csv", index=False)
popular_movies.to_csv(f"{output_path}/popular_movies.csv", index=False)
ratings_summary.to_csv(f"{output_path}/ratings_summary.csv", index=False)
user_summary.to_csv(f"{output_path}/user_summary.csv", index=False)

# =========================
# 14. Preview output
# =========================
print("\nProcessed movie_data shape:", movie_data.shape)
print("Popular movies shape:", popular_movies.shape)

display(movie_data.head())
display(popular_movies.head())

print(f"\nSaved files in: {output_path}")
