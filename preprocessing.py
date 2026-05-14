# =========================
# FINAL INTERACTIVE MOVIE RECOMMENDER
# Content-Based Recommender with:
# - TF-IDF + Cosine Similarity
# - Text input + Search button
# - Nearest title suggestions as clickable buttons
# - Clean output table
# - Outputs saved strictly to output_path in Google Drive
# =========================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Imports
import os
import pandas as pd
import numpy as np
from difflib import get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from google.colab import data_table

# Enable interactive dataframe display in Colab
data_table.enable_dataframe_formatter()

# 3. Paths
data_path = "/content/drive/MyDrive/movie_archive"
output_path = f"{data_path}/processed_output"
os.makedirs(output_path, exist_ok=True)

# 4. Load processed movie data
movie_data = pd.read_csv(f"{output_path}/processed_movies.csv")

# 5. Safety checks
required_cols = ["movieId", "title", "genres", "avg_rating", "num_ratings", "metadata"]
missing_cols = [col for col in required_cols if col not in movie_data.columns]

if missing_cols:
    raise ValueError(f"Missing required columns in processed_movies.csv: {missing_cols}")

# 6. Clean data
recommender_df = movie_data.copy()

recommender_df["title"] = recommender_df["title"].fillna("").astype(str).str.strip()
recommender_df["genres"] = recommender_df["genres"].fillna("").astype(str).str.strip()
recommender_df["metadata"] = recommender_df["metadata"].fillna("").astype(str).str.strip()

recommender_df["avg_rating"] = pd.to_numeric(recommender_df["avg_rating"], errors="coerce").fillna(0.0)
recommender_df["num_ratings"] = pd.to_numeric(recommender_df["num_ratings"], errors="coerce").fillna(0).astype(int)

recommender_df = recommender_df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
recommender_df = recommender_df[recommender_df["metadata"] != ""].reset_index(drop=True)

print("Recommender dataframe shape:", recommender_df.shape)

# 7. TF-IDF vectorization
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

tfidf_matrix = tfidf.fit_transform(recommender_df["metadata"])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 8. Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 9. Title mapping
title_to_index = pd.Series(recommender_df.index, index=recommender_df["title"]).drop_duplicates()
all_titles = recommender_df["title"].dropna().astype(str).tolist()

# 10. Save reusable files strictly to output_path
recommender_df.to_csv(f"{output_path}/recommender_base_data.csv", index=False)
recommender_df[["movieId", "title", "genres", "avg_rating", "num_ratings"]].to_csv(
    f"{output_path}/movie_lookup.csv", index=False
)

# 11. Helper functions
def find_closest_titles(user_input, n=5, cutoff=0.5):
    return get_close_matches(user_input, all_titles, n=n, cutoff=cutoff)

def get_recommendations_from_title(matched_title, top_n=10, min_ratings=20):
    if matched_title not in title_to_index:
        return pd.DataFrame(columns=[
            "searched_title",
            "matched_title",
            "recommended_title",
            "similarity_score",
            "genres",
            "avg_rating",
            "num_ratings"
        ])

    idx = title_to_index[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie_idx, score in sim_scores[1:]:
        row = recommender_df.iloc[movie_idx]

        if row["num_ratings"] < min_ratings:
            continue

        recommendations.append({
            "searched_title": search_box.value.strip(),
            "matched_title": matched_title,
            "recommended_title": row["title"],
            "similarity_score": round(float(score), 4),
            "genres": row["genres"],
            "avg_rating": round(float(row["avg_rating"]), 2),
            "num_ratings": int(row["num_ratings"])
        })

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)

def format_and_display(df, matched_title):
    with output_area:
        clear_output(wait=True)

        if df.empty:
            display(HTML("<b>No recommendations found.</b>"))
            return

        save_path = f"{output_path}/user_movie_recommendations.csv"
        df.to_csv(save_path, index=False)

        clean_df = df[[
            "recommended_title",
            "similarity_score",
            "genres",
            "avg_rating",
            "num_ratings"
        ]].copy()

        clean_df.columns = [
            "Recommended Movie",
            "Similarity",
            "Genres",
            "Avg Rating",
            "Rating Count"
        ]

        clean_df.index = range(1, len(clean_df) + 1)

        display(HTML(f"""
        <div style="padding:10px 0;">
            <h3 style="margin:0;">Movie Recommender Results</h3>
            <p style="margin:6px 0;"><b>Your search:</b> {search_box.value.strip()}</p>
            <p style="margin:6px 0;"><b>Matched title:</b> {matched_title}</p>
            <p style="margin:6px 0; color:green;"><b>Saved to:</b> {save_path}</p>
        </div>
        """))

        display(
            clean_df.style
            .format({
                "Similarity": "{:.3f}",
                "Avg Rating": "{:.2f}",
                "Rating Count": "{:,}"
            })
            .set_properties(**{
                "text-align": "left",
                "white-space": "normal"
            })
            .set_caption("Top 10 Recommended Movies")
        )

def handle_exact_or_best_match(selected_title):
    recommendations_df = get_recommendations_from_title(
        matched_title=selected_title,
        top_n=10,
        min_ratings=20
    )
    format_and_display(recommendations_df, selected_title)

def on_suggestion_click(title):
    def _handler(button):
        handle_exact_or_best_match(title)
    return _handler

def search_movie(_):
    query = search_box.value.strip()

    with output_area:
        clear_output(wait=True)

    with suggestion_area:
        clear_output(wait=True)

    if not query:
        with output_area:
            display(HTML("<b>Please enter a movie name.</b>"))
        return

    # Exact match
    if query in title_to_index:
        handle_exact_or_best_match(query)
        return

    # Fuzzy matches
    matches = find_closest_titles(query, n=5, cutoff=0.5)

    if not matches:
        with output_area:
            display(HTML(f"<b>No close matches found for:</b> {query}"))
        return

    with suggestion_area:
        display(HTML("<b>Select the closest movie title:</b>"))
        buttons = []
        for title in matches:
            btn = widgets.Button(
                description=title[:70],
                layout=widgets.Layout(width='auto'),
                button_style='info'
            )
            btn.on_click(on_suggestion_click(title))
            buttons.append(btn)

        display(widgets.VBox(buttons))

# 12. Widgets
title_html = widgets.HTML("<h2>Interactive Movie Recommender</h2>")
subtitle_html = widgets.HTML(
    "<p>Type a movie name. If it doesn't match exactly, nearest titles will appear as clickable buttons.</p>"
)

search_box = widgets.Text(
    value="",
    placeholder="Enter movie name, e.g. django unchained",
    description="Movie:",
    layout=widgets.Layout(width='600px')
)

search_button = widgets.Button(
    description="Search Recommendations",
    button_style="success",
    icon="search",
    layout=widgets.Layout(width='220px')
)

suggestion_area = widgets.Output()
output_area = widgets.Output()

search_button.on_click(search_movie)

# 13. Display UI
display(title_html, subtitle_html, widgets.HBox([search_box, search_button]), suggestion_area, output_area)
