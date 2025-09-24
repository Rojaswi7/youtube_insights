import re
import nltk
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from langdetect import detect

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def analyze_comments(comments, n_clusters=3):
    """Analyze comments: clean, embed, cluster, extract keywords."""
    if not comments:
        return pd.DataFrame(), []

    df = pd.DataFrame(comments)

    if "text" not in df.columns:
        raise KeyError("Each comment dict must contain a 'text' key")

    # clean + language detection
    df["cleaned_text"] = df["text"].astype(str).apply(clean_text)
    df["language"] = df["text"].apply(lambda x: safe_detect(x))

    # embeddings
    embeddings = embedder.encode(df["cleaned_text"].tolist(), show_progress_bar=False)

    # cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    df["cluster"] = clusters

    # extract keywords per cluster
    vectorizer = CountVectorizer(max_features=20)
    X = vectorizer.fit_transform(df["cleaned_text"])
    terms = vectorizer.get_feature_names_out()

    cluster_keywords = []
    for cluster_id in range(n_clusters):
        cluster_texts = df[df["cluster"] == cluster_id]["cleaned_text"].tolist()
        if cluster_texts:
            cluster_vectorizer = CountVectorizer(stop_words="english", max_features=5)
            cluster_X = cluster_vectorizer.fit_transform(cluster_texts)
            kws = cluster_vectorizer.get_feature_names_out().tolist()
        else:
            kws = []
        cluster_keywords.append({"cluster": cluster_id, "keywords": kws})

    return df, cluster_keywords


def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"
