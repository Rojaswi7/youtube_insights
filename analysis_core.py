# analysis_core.py
import re
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter
import nltk
nltk.download('stopwords')

DetectorFactory.seed = 0

# ---- Text cleaning ----
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language_safe(text):
    try:
        if not text or len(text.strip()) < 3:
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"

def is_spam_heuristic(text):
    if not text or len(text.strip()) == 0:
        return True
    if len(re.findall(r'http\S+|www\.\S+', text)) >= 1:
        return True
    words = text.split()
    if len(words) < 3:
        return True
    if re.search(r'(.)\1\1\1', text):
        return True
    punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text))
    if punct_ratio > 0.25:
        return True
    return False

# ---- Embeddings & clustering ----
EMBED_MODEL = None
def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer(model_name)
    return EMBED_MODEL

def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def choose_k(n_comments):
    if n_comments < 30:
        return max(2, min(3, n_comments))
    return min(10, max(3, n_comments // 50 + 2))

def cluster_embeddings(embeddings, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

# ---- Keywords ----
def top_keywords_by_cluster(texts, labels, top_n=6):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words=stopwords.words('english'))
    X = vec.fit_transform(texts)
    feature_names = np.array(vec.get_feature_names_out())
    clusters = {}
    for cluster_id in sorted(set(labels)):
        idxs = np.where(labels == cluster_id)[0]
        submat = X[idxs]
        tfidf_sum = np.asarray(submat.sum(axis=0)).ravel()
        top_idx = tfidf_sum.argsort()[::-1][:top_n]
        clusters[cluster_id] = feature_names[top_idx].tolist()
    return clusters

# ---- Sentiment ----
SENT_PIPE = None
def get_sentiment_pipeline(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    global SENT_PIPE
    if SENT_PIPE is None:
        SENT_PIPE = pipeline("sentiment-analysis", model=model_name, device=-1)
    return SENT_PIPE

def run_sentiment(texts):
    pipe = get_sentiment_pipeline()
    results = pipe(texts, truncation=True)
    normalized = []
    for r in results:
        lab = r.get('label', '').lower()
        m = re.search(r'(\d)', lab)
        if m:
            val = int(m.group(1))
            if val <= 2:
                normalized.append('negative')
            elif val == 3:
                normalized.append('neutral')
            else:
                normalized.append('positive')
        else:
            score = r.get('score', 0.0)
            if score >= 0.6:
                normalized.append('positive')
            elif score >= 0.4:
                normalized.append('neutral')
            else:
                normalized.append('negative')
    return normalized

# ---- Main pipeline ----
def analyze_comments(comments_list):
    df = pd.DataFrame(comments_list)
    if 'text' not in df.columns:
        raise KeyError("Each comment dict must contain a 'text' key")
    df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
    df['lang'] = df['cleaned_text'].apply(detect_language_safe)
    df['is_spam'] = df['cleaned_text'].apply(is_spam_heuristic)

    df_for_embedding = df[~df['is_spam']].reset_index(drop=True)
    texts = df_for_embedding['cleaned_text'].tolist()
    if len(texts) == 0:
        raise ValueError("No non-spam comments to analyze.")

    embeddings = compute_embeddings(texts)
    k = choose_k(len(texts))
    labels, km = cluster_embeddings(embeddings, n_clusters=k)
    sentiments = run_sentiment(texts)

    df_for_embedding['cluster'] = labels
    df_for_embedding['sentiment'] = sentiments

    kw = top_keywords_by_cluster(df_for_embedding['cleaned_text'].tolist(), labels)
    df_all = df.merge(df_for_embedding[['comment_id','cluster','sentiment']], on='comment_id', how='left')
    return df_all, kw

def cluster_summary(df_all, kw_dict, top_examples=3):
    summaries = []
    df_ok = df_all[~df_all['is_spam']].copy()
    for c in sorted(df_ok['cluster'].dropna().unique()):
        c = int(c)
        sub = df_ok[df_ok['cluster'] == c]
        count = len(sub)
        sentiments = Counter(sub['sentiment'].fillna('unknown').tolist())
        examples = sub['cleaned_text'].dropna().unique()[:top_examples].tolist()
        summaries.append({
            'cluster_id': c,
            'count': count,
            'keywords': kw_dict.get(c, []),
            'sentiment': dict(sentiments),
            'examples': examples
        })
    return summaries
