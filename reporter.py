# reporter.py
import pandas as pd
from collections import defaultdict

def export_csv(df_all, out_path="youtube_comments_analysis.csv"):
    df_all.to_csv(out_path, index=False)
    print(f"Exported CSV to {out_path}")

def cross_video_cluster_matrix(df_all):
    """
    Assumes df_all has columns video_id, cluster (clusters are integers or NaN)
    Returns a pivot dataframe: rows=cluster, cols=video_id, values=counts
    """
    df = df_all[~df_all['is_spam']].copy()
    pivot = pd.pivot_table(df, index='cluster', columns='video_id', values='comment_id', aggfunc='count', fill_value=0)
    pivot = pivot.sort_index()
    return pivot

def compute_trends(pivot_df):
    """
    For each cluster, compute percent change from first video to last video
    (Works when columns are ordered by user-provided order)
    """
    cols = list(pivot_df.columns)
    if len(cols) < 2:
        return {}
    first = cols[0]
    last = cols[-1]
    trends = {}
    for cluster in pivot_df.index:
        a = pivot_df.loc[cluster, first]
        b = pivot_df.loc[cluster, last]
        if a == 0:
            # if first is 0, compute absolute or mark as new
            if b == 0:
                pct = 0.0
            else:
                pct = float('inf')  # indicates new cluster growth from zero
        else:
            pct = (b - a) / a * 100.0
        trends[cluster] = {'first': int(a), 'last': int(b), 'pct_change': pct}
    return trends
