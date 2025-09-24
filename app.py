import streamlit as st
import pandas as pd

from fetch_comments import fetch_comments_for_videos, fetch_video_title
from analysis_core import analyze_comments

st.set_page_config(page_title="YouTube Comment Insights", layout="wide")

st.title("📊 YouTube Comment Insights Agent")

video_input = st.text_area("Enter YouTube video IDs (comma-separated):")
video_ids = [v.strip() for v in video_input.split(",") if v.strip()]

if st.button("Analyze") and video_ids:
    with st.spinner("Fetching comments..."):
        comments = fetch_comments_for_videos(video_ids, max_results=50)
        video_titles = {vid: fetch_video_title(vid) for vid in video_ids}

    st.success(f"Fetched {len(comments)} comments ✅")

    if comments:
        df_all, summaries = analyze_comments(comments)

        st.header("Raw Comments")
        st.dataframe(df_all[["video_id", "text", "author", "like_count"]])

        # Cross-video Cluster Insights
        st.header("Cross-video Cluster Insights")

        if summaries:
            for cluster_id, summary in enumerate(summaries):
                st.subheader(f"Cluster {cluster_id + 1} — Keywords: {', '.join(summary['keywords'])}")
                cluster_counts = df_all[df_all["cluster"] == cluster_id].groupby("video_id")["comment_id"].count()

                # replace video_id with title
                cluster_counts.index = [video_titles.get(vid, vid) for vid in cluster_counts.index]

                st.bar_chart(cluster_counts)

        else:
            st.warning("No clusters created yet. Try with more comments.")
