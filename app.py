# app.py
import streamlit as st
from fetch_comments import fetch_comments_for_videos
from analysis_core import analyze_comments, cluster_summary
import pandas as pd

st.set_page_config(page_title="YouTube Comment Insights", layout="wide")
st.title("YouTube Comment Insights — Phase 1 + Phase 2 (Free)")

with st.sidebar:
    st.header("Inputs")
    api_key = st.text_input("YouTube API key", type="password")
    videos_text = st.text_area("Video URLs or IDs (one per line)", value="")
    max_comments = st.slider("Max comments per video", 50, 2000, 500, 50)
    run_button = st.button("Run analysis")

if run_button:
    if not api_key:
        st.error("Please provide your YouTube API key in the sidebar.")
    else:
        video_lines = [v.strip() for v in videos_text.splitlines() if v.strip()]
        if not video_lines:
            st.error("Provide at least one video URL or ID.")
        else:
            with st.spinner("Fetching comments..."):
                comments = fetch_comments_for_videos(video_lines, api_key, max_comments_per_video=max_comments)
            st.success(f"Fetched {len(comments)} comments (including replies).")

            # remove comments without 'text' or empty text
            comments = [c for c in comments if 'text' in c and c['text'].strip() != '']

            if not comments:
                st.error("No valid comments to analyze.")
            else:
                with st.spinner("Analyzing comments (embeddings, clusters, sentiment)..."):
                    df_all, kw = analyze_comments(comments)
                st.success("Analysis complete.")

                summaries = cluster_summary(df_all, kw, top_examples=3)
                st.header("Cluster Summaries")
                for s in summaries:
                    st.subheader(f"Cluster {s['cluster_id']} — {s['count']} comments")
                    st.write("Keywords:", ", ".join(s['keywords']))
                    st.write("Sentiment:", s['sentiment'])
                    st.write("Examples:")
                    for ex in s['examples']:
                        st.write("- ", ex)

                st.header("Cross-video cluster counts")
                if summaries:
                   for cluster_id, summary in enumerate(summaries):
                       st.subheader(f"Cluster {cluster_id + 1} — Keywords: {', '.join(summary['keywords'])}")
                       st.write(f"📊 Sentiment → {summary['sentiment']}")
        
        # Bar chart: count of comments in this cluster per video
                       cluster_counts = df_all[df_all['cluster'] == cluster_id].groupby('video_id')['comment_id'].count()
        
        # Clean video labels (titles if you fetch them; else IDs)
                       cluster_counts.index = [video_titles.get(vid, vid) for vid in cluster_counts.index]  
        
                      st.bar_chart(cluster_counts)
                else:
                  st.warning("No clusters available yet. Try analyzing more comments.")

