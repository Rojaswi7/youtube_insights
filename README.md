# YouTube Comment Insights Agent

An AI-powered Streamlit app to analyze YouTube video comments:
- Fetches comments via YouTube Data API v3
- Cleans + detects language
- Embeds using SentenceTransformers
- Clusters via KMeans
- Extracts top keywords per cluster
- Visualizes cross-video comment clusters with charts

## 🚀 Run locally

```bash
git clone https://github.com/<your-username>/youtube_insights.git
cd youtube_insights
pip install -r requirements.txt
streamlit run app.py
