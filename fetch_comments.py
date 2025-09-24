import os
from googleapiclient.discovery import build
from tqdm import tqdm

API_KEY = os.getenv("YOUTUBE_API_KEY")  # keep your key in Streamlit Secrets
youtube = build("youtube", "v3", developerKey=API_KEY)


def fetch_comments_for_videos(video_ids, max_results=100):
    """Fetch comments for a list of YouTube video IDs."""
    all_comments = []
    for vid in tqdm(video_ids, desc="Fetching videos"):
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=vid,
            maxResults=min(max_results, 100),
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_text = snippet.get("textDisplay", "")
            if comment_text:
                all_comments.append(
                    {
                        "video_id": vid,
                        "comment_id": item["id"],
                        "text": comment_text,
                        "author": snippet.get("authorDisplayName", ""),
                        "like_count": snippet.get("likeCount", 0),
                    }
                )
    return all_comments


def fetch_video_title(video_id: str) -> str:
    """Fetch the title of a YouTube video given its ID."""
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return video_id
    return items[0]["snippet"]["title"]
