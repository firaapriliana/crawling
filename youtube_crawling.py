import requests
import re
import pandas as pd
from googleapiclient.discovery import build

class YoutubeCrawling:
    def __init__(self, api_key: str):
        """Inisialisasi YouTube API."""
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def extract_video_id(self, url: str) -> str:
        """Mengekstrak video ID dari URL YouTube."""
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

    def get_comments(self, video_url: str, max_comments: int = 100, order="relevance"):
        """Mengambil komentar dari video YouTube."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("URL YouTube tidak valid.")

        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            response = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(100, max_comments - len(comments)),
                order=order,
                pageToken=next_page_token
            ).execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # Hentikan jika tidak ada komentar lagi

        return comments