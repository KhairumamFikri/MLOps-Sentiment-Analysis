from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
from datetime import datetime
import os

RAW_DIR = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=neJ1pOBFuN0",
    "https://www.youtube.com/watch?v=a2WXt0aW76g",
    "https://www.youtube.com/watch?v=BSZ0M9uFBDs"
]

downloader = YoutubeCommentDownloader()

comments = []

for video_url in VIDEO_URLS:

    print(f"Mengambil komentar dari: {video_url}")

    try:

        for comment in downloader.get_comments_from_url(video_url):

            comments.append({
                "author": comment["author"],
                "text": comment["text"],
                "time": comment["time"],
                "votes": comment["votes"]
            })

    except Exception as e:
        print(f"Error pada video: {video_url}")
        print(e)

df = pd.DataFrame(comments)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"{RAW_DIR}/youtube_comments_{timestamp}.csv"

df.to_csv(filename, index=False)

print(f"Data berhasil disimpan ke: {filename}")
print(f"Jumlah komentar: {len(df)}")