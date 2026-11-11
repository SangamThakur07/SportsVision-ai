"""
SportsVision AI: Test Video Acquisition Script
Downloads open-source broadcast football match footage from DFL Bundesliga computer vision datasets.
"""

import os
import gdown

DATASETS = {
    "bundesliga_match_1.mp4": {
        "url": "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
        "description": "DFL Bundesliga Broadcast Match Clip 1 (1920x1080 @ 25fps, 30s)"
    },
    "bundesliga_match_2.mp4": {
        "url": "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
        "description": "DFL Bundesliga Broadcast Match Clip 2 (1920x1080 @ 25fps, 30s)"
    },
    "bundesliga_match_3.mp4": {
        "url": "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
        "description": "DFL Bundesliga Broadcast Match Clip 3 (1920x1080 @ 25fps, 30s)"
    }
}


def download_test_videos(target_dir: str = "data/sample"):
    os.makedirs(target_dir, exist_ok=True)
    print(f"Acquiring sports video test assets into {target_dir}...")
    for filename, info in DATASETS.items():
        dest = os.path.join(target_dir, filename)
        if os.path.exists(dest):
            print(f"[EXISTS] {filename} ({info['description']})")
        else:
            print(f"[DOWNLOADING] {filename} ({info['description']})...")
            gdown.download(info["url"], dest, quiet=False)
            print(f"[DONE] Saved to {dest}")


if __name__ == "__main__":
    download_test_videos()
