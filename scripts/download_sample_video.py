"""
Sample Football Video Downloader
Downloads a standard broadcast football clip for pipeline testing and demonstration.
"""

import os
import sys
import urllib.request


SAMPLE_URL = "https://media.roboflow.com/sports/football-ai.mp4"
DEFAULT_TARGET = "data/sample/football_match.mp4"


def download_sample_video(target_path: str = DEFAULT_TARGET, url: str = SAMPLE_URL) -> str:
    """Download sample broadcast football match clip."""
    target_path = os.path.abspath(target_path)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"Sample video already exists: {target_path} ({os.path.getsize(target_path) / 1024 / 1024:.2f} MB)")
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"Downloading sample football video from:\n  {url}\nTo:\n  {target_path}...")

    def progress_callback(blocks, block_size, total_size):
        downloaded = blocks * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rDownload progress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")
        else:
            sys.stdout.write(f"\rDownloaded: {downloaded / 1024 / 1024:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=progress_callback)
        print("\nDownload completed successfully!")
        return target_path
    except Exception as e:
        print(f"\nFailed to download from {url}: {e}")
        return None


if __name__ == "__main__":
    download_sample_video()
