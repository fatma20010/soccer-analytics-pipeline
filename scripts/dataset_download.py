"""Utility to guide user to acquire a small public football dataset segment.
We do not redistribute clips directly to respect licenses.

Suggested sources:
1. SoccerNet (https://www.soccer-net.org/) -> register and download a small snippet (games trimmed around events). Use their "mini" subset.
2. YouTube public highlight clips (ensure personal / educational use) -> download with yt-dlp (respect ToS).

This script will optionally download a CC sample of a generic green field video for pipeline smoke test.
"""
import os
import urllib.request

SAMPLE_URL = 'https://raw.githubusercontent.com/roboflow-ai/notebooks/main/data/football_demo/frame_0001.jpg'
SAMPLE_DEST = 'data/sample_frame.jpg'

def download_sample():
    os.makedirs('data', exist_ok=True)
    print('Downloading sample frame...')
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_DEST)
    print('Saved to', SAMPLE_DEST)

if __name__ == '__main__':
    download_sample()
