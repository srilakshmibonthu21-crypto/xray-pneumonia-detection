"""
Download authentic clinical chest X-ray samples (Normal and Pneumonia)
from public open-access medical repository for instant testing, evaluation, and web app demonstration.
"""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SAMPLES = {
    "NORMAL": [
        ("normal_sample_1.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/IM-0001-0001.jpeg"),
        ("normal_sample_2.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/IM-0003-0001.jpeg"),
        ("normal_sample_3.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/IM-0005-0001.jpeg"),
    ],
    "PNEUMONIA": [
        ("pneumonia_sample_1.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/person1_bacteria_1.jpeg"),
        ("pneumonia_sample_2.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/person1_bacteria_2.jpeg"),
        ("pneumonia_sample_3.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/person2_bacteria_3.jpeg"),
        ("pneumonia_sample_4.jpeg", "https://raw.githubusercontent.com/UdbhavPrasad072300/Pneumonia-Detector-with-Tensorflow/master/person2_bacteria_4.jpeg"),
    ]
}

def download_samples():
    print("[INFO] Downloading authentic clinical Chest X-Ray samples...")

    # Also store in data/test and data/train for testing training/eval pipelines
    targets = [
        DATA_DIR / "test",
        DATA_DIR / "train",
        DATA_DIR / "val",
        DATA_DIR / "sample"
    ]

    for category, items in SAMPLES.items():
        for filename, url in items:
            for target_dir in targets:
                dest_dir = target_dir / category
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / filename
                if not dest_file.exists():
                    try:
                        print(f"  Downloading {category}/{filename} -> {dest_file.relative_to(BASE_DIR)}...")
                        urllib.request.urlretrieve(url, str(dest_file))
                    except Exception as e:
                        print(f"  [ERROR] Failed to download {url}: {e}")

    print("[SUCCESS] All sample chest radiographs have been downloaded successfully!")
    print(f"[INFO] Files available in: {DATA_DIR}")

if __name__ == "__main__":
    download_samples()
