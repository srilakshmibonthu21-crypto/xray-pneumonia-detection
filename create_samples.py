"""
Generate synthetic sample X-rays for immediate testing without external dataset download.
"""
from src.dataset import create_synthetic_sample_data

if __name__ == "__main__":
    create_synthetic_sample_data()
