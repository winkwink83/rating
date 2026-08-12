from pathlib import Path

import numpy as np

from dataset import load_ratings_data
from features import extract_image_features


EMBEDDINGS_FILE = Path("data/embeddings_cache.npz")


def build_embeddings_cache():
    ratings_data = load_ratings_data()

    if not ratings_data:
        print("Brak ocen w data/ratings.csv")
        return

    filenames = []
    embeddings = []

    total = len(ratings_data)

    for index, item in enumerate(ratings_data, start=1):
        image_path = item["image_path"]

        print(f"[{index}/{total}] Processing: {image_path.name}")

        features = extract_image_features(image_path)

        filenames.append(image_path.name)
        embeddings.append(features)

    embeddings = np.array(embeddings, dtype=np.float32)
    filenames = np.array(filenames)

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        EMBEDDINGS_FILE,
        filenames=filenames,
        embeddings=embeddings,
    )

    print()
    print(f"Saved embeddings cache to: {EMBEDDINGS_FILE}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    build_embeddings_cache()