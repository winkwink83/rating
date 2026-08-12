from pathlib import Path

import numpy as np
from joblib import dump

from dataset import load_ratings_data

from sklearn.ensemble import RandomForestRegressor


EMBEDDINGS_FILE = Path("data/embeddings_cache.npz")
MODEL_FILE = Path("data/models/photo_rating_model.joblib")


def load_embeddings_cache():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku {EMBEDDINGS_FILE}. "
            f"Najpierw uruchom: python src/build_embeddings.py"
        )

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)

    filenames = data["filenames"]
    embeddings = data["embeddings"]

    embeddings_by_filename = {}

    for filename, embedding in zip(filenames, embeddings):
        embeddings_by_filename[str(filename)] = embedding

    return embeddings_by_filename


def prepare_training_data():
    ratings_data = load_ratings_data()
    embeddings_by_filename = load_embeddings_cache()

    X = []
    y = []

    missing_embeddings = []

    for item in ratings_data:
        image_name = item["image_path"].name

        if image_name not in embeddings_by_filename:
            missing_embeddings.append(image_name)
            continue

        X.append(embeddings_by_filename[image_name])
        y.append(item["rating"])

    if missing_embeddings:
        print(f"Brak embeddingów dla {len(missing_embeddings)} zdjęć.")
        print("Uruchom ponownie: python src/build_embeddings.py")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def train_final_model():
    X, y = prepare_training_data()

    if len(X) < 10:
        print("Za mało danych treningowych. Zbierz przynajmniej 10 ocen.")
        return

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X, y)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    dump(model, MODEL_FILE)

    print("Final model trained successfully.")
    print(f"Training samples: {len(X)}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_final_model()