import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import load

from features import extract_image_features


TEST_IMAGES_DIR = Path(
    r"C:\Users\macie\Desktop\Projekty\photo_rating_system\images\tests_images"
)

TEST_RATINGS_FILE = Path(
    r"C:\Users\macie\Desktop\Projekty\photo_rating_system\data\blind_tests\test_ratings.csv"
)

MODEL_FILE = Path("data/models/photo_rating_model.joblib")

RESULTS_DIR = Path("data/blind_tests")
RESULTS_CSV_FILE = RESULTS_DIR / "blind_test_results.csv"


def load_test_ratings():
    if not TEST_RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {TEST_RATINGS_FILE}"
        )

    ratings = []

    with open(TEST_RATINGS_FILE, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            ratings.append(
                {
                    "filename": row["filename"],
                    "true_rating": float(row["rating"]),
                }
            )

    return ratings


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_FILE}\n"
            f"Najpierw uruchom: python src/train.py"
        )

    return load(MODEL_FILE)


def predict_image(model, image_path):
    features = extract_image_features(image_path)

    X = np.array([features], dtype=np.float32)

    prediction = model.predict(X)[0]
    prediction = float(np.clip(prediction, 1.0, 10.0))

    return prediction


def save_results(results):
    with open(RESULTS_CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "filename",
                "true_rating",
                "prediction",
                "absolute_error",
            ]
        )

        for item in results:
            writer.writerow(
                [
                    item["filename"],
                    f"{item['true_rating']:.2f}",
                    f"{item['prediction']:.2f}",
                    f"{item['absolute_error']:.2f}",
                ]
            )


def create_report(results):
    true_ratings = np.array(
        [item["true_rating"] for item in results],
        dtype=np.float32,
    )

    predictions = np.array(
        [item["prediction"] for item in results],
        dtype=np.float32,
    )

    errors = np.abs(true_ratings - predictions)

    mae = np.mean(errors)
    mse = np.mean((true_ratings - predictions) ** 2)
    rmse = np.sqrt(mse)

    baseline_rating = np.mean(true_ratings)
    baseline_predictions = np.full_like(true_ratings, baseline_rating)
    baseline_mae = np.mean(np.abs(true_ratings - baseline_predictions))

    biggest_mistake = max(
        results,
        key=lambda item: item["absolute_error"],
    )

    smallest_mistake = min(
        results,
        key=lambda item: item["absolute_error"],
    )

    report = f"""
BLIND TEST REPORT
==============================

Timestamp:
{datetime.now()}

Dataset:
Test images directory: {TEST_IMAGES_DIR}
Ratings file: {TEST_RATINGS_FILE}
Images tested: {len(results)}

Features:
Generated live from test images using CLIP

Model:
Loaded from: {MODEL_FILE}

Metrics:
MAE: {mae:.2f}
MSE: {mse:.2f}
RMSE: {rmse:.2f}

Baseline:
Baseline MAE: {baseline_mae:.2f}

Biggest mistake:
Filename: {biggest_mistake["filename"]}
True rating: {biggest_mistake["true_rating"]:.2f}
Predicted rating: {biggest_mistake["prediction"]:.2f}
Absolute error: {biggest_mistake["absolute_error"]:.2f}

Smallest mistake:
Filename: {smallest_mistake["filename"]}
True rating: {smallest_mistake["true_rating"]:.2f}
Predicted rating: {smallest_mistake["prediction"]:.2f}
Absolute error: {smallest_mistake["absolute_error"]:.2f}

Interpretation:
MAE mówi, o ile punktów średnio myli się model.
RMSE mocniej karze duże błędy.
Baseline MAE pokazuje wynik prostego zgadywania średniej oceny.
"""

    return report


def save_distribution_plot(results, plot_file):
    true_ratings = [
        int(round(item["true_rating"]))
        for item in results
    ]

    predicted_ratings = [
        int(np.clip(round(item["prediction"]), 1, 10))
        for item in results
    ]

    true_counter = Counter(true_ratings)
    predicted_counter = Counter(predicted_ratings)

    ratings = list(range(1, 11))

    true_counts = [
        true_counter.get(rating, 0)
        for rating in ratings
    ]

    predicted_counts = [
        predicted_counter.get(rating, 0)
        for rating in ratings
    ]

    x = np.arange(len(ratings))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        true_counts,
        width,
        label="Your ratings",
    )

    plt.bar(
        x + width / 2,
        predicted_counts,
        width,
        label="Model ratings",
    )

    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Blind test rating distribution")
    plt.xticks(x, ratings)
    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_file)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    ratings = load_test_ratings()

    results = []

    for item in ratings:
        image_path = TEST_IMAGES_DIR / item["filename"]

        if not image_path.exists():
            print(f"Pominięto, brak pliku: {image_path}")
            continue

        prediction = predict_image(model, image_path)
        absolute_error = abs(item["true_rating"] - prediction)

        results.append(
            {
                "filename": item["filename"],
                "true_rating": item["true_rating"],
                "prediction": prediction,
                "absolute_error": absolute_error,
            }
        )

    if not results:
        print("Brak wyników do zapisania.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_file = RESULTS_DIR / f"blind_test_report_{timestamp}.txt"
    plot_file = RESULTS_DIR / f"blind_test_distribution_{timestamp}.png"

    save_results(results)
    save_distribution_plot(results, plot_file)

    report = create_report(results)
    report_file.write_text(report, encoding="utf-8")

    print()
    print("DONE")
    print(f"Results saved to: {RESULTS_CSV_FILE}")
    print(f"Report saved to: {report_file}")
    print(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()