from pathlib import Path
import csv


IMAGES_DIR = Path("images")
RATINGS_FILE = Path("data/ratings.csv")


def get_image_files():
    image_extensions = [".jpg", ".jpeg", ".png"]

    image_files = []

    for file_path in IMAGES_DIR.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    image_files.sort()

    return image_files


def get_rated_images():
    rated_images = set()

    if not RATINGS_FILE.exists():
        return rated_images

    with open(RATINGS_FILE, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rated_images.add(row["image_name"])

    return rated_images


def get_unrated_image_files():
    image_files = get_image_files()

    rated_images = get_rated_images()

    unrated_images = []

    for image_path in image_files:
        if image_path.name not in rated_images:
            unrated_images.append(image_path)

    return unrated_images


def load_ratings_data():
    ratings_data = []

    if not RATINGS_FILE.exists():
        return ratings_data

    with open(RATINGS_FILE, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = IMAGES_DIR / row["image_name"]

            if image_path.exists():
                ratings_data.append(
                    {
                        "image_path": image_path,
                        "rating": float(row["rating"]),
                    }
                )

    return ratings_data


def count_images():
    return len(get_image_files())


def count_ratings():
    rated_images = get_rated_images()

    return len(rated_images)