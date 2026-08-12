import csv
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


TEST_IMAGES_DIR = Path(
    r"C:\Users\macie\Desktop\Projekty\photo_rating_system\images\tests_images"
)

OUTPUT_FILE = Path("data/blind_tests/test_ratings.csv")

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

MAX_IMAGE_WIDTH = 1000
MAX_IMAGE_HEIGHT = 700


class TestRatingCollector:
    def __init__(self, root):
        self.root = root

        self.root.title("Test Rating Collector")

        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)

        self.image_files = self.get_test_images()

        self.current_index = 0
        self.current_photo = None

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        self.results = []

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        self.image_label = tk.Label(main_frame)
        self.image_label.pack(pady=20, expand=True)

        self.info_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 12),
        )
        self.info_label.pack(pady=10)

        rating_frame = tk.Frame(main_frame)
        rating_frame.pack(pady=10)

        for rating in range(1, 11):
            button = tk.Button(
                rating_frame,
                text=str(rating),
                width=4,
                height=2,
                command=lambda value=rating: self.save_rating(value),
            )
            button.pack(side=tk.LEFT, padx=3)

        self.show_current_image()

    def get_test_images(self):
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".gif",
            ".tiff",
            ".tif",
            ".jfif",
            ".heic",
            ".heif",
            ".avif",
        }

        image_files = []

        for file_path in TEST_IMAGES_DIR.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        image_files.sort()

        return image_files

    def show_current_image(self):
        if self.current_index >= len(self.image_files):
            self.finish()
            return

        image_path = self.image_files[self.current_index]

        image = Image.open(image_path)

        image.thumbnail(
            (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
        )

        self.current_photo = ImageTk.PhotoImage(image)

        self.image_label.config(
            image=self.current_photo
        )

        self.info_label.config(
            text=(
                f"{self.current_index + 1}/"
                f"{len(self.image_files)} - "
                f"{image_path.name}"
            )
        )

    def save_rating(self, rating):
        image_path = self.image_files[self.current_index]

        self.results.append(
            {
                "filename": image_path.name,
                "rating": float(rating),
            }
        )

        self.current_index += 1

        self.show_current_image()

    def finish(self):
        with open(
            OUTPUT_FILE,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "filename",
                    "rating",
                ]
            )

            for item in self.results:
                writer.writerow(
                    [
                        item["filename"],
                        item["rating"],
                    ]
                )

        print()
        print("DONE")
        print(f"Saved ratings to: {OUTPUT_FILE}")

        self.root.destroy()


def main():
    root = tk.Tk()

    app = TestRatingCollector(root)

    root.mainloop()


if __name__ == "__main__":
    main()