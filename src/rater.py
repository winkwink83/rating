import csv
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

from dataset import get_unrated_image_files


RATINGS_FILE = Path("data/ratings.csv")

IMAGES_DIR = Path("images")
TRASH_DIR = IMAGES_DIR / "trash"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

MAX_IMAGE_WIDTH = 1000
MAX_IMAGE_HEIGHT = 700


class ImageRaterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Rating System")

        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)

        TRASH_DIR.mkdir(parents=True, exist_ok=True)

        self.image_files = get_unrated_image_files()
        self.current_index = 0
        self.current_photo = None
        self.history = []

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        self.image_label = tk.Label(main_frame)
        self.image_label.pack(pady=20, expand=True)

        self.info_label = tk.Label(main_frame, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)

        rating_frame = tk.Frame(main_frame)
        rating_frame.pack(pady=10)

        for rating in range(1, 11):
            button = tk.Button(
                rating_frame,
                text=str(rating),
                width=4,
                height=2,
                command=lambda value=rating: self.rate_current_image(value),
            )
            button.pack(side=tk.LEFT, padx=3)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=20)

        undo_button = tk.Button(
            control_frame,
            text="Cofnij ostatnią akcję",
            width=22,
            height=2,
            command=self.undo_last_action,
        )
        undo_button.pack(side=tk.LEFT, padx=10)

        skip_button = tk.Button(
            control_frame,
            text="Pomiń zdjęcie",
            width=18,
            height=2,
            command=self.skip_image,
        )
        skip_button.pack(side=tk.LEFT, padx=10)

        finish_button = tk.Button(
            control_frame,
            text="Zakończ zbieranie danych",
            width=22,
            height=2,
            command=self.finish,
        )
        finish_button.pack(side=tk.LEFT, padx=10)

        self.show_current_image()

    def show_current_image(self):
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Koniec", "Oceniono wszystkie zdjęcia.")
            self.root.destroy()
            return

        image_path = self.image_files[self.current_index]

        image = Image.open(image_path)
        image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))

        self.current_photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=self.current_photo)

        self.info_label.config(
            text=f"{self.current_index + 1}/{len(self.image_files)} — {image_path.name}"
        )

    def rate_current_image(self, rating):
        image_path = self.image_files[self.current_index]

        with open(RATINGS_FILE, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([image_path.name, rating])

        self.history.append(
            {
                "action": "rate",
                "image_path": image_path,
                "index": self.current_index,
            }
        )

        self.image_files.pop(self.current_index)

        self.show_current_image()

    def skip_image(self):
        image_path = self.image_files[self.current_index]

        destination = TRASH_DIR / image_path.name

        shutil.move(str(image_path), str(destination))

        self.history.append(
            {
                "action": "skip",
                "image_path": image_path,
                "trash_path": destination,
                "index": self.current_index,
            }
        )

        self.image_files.pop(self.current_index)

        self.show_current_image()

    def undo_last_action(self):
        if not self.history:
            messagebox.showinfo("Info", "Nie ma czego cofnąć.")
            return

        last_action = self.history.pop()

        action_type = last_action["action"]
        image_path = last_action["image_path"]
        index = last_action["index"]

        if action_type == "rate":
            self.remove_last_rating_from_csv(image_path.name)

        elif action_type == "skip":
            trash_path = last_action["trash_path"]

            if trash_path.exists():
                shutil.move(str(trash_path), str(image_path))

        self.image_files.insert(index, image_path)
        self.current_index = index

        self.show_current_image()

    def remove_last_rating_from_csv(self, image_name):
        with open(RATINGS_FILE, mode="r", newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        for index in range(len(rows) - 1, 0, -1):
            if rows[index][0] == image_name:
                rows.pop(index)
                break

        with open(RATINGS_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def finish(self):
        self.root.destroy()


def rate_images():
    root = tk.Tk()
    app = ImageRaterApp(root)
    root.mainloop()