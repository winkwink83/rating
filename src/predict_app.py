from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from joblib import load
from tkinterdnd2 import DND_FILES, TkinterDnD

from features import extract_image_features


MODEL_FILE = Path("data/models/photo_rating_model.joblib")

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500


class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Rating Predictor")

        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)

        self.model = self.load_model()

        self.selected_image_path = None

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.instruction_label = tk.Label(
            main_frame,
            text="Przeciągnij i upuść zdjęcie do okna.",
            font=("Arial", 16),
        )
        self.instruction_label.pack(pady=20)

        self.drop_area = tk.Label(
            main_frame,
            text="UPUŚĆ ZDJĘCIE TUTAJ",
            font=("Arial", 18),
            relief="ridge",
            width=35,
            height=5,
        )
        self.drop_area.pack(pady=10)

        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self.handle_drop)

        self.info_label = tk.Label(
            main_frame,
            text="Brak zdjęcia",
            font=("Arial", 12),
        )
        self.info_label.pack(pady=15)

        self.predict_button = tk.Button(
            main_frame,
            text="Pokaż ocenę",
            font=("Arial", 14),
            width=18,
            height=2,
            command=self.predict_rating,
            state=tk.DISABLED,
        )
        self.predict_button.pack(pady=15)

        self.result_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 22, "bold"),
        )
        self.result_label.pack(pady=15)

    def load_model(self):
        if not MODEL_FILE.exists():
            messagebox.showerror(
                "Brak modelu",
                f"Nie znaleziono modelu:\n{MODEL_FILE}\n\n"
                f"Najpierw uruchom:\npython src/train.py",
            )
            raise FileNotFoundError(MODEL_FILE)

        return load(MODEL_FILE)

    def handle_drop(self, event):
        file_path = self.clean_drop_path(event.data)

        if not file_path.exists():
            messagebox.showerror("Błąd", "Nie znaleziono pliku.")
            return

        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            messagebox.showerror("Błąd", "To nie jest obsługiwany format zdjęcia.")
            return

        self.selected_image_path = file_path

        self.info_label.config(
            text=f"Wczytano: {file_path.name}"
        )

        self.result_label.config(text="")

        self.predict_button.config(state=tk.NORMAL)

    def clean_drop_path(self, raw_path):
        path_text = raw_path.strip()

        if path_text.startswith("{") and path_text.endswith("}"):
            path_text = path_text[1:-1]

        return Path(path_text)

    def predict_rating(self):
        if self.selected_image_path is None:
            messagebox.showinfo("Info", "Najpierw przeciągnij zdjęcie.")
            return

        self.result_label.config(text="Liczenie...")
        self.root.update_idletasks()

        features = extract_image_features(self.selected_image_path)

        X = np.array([features], dtype=np.float32)

        predicted_rating = self.model.predict(X)[0]

        predicted_rating = float(np.clip(predicted_rating, 1.0, 10.0))

        self.result_label.config(
            text=f"{predicted_rating:.2f} / 10"
        )


def main():
    root = TkinterDnD.Tk()

    app = PredictApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()