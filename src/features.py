import clip
import torch

from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


def extract_image_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True,
        )

    features = image_features.squeeze().cpu().numpy()

    return features