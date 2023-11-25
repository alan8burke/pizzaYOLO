import json
import os
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import pipeline


if torch.cuda.is_available():
    DEVICE = 0  # GPU cuda:0
    ENCODER = "google/owlvit-large-patch14"
else:
    DEVICE = -1  # CPU
    ENCODER = "google/owlvit-base-patch32"


def load_images_from_folder(folder_path: str) -> Tuple[List[Image.Image], List[str]]:
    """
    Returns a tuple with a list containing PIL img object (converted to RGB)
    and a list of string with the image paths
    """
    images = []
    imgs_path = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(image_path)
                images.append(img.convert("RGB"))
                imgs_path.append(image_path)
                print(f"Loaded image: {filename}")
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    return images, imgs_path


def run_annotations(input_folder, text_prompts, output_folder) -> None:
    # As the dataset is small, we load everything in memory
    # If the dataset was large, either use a Dataset object or generator
    imgs, imgs_path = load_images_from_folder(input_folder)

    # Zero-shot object detection pipeline
    checkpoint = ENCODER
    detector = pipeline(
        model=checkpoint, task="zero-shot-object-detection", device=DEVICE
    )

    annotated_dataset = []
    for idx, img in enumerate(tqdm(imgs)):
        preds = detector(img, candidate_labels=text_prompts)

        if preds:
            # Add filepath to new dict
            updated_pred = {"prediction": preds, "img_path": imgs_path[idx]}
            annotated_dataset.append(updated_pred)

            # Draw img
            draw = ImageDraw.Draw(imgs[idx])
            for pred in preds:
                box = pred["box"]
                label = pred["label"]
                score = pred["score"]

                xmin, ymin, xmax, ymax = box.values()
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

            # Save annotated img to disk
            imgs[idx].save(
                output_folder + imgs_path[idx].split("imgs")[-1], format="JPEG"
            )

    with open(output_folder + "/annotated.json", "w", encoding="utf-8") as f:
        json.dump(annotated_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Params
    folder_path = "imgs"
    text_prompts = ["pizza"]
    output_folder_path = "annotated"

    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    run_annotations(folder_path, text_prompts, output_folder_path)
