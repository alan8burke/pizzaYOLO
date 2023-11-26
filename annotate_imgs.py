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


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area_box2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = area_box1 + area_box2 - intersection

    iou = intersection / union
    return iou


def non_max_suppression(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to the list of predictions.
    """
    predictions.sort(key=lambda x: x['score'], reverse=True)
    keep = []

    for i in range(len(predictions)):
        keep.append(True)

        for j in range(i + 1, len(predictions)):
            iou = calculate_iou(predictions[i]['box'], predictions[j]['box'])

            if iou >= iou_threshold:
                keep[-1] = False

    nms_predictions = [predictions[i] for i in range(len(predictions)) if keep[i]]
    return nms_predictions


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
            # Apply NMS
            preds_after_nms = non_max_suppression(preds, iou_threshold=0.2)

            # Add filepath to new dict
            updated_pred = {"prediction": preds_after_nms, "img_path": imgs_path[idx]}
            annotated_dataset.append(updated_pred)

            # Draw img
            draw = ImageDraw.Draw(imgs[idx])
            for pred in preds_after_nms:
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
