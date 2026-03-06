import logging
import os
import sys
from typing import List

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchmetrics.functional import f1_score as _dice, jaccard_index as _iou
from monai.metrics import compute_hausdorff_distance as _hd95, compute_average_surface_distance as _asd


def setup_logging(args, log_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def format_params(num):
    if num >= 1e6:
        return f"{num / 1e6:.1f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f} K"
    else:
        return str(num)


def model_summary(model_dict):
    summary_data = []

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, model in model_dict.items():
        model_params = sum(p.numel() for p in model.parameters())
        trainable = any(p.requires_grad for p in model.parameters())
        mode = "train" if trainable else "eval"

        total_params += model_params
        if trainable:
            trainable_params += model_params
        else:
            non_trainable_params += model_params

        summary_data.append([name, model.__class__.__name__, format_params(model_params), mode])

    df = pd.DataFrame(summary_data, columns=["Name", "Type", "Params", "Mode"])

    print("=========================================================")
    print(df.to_string(index=False))
    print("---------------------------------------------------------")
    print(f"{format_params(trainable_params)} Trainable params")
    print(f"{format_params(non_trainable_params)} Non-trainable params")
    print(f"{format_params(total_params)} Total params")
    print(f"{total_params / 1e6 * 4:.3f} Total estimated model params size (MB)")
    print("=========================================================")


def calculate_metric_percase(preds: Tensor, labels: Tensor):
    batch_size = preds.shape[0]
    metrics = {"dice": [], "iou": [], "hd95": [], "asd": []}

    for i in range(batch_size):
        metrics["dice"].append(_dice(preds[i], labels[i], task="binary").item())
        metrics["iou"].append(_iou(preds[i], labels[i], task="binary").item())
        metrics["hd95"].append(
            _hd95(preds[i].unsqueeze(1), labels[i].unsqueeze(1), include_background=False, percentile=95).item()
        )
        metrics["asd"].append(_asd(preds[i].unsqueeze(1), labels[i].unsqueeze(1), include_background=False).item())

    return metrics


def test_batch(images, labels, model, viz=False, viz_path=None, names=None):
    images, labels = images.cuda(), labels.cuda()
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(torch.softmax(model(images), dim=1), dim=1).unsqueeze(1)

        # Visualize the results
        if viz:
            assert viz_path is not None, "Visualization path not provided during testing"
            assert names is not None, "File names not provided during visualization"
            visualize(images, labels, preds, names, viz_path)

    return calculate_metric_percase(preds == 1, labels == 1)


def visualize(
    images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, file_names: List[str], viz_path: str
) -> None:
    os.makedirs(viz_path, exist_ok=True)

    for i, file_name in enumerate(file_names):
        # Convert to numpy, scale to 0-255 and convert to uint8
        image_i = (images[i].cpu().numpy().squeeze(0) * 255).astype(np.uint8)
        labels_i = (labels[i].cpu().numpy() * 255).astype(np.uint8)
        preds_i = (preds[i].cpu().numpy().squeeze(0) * 255).astype(np.uint8)

        # Create RGB image with Red - Ground truth, Green - Predicted
        rgb_image = np.zeros((image_i.shape[0], image_i.shape[1], 3), dtype=np.uint8)
        rgb_image[:, :, 0] = np.maximum(rgb_image[:, :, 0], labels_i)
        rgb_image[:, :, 1] = np.maximum(rgb_image[:, :, 1], preds_i)

        # Save the overlay images with ground truth and predicted images, but without input
        Image.fromarray(rgb_image).save(f"{viz_path}/{os.path.splitext(file_name)[0]}.png", mode="RGB")

        # Save the overlay images with input, ground truth and predicted images
        rgb_image[:, :, 0] = np.maximum(image_i, labels_i)
        rgb_image[:, :, 1] = np.maximum(image_i, preds_i)
        rgb_image[:, :, 2] = image_i
        Image.fromarray(rgb_image).save(f"{viz_path}/{os.path.splitext(file_name)[0]}_overlay.png", mode="RGB")

        # Save only the predicted images (binary mask)
        Image.fromarray(preds_i).save(f"{viz_path}/{os.path.splitext(file_name)[0]}_pred.png", mode="L")
