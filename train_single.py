import os
import shutil
import logging
import argparse
import random
import datetime
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset as dataset
from losses import DiceLoss
from util import model_summary, setup_logging, test_batch
from zoo.UNet import UNet


def get_args():
    parser = argparse.ArgumentParser()

    # Data related
    parser.add_argument("--exp", type=str, default="SINGLE", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="LN", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=256, help="Image width and height")
    parser.add_argument("--strong_augs", default=False, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=False, action=argparse.BooleanOptionalAction, help="Use weak augs")
    parser.add_argument("--num_strong_augs", type=int, default=1, help="Number of strong augs")
    parser.add_argument("--num_weak_augs", type=int, default=1, help="Number of weak augs")
    parser.add_argument("--randn_strong_augs", default=True, action="store_true", help="Random k strong augs")
    parser.add_argument("--randn_weak_augs", default=True, action="store_true", help="Random k weak augs")
    parser.add_argument("--num_workers", type=int, default=8)

    # Model related
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)

    # Training related
    parser.add_argument("--deterministic", default=True, action="store_true", help="Whether use deterministic training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--self_iters", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--labeled_bs", type=int, default=None, help="Batch size for labeled data")
    parser.add_argument("--labeled_ratio", type=float, default=0.05, help="Ratio of labeled data")
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Testing related
    parser.add_argument("--test", default=False, action="store_true", help="Load local checkpoint for testing")

    return parser.parse_args()


args = get_args()

# Set labeled batch size as half of the total batch size if not specified
if args.labeled_ratio == 1:
    args.labeled_bs = args.batch_size

if args.labeled_bs is None:
    args.labeled_bs = args.batch_size // 2

# Loss functions
CE_FN = nn.CrossEntropyLoss()
DICE_FN = DiceLoss(n_classes=args.num_classes)


def create_net(in_channels, num_classes):
    return UNet(in_channels=in_channels, num_classes=num_classes).cuda()


def self_train(args):
    # Model initialization
    model = create_net(args.in_channels, args.num_classes)
    model.train()

    model_summary({"model": model})

    # Data initialization
    dm = dataset.DataModule(args)
    trainloader, valloader, testloader = (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())

    # Optimizer initialization
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.self_iters)

    writer = SummaryWriter(args.self_snapshot_path + "/log")
    logging.info("Start self-training")

    iter_num = 0
    best_performance = 0.0
    max_iters = args.self_iters
    max_epoch = max_iters // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    labeled_bs = args.labeled_bs

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, _ = sampled_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Split labeled and unlabeled data
            images = volume_batch[:labeled_bs]
            labels = label_batch[:labeled_bs]

            # Get predictions from the models
            outs = model(images)

            # Compute loss
            loss_ce, loss_dice = (CE_FN(outs, labels.squeeze(1).long()), DICE_FN(outs, labels))
            loss = (loss_ce + loss_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_num += 1

            if iter_num % 10 == 0:
                writer.add_scalar("train/ce_loss", loss_ce.item(), iter_num)
                writer.add_scalar("train/dice_loss", loss_dice.item(), iter_num)
                writer.add_scalar("train/total_loss", loss.item(), iter_num)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iter_num)

            # Log images for visualization (first batch of every n iters)
            if iter_num % 1000 == 0:
                input_images = images[:4]
                outs_viz = torch.softmax(outs[:4], dim=1)
                labels_viz = labels[:4]
                outs_viz = torch.argmax(outs_viz, dim=1).unsqueeze(1)
                writer.add_images("self_train/images", input_images, iter_num)
                writer.add_images("self_train/outputs", outs_viz, iter_num)
                writer.add_images("self_train/labels", labels_viz, iter_num)

            # Explicitly delete tensors to free up memory immediately
            del (volume_batch, label_batch, images, labels, outs, loss, loss_ce, loss_dice)
            del sampled_batch

            # Validation
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_dict = {"dice": [], "iou": [], "hd95": [], "asd": []}
                for _, sampled_batch in enumerate(valloader):
                    metrics = test_batch(images=sampled_batch[0], labels=sampled_batch[1], model=model)
                    for key in metric_dict:
                        metric_dict[key].extend(metrics[key])

                dice_mean = np.mean(metric_dict["dice"])
                iou_mean = np.mean(metric_dict["iou"])
                hd95_values = np.array(metric_dict["hd95"])
                hd95_mean = (
                    np.mean(hd95_values[np.isfinite(hd95_values)]) if np.any(np.isfinite(hd95_values)) else np.nan
                )
                asd_values = np.array(metric_dict["asd"])
                asd_mean = np.mean(asd_values[np.isfinite(asd_values)]) if np.any(np.isfinite(asd_values)) else np.nan

                writer.add_scalar("info/val_dice", dice_mean, iter_num)
                writer.add_scalar("info/val_iou", iou_mean, iter_num)
                writer.add_scalar("info/val_hd95", hd95_mean, iter_num)
                writer.add_scalar("info/val_asd", asd_mean, iter_num)

                # Save best model
                if dice_mean > best_performance:
                    best_performance = dice_mean
                    save_best = os.path.join(args.self_snapshot_path, "unet_best_model.pth")
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    f"\titer: {iter_num}, dice: {dice_mean * 100:.2f}, iou: {iou_mean * 100:.2f}, "
                    f"hd95: {hd95_mean:.2f}, asd: {asd_mean:.2f}"
                )

                # Testing
                metric_dict = {"dice": [], "iou": [], "hd95": [], "asd": []}
                for _, sampled_batch in enumerate(testloader):
                    metrics = test_batch(images=sampled_batch[0], labels=sampled_batch[1], model=model)
                    for key in metric_dict:
                        metric_dict[key].extend(metrics[key])

                dice_mean = np.mean(metric_dict["dice"])
                iou_mean = np.mean(metric_dict["iou"])
                hd95_values = np.array(metric_dict["hd95"])
                hd95_mean = (
                    np.mean(hd95_values[np.isfinite(hd95_values)]) if np.any(np.isfinite(hd95_values)) else np.nan
                )
                asd_values = np.array(metric_dict["asd"])
                asd_mean = np.mean(asd_values[np.isfinite(asd_values)]) if np.any(np.isfinite(asd_values)) else np.nan

                writer.add_scalar("info/test_dice", dice_mean, iter_num)
                writer.add_scalar("info/test_iou", iou_mean, iter_num)
                writer.add_scalar("info/test_hd95", hd95_mean, iter_num)
                writer.add_scalar("info/test_asd", asd_mean, iter_num)

                # Clean up validation metrics
                del metric_dict, hd95_values, asd_values
                gc.collect()

                # Switch back to train mode
                model.train()
                torch.cuda.empty_cache()

            if iter_num >= max_iters:
                iterator.close()
                break

        if iter_num >= max_iters:
            iterator.close()
            break

    # Ensure a checkpoint is always saved at the end of training
    final_save = os.path.join(args.self_snapshot_path, "unet_best_model.pth")
    if not os.path.exists(final_save):
        torch.save(model.state_dict(), final_save)

    writer.close()
    return "Self-Training Finished!"


@torch.no_grad()
def test(args):
    print("Start testing")
    # Model initialization
    model = create_net(args.in_channels, args.num_classes)
    saved_best = os.path.join(args.self_snapshot_path, "unet_best_model.pth")
    model.load_state_dict(torch.load(saved_best, weights_only=True), strict=False)
    model.eval()

    # Data initialization
    dm = dataset.DataModule(args)
    testloader = dm.test_dataloader()

    # Create visualization folder
    viz_path = args.test_snapshot_path + "/viz"
    if os.path.exists(viz_path):
        shutil.rmtree(viz_path)
    os.makedirs(viz_path)

    metric_dict = {"dice": [], "iou": [], "hd95": [], "asd": []}
    for sampled_batch in tqdm(testloader, ncols=70):
        metrics = test_batch(
            images=sampled_batch[0],
            labels=sampled_batch[1],
            model=model,
            viz=True,
            viz_path=viz_path,
            names=sampled_batch[2],
        )
        for key in metric_dict:
            metric_dict[key].extend(metrics[key])

    dice_mean, dice_std = np.mean(metric_dict["dice"]), np.std(metric_dict["dice"])
    iou_mean, iou_std = np.mean(metric_dict["iou"]), np.std(metric_dict["iou"])
    hd95_values = np.array(metric_dict["hd95"])
    hd95_finite = hd95_values[np.isfinite(hd95_values)]
    hd95_mean, hd95_std = (np.mean(hd95_finite), np.std(hd95_finite)) if len(hd95_finite) > 0 else (np.nan, np.nan)
    asd_values = np.array(metric_dict["asd"])
    asd_finite = asd_values[np.isfinite(asd_values)]
    asd_mean, asd_std = (np.mean(asd_finite), np.std(asd_finite)) if len(asd_finite) > 0 else (np.nan, np.nan)

    df = pd.DataFrame(
        {
            "Metric": ["Dice", "IoU", "HD95", "ASD"],
            "Mean": [dice_mean, iou_mean, hd95_mean, asd_mean],
            "Std": [dice_std, iou_std, hd95_std, asd_std],
        }
    )

    result_str = (
        "\n=========================================================\n"
        + df.to_string(index=False, float_format="%.8f")
        + "\n=========================================================\n"
    )
    logging.info(result_str)

    # Create a sub-folder for single test with time and IoU
    backup_folder = os.path.join(
        args.test_snapshot_path, f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_IoU={iou_mean * 100:.2f}"
    )
    os.makedirs(backup_folder)

    # Backup the best model, log, and visualization
    shutil.copy(saved_best, os.path.join(backup_folder, "unet_best_model.pth"))
    shutil.move(viz_path, os.path.join(backup_folder, "viz"))
    shutil.move(os.path.join(args.test_snapshot_path, "log.txt"), os.path.join(backup_folder, "log.txt"))

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Logging
    snapshot_path_list = [
        f"runs/{args.exp}_unet_{args.dataset}_{args.labeled_ratio}_labeled/self_train",
        f"runs/{args.exp}_unet_{args.dataset}_{args.labeled_ratio}_labeled/test",
    ]
    args.self_snapshot_path = snapshot_path_list[0]
    args.test_snapshot_path = snapshot_path_list[1]

    for path in snapshot_path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    if not args.test:
        # Self-Train
        setup_logging(args, args.self_snapshot_path)
        self_train(args)

    # Test
    setup_logging(args, args.test_snapshot_path)
    test(args)
