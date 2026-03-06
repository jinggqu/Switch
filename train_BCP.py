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
from skimage.measure import label

import dataset as dataset
from losses import DiceLoss
from util import model_summary, setup_logging, test_batch
from zoo.UNet import UNet


def get_args():
    parser = argparse.ArgumentParser()

    # Data related
    parser.add_argument("--exp", type=str, default="BCP", help="Experiment name")
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
    parser.add_argument("--pre_iters", type=int, default=10000)
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
if args.labeled_bs is None:
    args.labeled_bs = args.batch_size // 2

# Loss functions
dice_loss = DiceLoss(n_classes=args.num_classes)


def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def generate_mask(img):
    batch_size, _, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones((batch_size, img_x, img_y), device=torch.device("cuda"))
    mask = torch.ones((img_x, img_y), device=torch.device("cuda"))
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w : w + patch_x, h : h + patch_y] = 0
    loss_mask[:, w : w + patch_x, h : h + patch_y] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = torch.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def get_largest_cc(probs):
    N = probs.shape[0]
    probs_np = probs.detach().cpu().numpy()

    batch_list = []
    for n in range(N):
        n_prob = probs_np[n]
        labels = label(n_prob)
        if labels.max() != 0:
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largest_cc = n_prob
        batch_list.append(largest_cc.astype(np.float32))  # Ensure correct type

    batch_tensor = torch.from_numpy(np.array(batch_list)).cuda()
    return batch_tensor


def get_cut_mask(out, thres=0.5, nms=False):
    probs = torch.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms:
        masks = get_largest_cc(masks)
    return masks


def load_net(net, path):
    state = torch.load(str(path), weights_only=True)
    net.load_state_dict(state["net"])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path), weights_only=True)
    net.load_state_dict(state["net"])
    optimizer.load_state_dict(state["opt"])


def save_net_opt(net, optimizer, path):
    state = {"net": net.state_dict(), "opt": optimizer.state_dict()}
    torch.save(state, str(path))


def create_net(in_channels, num_classes, ema=False):
    model = UNet(in_channels=in_channels, num_classes=num_classes).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def pre_train(args):
    # Model initialization
    model = create_net(args.in_channels, args.num_classes)
    model.train()

    model_summary({"model": model})

    # Data initialization
    dm = dataset.DataModule(args)
    trainloader, valloader, testloader = (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())

    # Optimizer initialization
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pre_iters)

    writer = SummaryWriter(args.pre_snapshot_path + "/log")
    logging.info("Start pre-training")

    iter_num = 0
    best_performance = 0.0
    max_iters = args.pre_iters
    max_epoch = max_iters // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    labeled_bs, labeled_sub_bs = args.labeled_bs, args.labeled_bs // 2

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, _ = sampled_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Split labeled and unlabeled data
            image_a, image_b = (volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:labeled_bs])
            label_a, label_b = (label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:labeled_bs])

            mask, loss_mask = generate_mask(image_a)

            input_image = image_a * mask + image_b * (1 - mask)
            gt_mixed = label_a * mask + label_b * (1 - mask)

            # Get predictions from the models
            outs = model(input_image)
            outs_soft = torch.softmax(outs, dim=1)

            # Compute loss
            loss_ce, loss_dice = mix_loss(
                outs, label_a.squeeze(1), label_b.squeeze(1), loss_mask, u_weight=1.0, unlab=True
            )
            loss = (loss_ce + loss_dice) / 2

            if iter_num % 10 == 0:
                writer.add_scalar("train/ce_loss", loss_ce.item(), iter_num)
                writer.add_scalar("train/dice_loss", loss_dice.item(), iter_num)
                writer.add_scalar("train/total_loss", loss.item(), iter_num)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iter_num)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_num += 1

            # Log images for visualization (first batch of every n iters)
            if iter_num % 1000 == 0:
                input_images = input_image[:4]
                outs_viz = outs_soft[:4]
                labels_viz = gt_mixed[:4]
                outs_viz = torch.argmax(outs_viz, dim=1).unsqueeze(1)
                writer.add_images("pre_train/input_images", input_images, iter_num)
                writer.add_images("pre_train/outputs", outs_viz, iter_num)
                writer.add_images("pre_train/labels", labels_viz, iter_num)

            # Explicitly delete tensors to free up memory immediately
            del volume_batch, label_batch, image_a, image_b, label_a, label_b
            del (mask, loss_mask, input_image, gt_mixed, outs, outs_soft, loss, loss_ce, loss_dice)
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
                    save_best = os.path.join(args.pre_snapshot_path, "unet_best_model.pth")
                    save_net_opt(model, optimizer, save_best)

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

    # Ensure a checkpoint is always saved at the end of pre-training
    final_save = os.path.join(args.pre_snapshot_path, "unet_best_model.pth")
    if not os.path.exists(final_save):
        save_net_opt(model, optimizer, final_save)

    writer.close()
    return "Pre-Training Finished!"


def self_train(args):
    # Model initialization
    model = create_net(args.in_channels, args.num_classes)
    ema_model = create_net(args.in_channels, args.num_classes, ema=True)
    model.train()
    ema_model.train()

    model_summary({"model": model, "ema_model": ema_model})

    # Data initialization
    dm = dataset.DataModule(args)
    trainloader, valloader, testloader = (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader())

    # Optimizer initialization
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.self_iters)

    # Load pre-trained model
    pre_trained_model_path = os.path.join(args.pre_snapshot_path, "unet_best_model.pth")
    load_net(ema_model, pre_trained_model_path)
    load_net_opt(model, optimizer, pre_trained_model_path)
    optimizer.param_groups[0]["lr"] = args.learning_rate  # Reset learning rate
    logging.info(f"Loaded pre-trained model from {pre_trained_model_path}")

    writer = SummaryWriter(args.self_snapshot_path + "/log")
    logging.info("Start self-training")

    iter_num = 0
    best_performance = 0.0
    max_iters = args.self_iters
    max_epoch = max_iters // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    labeled_bs = args.labeled_bs
    labeled_sub_bs, unlabeled_sub_bs = (int(args.labeled_bs // 2), int((args.batch_size - args.labeled_bs) // 2))

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, _ = sampled_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # Images with true labels
            l_img_a, l_img_b = (volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:labeled_bs])
            l_lab_a, l_lab_b = (label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:labeled_bs])

            # Images without true labels
            u_img_a, u_img_b = (
                volume_batch[labeled_bs : labeled_bs + unlabeled_sub_bs],
                volume_batch[labeled_bs + unlabeled_sub_bs :],
            )
            u_lab_a, u_lab_b = (
                label_batch[labeled_bs : labeled_bs + unlabeled_sub_bs],
                label_batch[labeled_bs + unlabeled_sub_bs :],
            )

            # EMA model inference
            with torch.no_grad():
                p_out_a = ema_model(u_img_a)  # Psuedo label for unlabeled image A
                p_out_b = ema_model(u_img_b)  # Psuedo label for unlabeled image B
                p_out_a = get_cut_mask(p_out_a, nms=True)
                p_out_b = get_cut_mask(p_out_b, nms=True)

            mask, loss_mask = generate_mask(l_img_a)

            u_input = u_img_a * mask + l_img_a * (1 - mask)  # Unlabeled-based image with labeled image
            l_input = l_img_b * mask + u_img_b * (1 - mask)  # Labeled-based image with unlabeled image
            u_lable = u_lab_a * mask + l_lab_a * (1 - mask)  # Unlabeled-based label with labeled label
            l_lable = l_lab_b * mask + u_lab_b * (1 - mask)  # Labeled-based label with unlabeled label

            u_outs = model(u_input)  # Unlabeled-based output
            l_outs = model(l_input)  # Labeled-based output
            u_outs_soft = torch.softmax(u_outs, dim=1)
            l_outs_soft = torch.softmax(l_outs, dim=1)

            loss_ce_a, loss_dice_a = mix_loss(u_outs, p_out_a, l_lab_a.squeeze(1), loss_mask, unlab=True)
            loss_ce_b, loss_dice_b = mix_loss(l_outs, l_lab_b.squeeze(1), p_out_b, loss_mask)

            mixed_ce = loss_ce_a + loss_ce_b
            mixed_dice = loss_dice_a + loss_dice_b

            loss = (mixed_ce + mixed_dice) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema_variables(model, ema_model, 0.99)
            iter_num += 1

            if iter_num % 10 == 0:
                writer.add_scalar("train/ce_loss", mixed_ce.item(), iter_num)
                writer.add_scalar("train/dice_loss", mixed_dice.item(), iter_num)
                writer.add_scalar("train/total_loss", loss.item(), iter_num)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iter_num)

            # Log images for visualization (first batch of every n iters)
            if iter_num % 1000 == 0:
                # Unlabeled-based images
                u_input_viz = u_input[:4]
                u_outs_viz = torch.argmax(u_outs_soft[:4], dim=1).unsqueeze(1)
                u_labels_viz = u_lable[:4]
                writer.add_images("self_train/unlabeled_based_images", u_input_viz, iter_num)
                writer.add_images("self_train/unlabeled_based_outputs", u_outs_viz, iter_num)
                writer.add_images("self_train/unlabeled_based_labels", u_labels_viz, iter_num)

                # Labeled-based images
                l_input_viz = l_input[:4]
                l_outs_viz = torch.argmax(l_outs_soft[:4], dim=1).unsqueeze(1)
                l_labels_viz = l_lable[:4]
                writer.add_images("self_train/labeled_based_images", l_input_viz, iter_num)
                writer.add_images("self_train/labeled_based_outputs", l_outs_viz, iter_num)
                writer.add_images("self_train/labeled_based_labels", l_labels_viz, iter_num)

            # Explicitly delete tensors to free up memory immediately
            del volume_batch, label_batch, l_img_a, l_img_b, l_lab_a, l_lab_b
            del u_img_a, u_img_b, u_lab_a, u_lab_b
            del p_out_a, p_out_b, mask, loss_mask
            del u_input, l_input, u_lable, l_lable
            del u_outs, l_outs, u_outs_soft, l_outs_soft
            del loss_ce_a, loss_dice_a, loss_ce_b, loss_dice_b
            del mixed_ce, mixed_dice, loss
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

    # Ensure a checkpoint is always saved at the end of self-training
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
        f"runs/{args.exp}_unet_{args.dataset}_{args.labeled_ratio}_labeled/pre_train",
        f"runs/{args.exp}_unet_{args.dataset}_{args.labeled_ratio}_labeled/self_train",
        f"runs/{args.exp}_unet_{args.dataset}_{args.labeled_ratio}_labeled/test",
    ]
    args.pre_snapshot_path = snapshot_path_list[0]
    args.self_snapshot_path = snapshot_path_list[1]
    args.test_snapshot_path = snapshot_path_list[2]

    for path in snapshot_path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    if not args.test:
        # Pre-train
        setup_logging(args, args.pre_snapshot_path)
        pre_train(args)

        # Self-Train
        setup_logging(args, args.self_snapshot_path)
        self_train(args)

    # Test
    setup_logging(args, args.test_snapshot_path)
    test(args)
