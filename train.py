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
from losses import ContrastiveLoss, DiceLoss
from util import model_summary, setup_logging, test_batch
from zoo.UNet import UNet


def get_args():
    parser = argparse.ArgumentParser()

    # Data related
    parser.add_argument("--exp", type=str, default="FULL", help="Experiment name")
    parser.add_argument("--dataset", type=str, default="LN", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=256, help="Image width and height")
    parser.add_argument("--strong_augs", default=True, action=argparse.BooleanOptionalAction, help="Use strong augs")
    parser.add_argument("--weak_augs", default=True, action=argparse.BooleanOptionalAction, help="Use weak augs")
    parser.add_argument("--num_strong_augs", type=int, default=1, help="Number of strong augs")
    parser.add_argument("--num_weak_augs", type=int, default=1, help="Number of weak augs")
    parser.add_argument("--randn_strong_augs", default=True, action="store_true", help="Random k strong augs")
    parser.add_argument("--randn_weak_augs", default=True, action="store_true", help="Random k weak augs")
    parser.add_argument("--fs_size", type=float, default=0.0175, help="Frequency area for amplitude switch")
    parser.add_argument("--num_coarse_patches", type=int, default=2, help="Number of coarse patches for MSS")
    parser.add_argument("--num_fine_patches", type=int, default=2, help="Number of fine patches for MSS")
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
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--consistency_weight", type=float, default=0.1)

    # Testing related
    parser.add_argument("--test", default=False, action="store_true", help="Load fine checkpoint for testing")

    return parser.parse_args()


args = get_args()

# Set labeled batch size as half of the total batch size if not specified
if args.labeled_bs is None:
    args.labeled_bs = args.batch_size // 2

# Loss functions
CE_FN = nn.CrossEntropyLoss(reduction="none")
DICE_FN = DiceLoss(n_classes=args.num_classes)
CONT_FN = ContrastiveLoss()
MSE_FN = nn.MSELoss()


def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def generate_mask(
    img_shape: tuple,
    num_coarse_patches: int = 2,
    num_fine_patches: int = 4,
    coarse_size: int = 128,
    fine_size: int = 32,
):
    """
    Generate the coarse and fine patches for the images.

    Parameters:
        img_shape (tuple): The shape of the input image (b, c, h, w).
        num_coarse_patches (int): The number of coarse patches to generate.
        num_fine_patches (int): The number of fine patches to generate.
        coarse_size (int): The patch size of the coarse patches.
        fine_size (int): The patch size of the fine patches.

    Returns:
        mask (torch.Tensor): The generated mask.
        loss_mask (torch.Tensor): The generated loss mask.
    """
    batch_size, _, img_x, img_y = img_shape
    mask = torch.ones((img_x, img_y), dtype=torch.bool, device=torch.device("cuda"))
    loss_mask = torch.ones((batch_size, img_x, img_y), dtype=torch.bool, device=torch.device("cuda"))

    for i in range(num_coarse_patches + num_fine_patches):
        patch_size = coarse_size if i < num_coarse_patches else fine_size

        upper_left_x = np.random.randint(0, img_x - patch_size)
        upper_left_y = np.random.randint(0, img_y - patch_size)

        # Patch as 0, others as 1
        mask[upper_left_x : upper_left_x + patch_size, upper_left_y : upper_left_y + patch_size] = False
        loss_mask[:, upper_left_x : upper_left_x + patch_size, upper_left_y : upper_left_y + patch_size] = False

    return mask, loss_mask


def mix_loss(
    preds: torch.Tensor,
    label_a: torch.Tensor,
    label_b: torch.Tensor,
    loss_mask: torch.Tensor,
    base_area_weight: float = 1.0,
    patch_area_weight: float = 0.5,
    unlabel_img_based=False,
    eps=1e-16,
):
    """
    Compute the loss for the mixed images.

    Parameters:
        preds (torch.Tensor): The model output for the mixed images. (first image as base, second image as patch)
        label_a (torch.Tensor): The ground truth labels for the first image.
        label_b (torch.Tensor): The ground truth labels for the second image.
        loss_mask (torch.Tensor): The mask for the mixed images.
        base_area_weight (float): The weight for the base area.
        patch_area_weight (float): The weight for the patch area.
        unlabel_img_based (bool): Whether the base image is unlabeled.

    Returns:
        loss_dice (torch.Tensor): The computed Dice loss for the mixed images.
        loss_ce (torch.Tensor): The computed cross-entropy loss for the mixed images.
    """
    if unlabel_img_based:
        base_area_weight, patch_area_weight = patch_area_weight, base_area_weight

    loss_mask_negative = ~loss_mask
    preds_soft = torch.softmax(preds, dim=1)

    loss_dice = DICE_FN(preds_soft, label_a, loss_mask.unsqueeze(1)) * base_area_weight
    loss_dice += DICE_FN(preds_soft, label_b, loss_mask_negative.unsqueeze(1)) * patch_area_weight

    label_a = label_a.squeeze(1).long()
    label_b = label_b.squeeze(1).long()

    loss_ce = (CE_FN(preds, label_a) * loss_mask).sum() / (loss_mask.sum() + eps) * base_area_weight
    loss_ce += (CE_FN(preds, label_b) * loss_mask_negative).sum() / (loss_mask_negative.sum() + eps) * patch_area_weight

    return loss_dice, loss_ce


def create_freq_mask(img_shape, fs_size=0.01):
    """
    Create a frequency mask for the images according to the length of the sqaured area.

    Parameters:
        img_shape (tuple): The shape of the input image (h, w).
        fs_size (float): The length of the squared area, range: (0, 1).

    Returns:
        mask (torch.Tensor): The generated mask.
    """
    H, W = img_shape

    # Calculate the size of the squared frequency area,
    # the total area is the specified percentage times the corresponding length
    freq_size_x, freq_size_y = int(H * fs_size), int(W * fs_size)

    # Define a mask to identify the low-frequency area
    mask = torch.zeros((H, W), dtype=torch.bool, device=torch.device("cuda"))

    # Calculate the starting point of the low-frequency area
    start_x = (H - freq_size_x) // 2
    start_y = (W - freq_size_y) // 2

    # Set the low-frequency area as 1
    mask[start_x : start_x + freq_size_x, start_y : start_y + freq_size_y] = True

    return mask


def mix_amplitude(images_a, images_b, mask):
    """
    Switch amplitude and phase between two batches of images, only for a specified low-frequency range.

    Parameters:
        images_a (torch.Tensor): Batch of images A, shape (B, C, H, W)
        images_b (torch.Tensor): Batch of images B, shape (B, C, H, W)
        mask (torch.Tensor): The mask for the low-frequency area

    Returns:
        restored_a: Batch of images A with amplitude of B and phase of A
        restored_b: Batch of images B with amplitude of A and phase of B
    """
    # Perform FFT
    fft_images_a = torch.fft.fft2(images_a)
    fft_images_b = torch.fft.fft2(images_b)

    # Shift the amplitude and phase to the center
    shifted_a = torch.fft.fftshift(fft_images_a)
    shifted_b = torch.fft.fftshift(fft_images_b)

    # Extract amplitude and phase
    amp_a = torch.abs(shifted_a)
    phase_a = torch.angle(shifted_a)
    amp_b = torch.abs(shifted_b)
    phase_b = torch.angle(shifted_b)

    # Switch the amplitude in the low-frequency area
    amp_a_new = amp_a * (~mask) + amp_b * mask
    amp_b_new = amp_b * (~mask) + amp_a * mask

    # Reconstruct the complex spectra
    swapped_a = amp_a_new * torch.exp(1j * phase_a)  # A with B's low-frequency amplitude
    swapped_b = amp_b_new * torch.exp(1j * phase_b)  # B with A's low-frequency amplitude

    # Shift back and perform inverse FFT
    swapped_a = torch.fft.ifftshift(swapped_a)
    swapped_b = torch.fft.ifftshift(swapped_b)

    restored_a = torch.fft.ifft2(swapped_a).real
    restored_b = torch.fft.ifft2(swapped_b).real

    return restored_a, restored_b


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

            # Split labeled batch into two halves
            l_img_i, l_img_j = (volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:labeled_bs])
            l_lab_i, l_lab_j = (label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:labeled_bs])

            mask, loss_mask = generate_mask(
                img_shape=l_img_i.shape,
                num_coarse_patches=args.num_coarse_patches,
                num_fine_patches=args.num_fine_patches,
            )

            # Build mixed images and labels
            ll_img = l_img_i * mask + l_img_j * ~mask  # i + j (as patch)
            ll_gt_lab = l_lab_i * mask + l_lab_j * ~mask

            # Get predictions from the models
            ll_outs = model(ll_img)

            # Compute loss
            loss_ce, loss_dice = mix_loss(ll_outs, l_lab_i, l_lab_j, loss_mask, unlabel_img_based=False)
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
                input_images = ll_img[:4]
                outs_viz = torch.softmax(ll_outs[:4], dim=1)
                labels_viz = ll_gt_lab[:4]
                outs_viz = torch.argmax(outs_viz, dim=1).unsqueeze(1)
                writer.add_images("pre_train/input_images", input_images, iter_num)
                writer.add_images("pre_train/outputs", outs_viz, iter_num)
                writer.add_images("pre_train/labels", labels_viz, iter_num)

            # Explicitly delete tensors to free up memory immediately
            del volume_batch, label_batch, l_img_i, l_img_j, l_lab_i, l_lab_j
            del mask, loss_mask, ll_img, ll_gt_lab, ll_outs, loss, loss_ce, loss_dice
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
    freq_mask = create_freq_mask(img_shape=(args.img_size, args.img_size), fs_size=args.fs_size)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, _ = sampled_batch
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # "l" prefix denotes images with true labels
            l_img_all = volume_batch[:labeled_bs]
            l_img_i, l_img_j = (volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:labeled_bs])
            l_lab_i, l_lab_j = (label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:labeled_bs])

            # "u" prefix denotes images without true labels (unlabeled)
            u_img_all = volume_batch[labeled_bs:]
            u_img_p, u_img_q = (
                volume_batch[labeled_bs : labeled_bs + unlabeled_sub_bs],
                volume_batch[labeled_bs + unlabeled_sub_bs :],
            )

            # LABELES OF "UNLABELED" IMAGES, *ONLY FOR VISUALIZATION*
            u_lab_p, u_lab_q = (
                label_batch[labeled_bs : labeled_bs + unlabeled_sub_bs],
                label_batch[labeled_bs + unlabeled_sub_bs :],
            )

            # ================================== #
            # Step 1: Multi-scale Switch (MSS)    #
            # ================================== #

            # EMA model inference
            with torch.no_grad():
                u_pseudo_out_p = ema_model(u_img_p)  # Psuedo label for unlabeled image P
                u_pseudo_out_q = ema_model(u_img_q)  # Psuedo label for unlabeled image Q
                u_pseudo_out_p = get_cut_mask(u_pseudo_out_p, nms=True).unsqueeze(1)
                u_pseudo_out_q = get_cut_mask(u_pseudo_out_q, nms=True).unsqueeze(1)

            # Generate mask for the mixed images
            mask, loss_mask = generate_mask(
                img_shape=l_img_i.shape,
                num_coarse_patches=args.num_coarse_patches,
                num_fine_patches=args.num_fine_patches,
            )

            # Mix images
            ul_img = u_img_p * mask + l_img_i * ~mask  # Unlabeled image as base, labeled image as patch
            lu_img = l_img_j * mask + u_img_q * ~mask  # Labeled image as base, unlabeled image as patch

            # Get logits and projected features from the models with mixed images
            ul_outs, feat_ul = model(ul_img, proj=True)
            lu_outs, feat_lu = model(lu_img, proj=True)

            # Compute switch loss
            loss_ce_ul, loss_dice_ul = mix_loss(ul_outs, u_pseudo_out_p, l_lab_i, loss_mask, unlabel_img_based=True)
            loss_ce_lu, loss_dice_lu = mix_loss(lu_outs, l_lab_j, u_pseudo_out_q, loss_mask, unlabel_img_based=False)
            mss_loss = (loss_ce_ul + loss_dice_ul + loss_ce_lu + loss_dice_lu) / 4

            # *ONLY FOR VISUALIZATION*, label of unlabeled image as base, label of labeled image as patch
            ul_gt_lab = u_lab_p * mask + l_lab_i * ~mask
            # *ONLY FOR VISUALIZATION*, label of labeled image as base, label of unlabeled image as patch
            lu_gt_lab = l_lab_j * mask + u_lab_q * ~mask

            # ============================ #
            # Step 2: Frequency Domain Switch (FDS) #
            # ============================ #

            # Switch amplitude between two batches of images
            restored_l_img, restored_u_img = mix_amplitude(l_img_all, u_img_all, freq_mask)
            r_l_img_i, r_l_img_j = (restored_l_img[:labeled_sub_bs], restored_l_img[labeled_sub_bs:])
            r_u_img_p, r_u_img_q = (restored_u_img[:unlabeled_sub_bs], restored_u_img[unlabeled_sub_bs:])

            # Restored unlabeled image as base, restored labeled image as patch
            r_ul_img = r_u_img_p * mask + r_l_img_i * ~mask
            # Restored labeled image as base, restored unlabeled image as patch
            r_lu_img = r_l_img_j * mask + r_u_img_q * ~mask

            # Get logits and projected features from the models with restored images
            r_ul_outs, feat_r_ul = model(r_ul_img, proj=True)
            r_lu_outs, feat_r_lu = model(r_lu_img, proj=True)

            # ============================ #
            # Step 3: Contrastive Learning #
            # ============================ #
            contrastive_loss = (CONT_FN(feat_ul, feat_r_ul) + CONT_FN(feat_lu, feat_r_lu)) / 2

            # =================== #
            # Step 4: Consistency #
            # =================== #
            consistency_loss = (MSE_FN(ul_outs, r_ul_outs) + MSE_FN(lu_outs, r_lu_outs)) / 2

            # Total loss
            loss = mss_loss + contrastive_loss * args.contrastive_weight + consistency_loss * args.consistency_weight

            if iter_num % 10 == 0:
                writer.add_scalar("train/mss_loss", mss_loss.item(), iter_num)
                writer.add_scalar("train/contrastive_loss", contrastive_loss.item(), iter_num)
                writer.add_scalar("train/consistency_loss", consistency_loss.item(), iter_num)
                writer.add_scalar("train/total_loss", loss.item(), iter_num)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], iter_num)

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            update_ema_variables(model, ema_model, 0.99)
            iter_num += 1

            # Log images for visualization (first batch of every n iters)
            if iter_num % 1000 == 0:
                ul_img_viz = ul_img[:4]
                lu_img_viz = lu_img[:4]
                ul_outs_viz = torch.softmax(ul_outs[:4], dim=1)
                lu_outs_viz = torch.softmax(lu_outs[:4], dim=1)
                ul_lab_viz = ul_gt_lab[:4]
                lu_lab_viz = lu_gt_lab[:4]
                ul_outs_viz = torch.argmax(ul_outs_viz, dim=1).unsqueeze(1)
                lu_outs_viz = torch.argmax(lu_outs_viz, dim=1).unsqueeze(1)
                writer.add_images("self_train/ul_input_images", ul_img_viz, iter_num)
                writer.add_images("self_train/lu_input_images", lu_img_viz, iter_num)
                writer.add_images("self_train/ul_outputs", ul_outs_viz, iter_num)
                writer.add_images("self_train/lu_outputs", lu_outs_viz, iter_num)
                writer.add_images("self_train/ul_labels", ul_lab_viz, iter_num)
                writer.add_images("self_train/lu_labels", lu_lab_viz, iter_num)
                r_ul_img_viz = r_ul_img[:4]
                r_lu_img_viz = r_lu_img[:4]
                writer.add_images("self_train/r_ul_input_images", r_ul_img_viz, iter_num)
                writer.add_images("self_train/r_lu_input_images", r_lu_img_viz, iter_num)

            # Explicitly delete tensors to free up memory immediately
            del volume_batch, label_batch, l_img_all, l_img_i, l_img_j, l_lab_i, l_lab_j
            del u_img_all, u_img_p, u_img_q, u_lab_p, u_lab_q
            del u_pseudo_out_p, u_pseudo_out_q, mask, loss_mask
            del ul_img, lu_img, ul_outs, feat_ul, lu_outs, feat_lu
            del loss_ce_ul, loss_dice_ul, loss_ce_lu, loss_dice_lu
            del (restored_l_img, restored_u_img, r_l_img_i, r_l_img_j, r_u_img_p, r_u_img_q)
            del r_ul_img, r_lu_img, r_ul_outs, feat_r_ul, r_lu_outs, feat_r_lu
            del mss_loss, contrastive_loss, consistency_loss, loss
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
    model.load_state_dict(torch.load(saved_best, weights_only=True))
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
