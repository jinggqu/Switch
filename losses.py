import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * mask)
        z_sum = torch.sum(score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), "predict & target shape do not match"
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes):
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]

        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=2)
        feat_k = F.normalize(feat_k, dim=-1, p=2)
        feat_k = feat_k.detach()

        # Pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  # (B*H*W, 1, 1)
        l_pos = l_pos.view(-1, 1)  # (B*H*W, 1)

        # Neg logit
        # Reshape features to batch size
        feat_q = feat_q.reshape(batch_size, -1, dim)  # (B, H*W, C)
        feat_k = feat_k.reshape(batch_size, -1, dim)  # (B, H*W, C)
        npatches = feat_q.size(1)  # H*W

        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # (B, H*W, H*W)

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]

        # Exclude self-contrast, only focus on the negative pairs (pixel in a pair should be at different locations)
        l_neg_curbatch.masked_fill_(diagonal, float("-inf"))
        l_neg = l_neg_curbatch.view(-1, npatches)  # (B*H*W, H*W)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  # (B*H*W, 1 + H*W)

        # Generate labels, make sure the first element is always the positive sample,
        # and the rest are negative samples. Then we can calculate the CrossEntropyLoss as the InfoNCE loss.
        loss = self.ce_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss
