# Copyright 2018 Adrian Wolny
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class GDiceLossV2(nn.Module):
    def __init__(
        self,
        apply_nonlin: Optional[Tuple[str, Dict[str, Any]]] = None,
        smooth=1e-5,
        weight=None,
        compact_data=True,
        self_compute_weight=False,
    ):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.class_weight = weight
        self.compact_data = compact_data
        self.self_compute_weight = self_compute_weight

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, value=1)

        nonlin_output = net_output
        if self.apply_nonlin is not None:
            nonlin_output = self.apply_nonlin(nonlin_output)

        my_in = self.flatten(nonlin_output)
        target = self.flatten(y_onehot)
        target = target.float()
        if self.self_compute_weight:
            target_sum = target.sum(-1)
            class_weights = 1.0 / (target_sum * target_sum).clamp(min=self.smooth)
            class_weights = class_weights.detach()

        if self.self_compute_weight:
            intersect = (my_in * target).sum(-1) * class_weights
        else:
            intersect = (my_in * target).sum(-1)
        if self.class_weight is not None:
            weight = self.class_weight.detach()
            intersect = weight * intersect
        if self.compact_data:
            intersect = intersect.sum()

        if self.self_compute_weight:
            denominator = ((my_in + target).sum(-1) * class_weights).sum()
        else:
            denominator = (my_in + target).sum(-1)
        if self.compact_data:
            denominator = denominator.sum()

        result = 1.0 - 2.0 * intersect / denominator.clamp(min=self.smooth)
        return result

    @classmethod
    def flatten(cls, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)
