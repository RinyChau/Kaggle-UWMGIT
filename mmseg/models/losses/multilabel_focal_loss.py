# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


# This method is used when cuda is not available
def py_sigmoid_focal_loss(pred,
                          target,
                          gamma=2.0,
                          alpha=0.5):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction with
            shape (N, C)
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
    """
    target = target.type(pred.type())
    logpt = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-logpt)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * logpt
    loss *= alpha * target + (1 - alpha) * (1 - target)
    loss = loss.mean()
    return loss


@LOSSES.register_module()
class MultiLabelFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(MultiLabelFocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction == 'mean', \
            "AssertionError: reduction should be 'mean'"
        assert isinstance(alpha, (float, )), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'

        assert pred.shape == target.shape, "pred shape should equals to target shape"

        if self.use_sigmoid:
            # num_classes = pred.size
            pred = pred.view(-1)
            target = target.view(-1)

            valid_mask = (target != ignore_index)
            pred = pred[valid_mask]
            target = target[valid_mask]
            loss_cls = self.loss_weight * py_sigmoid_focal_loss(
                pred,
                target,
                gamma=self.gamma,
                alpha=self.alpha,)
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name