import torch
import torch.nn as nn
import numpy as np

def sparse_to_onehot(msk, num_channels=3):
    numdim = len(msk.shape)
    
    if numdim == 3:
        res = torch.zeros((int(msk.shape[0]), int(num_channels), 
                           int(msk.shape[-2]), int(msk.shape[-1])))
    else:
        res = torch.zeros((int(num_channels), 
                           int(msk.shape[-2]), int(msk.shape[-1])))

    if msk.type().startswith('torch.cuda'):
        res = res.cuda()
    for ii in range(num_channels):
        res[...,ii,:,:] = msk==ii
    return res
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 activation = nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = activation 
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 activation = nn.ReLU(inplace=True),):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None, cuda=False):
        super(BBoxTransform, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, boxes, deltas):
        if self.mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        if self.std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        if deltas.type().startswith('torch.cuda'):
            self.std = self.std.cuda()
            self.mean = self.mean.cuda()
        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class BBoxInvTransform(nn.Module):
    """Inverse """
    def __init__(self, mean=None, std=None, cuda=False):
        super(BBoxInvTransform, self).__init__()
        self.mean = mean
        self.std = std
        if self.mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            NotImplementedError("non-zero mean handling is not implemented ")
        if self.std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))

    def forward(self, anchor, assigned_annotations):
        if anchor.type().startswith('torch.cuda'):
            self.std = self.std.cuda()
            self.mean = self.mean.cuda()
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        anchor_widths_pi = anchor_widths
        anchor_heights_pi = anchor_heights
        anchor_ctr_x_pi = anchor_ctr_x
        anchor_ctr_y_pi = anchor_ctr_y

        gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
        gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

        # clip widths to 1
        gt_widths  = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)

        targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
        targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
        targets_dw = torch.log(gt_widths / anchor_widths_pi)
        targets_dh = torch.log(gt_heights / anchor_heights_pi)

        regr_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).t()
        regr_targets = regr_targets / self.std
        return regr_targets


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()
        self.width = width
        self.height = height

    def forward(self, boxes, width=None, height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

        # _, _, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=self.width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=self.height)
      
        return boxes
