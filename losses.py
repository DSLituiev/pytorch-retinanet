import numpy as np
import torch
import torch.nn as nn

def iou_per_channel_np(mask, pred, stabilize = 1e-12):
    """
    Input:
        - mask [batch_size, width, height]
        - pred [batch_size, width, height]
    """
    pred = pred>0.5
    union = np.sum(np.logical_or(mask, pred))
    intersection = np.sum(np.logical_and(mask, pred))
    iou = (intersection+stabilize)/(union+stabilize)
    return iou

def sparse_iou_np(mask, pred, skip_bg = True,
                  reduce=True,
                  stabilize = 1e-12):
    """
    Input:
        - mask [batch_size, width, height]            -- ground truth
        - pred [batch_size, channels, width, height]  -- predicted probabilities
    """
    start = 1 if skip_bg else 0
    iou_ = np.zeros(pred.shape[1] - start)
    for cc in range(start, pred.shape[1]):
        prob_channel = pred[:,cc,...]
        mask_channel = (mask == cc)
        iou_[cc-start] = iou_per_channel_np(mask_channel, prob_channel,
                                            stabilize=stabilize)
    if reduce:
        iou_ = np.sum(iou_)
    return iou_

def iou_per_channel_pt(mask_channel, prob_channel, stabilize = 1e-12):
    """
    Input:
        - mask [batch_size, width, height]
        - pred [batch_size, width, height]
    """
    pred = prob_channel>0.5
    union = torch.sum(mask_channel | pred)
    intersection = torch.sum(mask_channel & pred)
    iou = (intersection.float()+stabilize)/(union.float()+stabilize)
    return iou

def sparse_iou_pt(mask, pred, skip_bg = True,
                  reduce=True,
                  stabilize = 1e-12):
    """
    Input:
        - mask [batch_size, width, height]            -- ground truth
        - pred [batch_size, channels, width, height]  -- predicted probabilities
    """
    start = 1 if skip_bg else 0
    iou_ = torch.zeros(pred.shape[1] - start)
    for cc in range(start, pred.shape[1]):
        prob_channel = pred[:,cc,...]
        mask_channel = (mask == cc)
        iou_[cc-start] = iou_per_channel_pt(mask_channel, prob_channel,
                                            stabilize=stabilize)
    if reduce:
        iou_ = np.sum(iou_)
    return iou_

def calc_iou(a, b):
    """
    a -- anchors
    b -- annotations
    """

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) -\
             torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) -\
             torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih
    IoU = intersection / ua
    return IoU


def focal_loss(targets, classification, alpha = 0.25,
               gamma=2.0,):
    use_gpu = targets.type().startswith('torch.cuda')
    alpha_factor = torch.ones(targets.shape) * alpha
    if use_gpu:
        alpha_factor = alpha_factor.cuda()

    alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
    focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

    bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
    # cls_loss = focal_weight * torch.pow(bce, gamma)
    cls_loss = focal_weight * bce
    zeros_ = torch.zeros(cls_loss.shape)
    if use_gpu:
        zeros_ = zeros_.cuda()
    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros_)
    return cls_loss

def focal_loss_detectron(target, logit,
                         alpha = 0.25, gamma=2.0,):

    use_gpu = target.type().startswith('torch.cuda')
    softplus = torch.nn.Softplus()
    if use_gpu:
        softplus = softplus.cuda()

    logit_pos_flag = torch.ge(logit, 0.0)
    p = torch.sigmoid(logit)

    term1 = (1-p)**gamma * torch.log(p)
    term2 = - p**gamma * \
                torch.where(logit_pos_flag, 
                            logit+softplus(-logit),
                            softplus(logit))
                
    loss = - torch.where(torch.gt(target,0), alpha*term1, torch.tensor(0.0)) \
           - torch.where(torch.lt(target,0), (1-alpha)*term2, torch.tensor(0.0))
    return loss

def intersect_annot_anchors(anchors, bbox_annotation, num_classes=2):
    if bbox_annotation.shape[0] == 0:
        return None, None, None
    use_gpu = anchors.type().startswith('torch.cuda')
    anchor = anchors[0, :, :]
    
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4]) 
    # num_anchors x num_annotations
    IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

    targets = torch.ones((int(anchor.shape[0]), num_classes)) * -1
    if use_gpu:
        targets = targets.cuda()

    # CLASSIFICATION TARGETS
    #targets[torch.lt(IoU_max, 0.4), :] = 0
    targets[torch.ge(IoU_max, 0.25), :] = 0.0

    positive_indices = torch.ge(IoU_max, 0.5)

    num_positive_anchors = positive_indices.sum()

    assigned_annotations = bbox_annotation[IoU_argmax, :]
    assigned_annotations = assigned_annotations[positive_indices, :]

    targets[positive_indices, :] = 0.0
    targets[positive_indices, assigned_annotations[:, 4].long()] = 1
    # REGRESSION TARGETS
    
    anchor_widths  = anchor[:, 2] - anchor[:, 0]
    anchor_heights = anchor[:, 3] - anchor[:, 1]
    anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

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
    
#    anch_w = torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
#    if use_gpu:
#        anch_w = anch_w.cuda()
#    regr_targets = regr_targets / anch_w
    return targets, regr_targets, positive_indices

def regr_loss(targets, predictions, reduce=False):
    #regression_loss = torch.abs(regr_targets - regression[positive_indices, :])
    regression_diff = torch.abs(targets - predictions)
    # HINGE LOSS
    regression_loss = torch.where(
        torch.le(regression_diff, 1.0 / 9.0),
        0.5 * 9.0 * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / 9.0
        )
    if reduce:
        return regression_loss.mean()
    else:
        return regression_loss
    
class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations,
        alpha = 0.25,
        gamma = 2.0,
        ):
        use_gpu = classifications.type().startswith('torch.cuda')

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
#
#        anchor_widths  = anchor[:, 2] - anchor[:, 0]
#        anchor_heights = anchor[:, 3] - anchor[:, 1]
#        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
#        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            ## Calculate IOU between ANNOTATIONS and ANCHORS
            if len(annotations.shape)>1:
                bbox_annotation = annotations[j, :, :]
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
                """
                This step can be pre-calculated and cached
                """
                targets, regr_targets, positive_indices = \
                    intersect_annot_anchors(anchors, bbox_annotation,
                                            num_classes=classification.shape[-1])
            else:
                targets = None
            if targets is None:
                "todo: penalize false positives using loss of the maximum or top-k predictions"
                if use_gpu:
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())


            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
#            cls_loss = focal_loss_detectron(targets, classification,
            cls_loss = focal_loss(targets, classification,
                                            alpha = 0.25, gamma=2.0,)

                    
            # torch has no dimension/axis argument for any()
            num_positive_anchors = (((targets>0.0).sum(1)>0).sum()).float()
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors, min=0.01))

            # compute the loss for regression

            if num_positive_anchors > 0:
                #regression_loss = torch.abs(regr_targets - regression[positive_indices, :])
                regression_loss = regr_loss(regr_targets, regression[positive_indices, :])
                regression_losses.append(regression_loss.mean())
            else:
                zero_loss = torch.tensor(0).float()
                if use_gpu:
                    zero_loss = zero_loss.cuda()
                regression_losses.append(zero_loss)

        return (torch.stack(classification_losses).mean(dim=0, keepdim=True), 
                torch.stack(regression_losses).mean(dim=0, keepdim=True)
                )

