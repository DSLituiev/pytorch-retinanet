import numpy as np
import torch
import torch.nn as nn
from functools import partial
from utils import BBoxInvTransform



def get_semantic_metrics(semantic_logits, msk, 
        loss_func_semantic_xe=nn.CrossEntropyLoss(reduce=True, size_average=True),
        ):
	# SEMANTIC SEGMENTATION
	semantic_loss = loss_func_semantic_xe(semantic_logits, msk) #/ nelements
	## CONVERT LOGITS TO PROBABLILITIES
	semantic_prob = nn.Softmax2d()(semantic_logits)
	semantic_prob = semantic_prob.detach()#.cpu().numpy()
	iou_ = sparse_iou_pt(msk, semantic_prob, reduce=False).cpu().detach().tolist()
	return semantic_loss, iou_


def iou_per_channel_np(mask, pred, stabilize = 1e-12, thr=0.5):
    """
    Input:
        - mask [batch_size, width, height]
        - pred [batch_size, width, height]
    """
    pred = pred>thr
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
    thr = 1.0/pred.shape[1]
    for cc in range(start, pred.shape[1]):
        prob_channel = pred[:,cc,...]
        mask_channel = (mask == cc)
        iou_[cc-start] = iou_per_channel_np(mask_channel, prob_channel,
                                            thr = thr,
                                            stabilize=stabilize)
    if reduce:
        iou_ = np.sum(iou_)
    return iou_


def iou_per_channel_pt(mask_channel, prob_channel, stabilize = 1e-12, thr=0.5):
    """
    Input:
        - mask [batch_size, width, height]
        - pred [batch_size, width, height]
    """
    pred = prob_channel > thr
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
    thr = 1.0/float(pred.shape[1])
    for cc in range(start, pred.shape[1]):
        prob_channel = pred[:,cc,...]
        mask_channel = (mask == cc)
        iou_[cc-start] = iou_per_channel_pt(mask_channel, prob_channel,
                                            thr=thr,
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


def intersect_annot_anchors(anchors, bbox_annotation, num_classes=2,
                            thr_iou_lo=0.4):
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
    targets[torch.ge(IoU_max, thr_iou_lo), :] = 0.0

    positive_indices = torch.ge(IoU_max, 0.5)
    num_positive_anchors = positive_indices.sum()
    if num_positive_anchors == 0:
        return None, None, None

    assigned_annotations = bbox_annotation[IoU_argmax, :]
    assigned_annotations = assigned_annotations[positive_indices, :]

    targets[positive_indices, :] = 0.0
    targets[positive_indices, assigned_annotations[:, 4].long()] = 1

    # REGRESSION TARGETS
    bbit = BBoxInvTransform()
    regr_targets = bbit(anchor[positive_indices], assigned_annotations)
    return targets, regr_targets, positive_indices


def regr_loss(targets, predictions, reduce=False,
              thr = 0.1):
    #regression_loss = torch.abs(regr_targets - regression[positive_indices, :])
    regression_diff = torch.abs(targets - predictions)
    # HINGE LOSS
    regression_loss = torch.where(
        torch.le(regression_diff, thr),
        0.5/thr * torch.pow(regression_diff, 2),
        regression_diff - 0.5 * thr
        )
    if reduce:
        return regression_loss.mean()
    else:
        return regression_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0,
                 regr_loss = regr_loss,
                 class_loss = focal_loss,):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.class_loss = partial(class_loss, alpha=0.25, gamma=2.0)
        self.regr_loss = regr_loss

    def forward(self, class_logits, regr_preds, anchors, annotations, ):
        use_gpu = class_logits.type().startswith('torch.cuda')

        batch_size = class_logits.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        if class_logits.shape[-1] != self.num_classes:
            self.num_classes = class_logits.shape[-1]
            print("updated self.num_classes to %d" % int(class_logits.shape[-1]))
        
        for j in range(batch_size):

            ## Calculate IOU between ANNOTATIONS and ANCHORS
            if len(annotations.shape)>1:
                bbox_annotation = annotations[j, :, :]
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
                """
                This step can be pre-calculated and cached
                """
                targets, regr_targets, positive_indices = \
                    intersect_annot_anchors(anchors, bbox_annotation,
                                            num_classes=self.num_classes)
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
                continue


            class_logits_sample = class_logits[j, :, :]
            regr_preds_sample = regr_preds[j, :, :]
            class_logits_sample = torch.clamp(class_logits_sample, 1e-4, 1.0 - 1e-4)
            cls_loss = self.class_loss(targets, class_logits_sample)

            # torch has no dimension/axis argument for any()
            num_positive_anchors = (((targets>0.0).sum(1)>0).sum()).float()
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors, min=0.01))

            # compute the loss for regr_preds_sample
            if num_positive_anchors > 0:
                #regression_loss = torch.abs(regr_targets - regr_preds_sample[positive_indices, :])
                regression_loss = regr_loss(regr_targets, regr_preds_sample[positive_indices, :])
                regression_losses.append(regression_loss.mean())
            else:
                zero_loss = torch.tensor(0).float()
                if use_gpu:
                    zero_loss = zero_loss.cuda()
                regression_losses.append(zero_loss)

        return (torch.stack(classification_losses).mean(dim=0, keepdim=True), 
                torch.stack(regression_losses).mean(dim=0, keepdim=True)
                )

