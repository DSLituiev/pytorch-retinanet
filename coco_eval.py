from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os

import torch
import torch.nn as nn
from losses import sparse_iou_np

def evaluate_coco(dataset, model, threshold=0.05, use_gpu=True):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        results_semantic = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            if 'scale' in data:
                scale = data['scale']
            else:
                scale = 1.0

            # run network
            img = torch.FloatTensor(data['img'])
            img = img.permute(2, 0, 1)
            msk = data['mask'][np.newaxis]

            if use_gpu:
                img = img.cuda()
            scores, labels, boxes, semsegm = model(img.float().unsqueeze(dim=0))
            # CONVERT LOGITS TO PROBABLILITIES
            semsegm = nn.Softmax2d()(semsegm)
            semsegm = semsegm.detach().cpu().numpy()
            iou_ = sparse_iou_np(msk, semsegm, reduce=False).tolist() 
            results_semantic.append({'image_id': dataset.image_ids[index],
                                     'iou':iou_})
            #print("iou", iou_)

            #if len(results_semantic)>1:
            #    break 
            if len(boxes.shape) == 1:
                print("no boxes predicted for the instance %d\tid = %s" % (index,
                      dataset.image_ids[index]))
                print(data.keys())
                print("skipping")
                continue
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()
            #semsegm = semsegm.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)
        json.dump(results_semantic, open('{}_semantic_results.json'.format(dataset.set_name), 'w'),
                  indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        return
