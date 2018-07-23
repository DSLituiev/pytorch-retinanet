from __future__ import print_function
import six
import csv
from collections import deque
from collections import OrderedDict
from collections import Iterable

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import os

import torch
import torch.nn as nn
import losses


def upd_mean(mu_t_1, x, t):
    # zero-based indexing
    t += 1
    return mu_t_1 + (x - mu_t_1)/t


def get_header(summary):
    return ['='.join(x.split('=')[:-1]).strip() for x in str(summary).split('\n')]

def evaluate_coco(dataset, model, threshold=0.05, use_gpu=True, 
                 save = False,
                 w_class=1.0,
                 w_regr = 1.0,
                 w_sem = 1.0,
                 num_classes = 2,
                 use_n_samples = None,
                 returntype = 'dict',
                 coco_header = None,
                 ):
    model.eval()
    print("model.training", model.training)

    loss_func_bbox = losses.FocalLoss()
    loss_func_semantic_xe = nn.CrossEntropyLoss(reduce=True, size_average=True)

    mean_loss_total = 0.0
    mean_loss_class = 0.0
    mean_loss_regr  = 0.0 
    mean_loss_sem   = 0.0 
    mean_ious       = [0.0]*num_classes

    if use_n_samples is None:
        use_n_samples = len(dataset)

    with torch.no_grad():

        # start collecting results
        results = []
        results_semantic = []
        image_ids = []

        for index in range(use_n_samples):
            data = dataset[index]
            if 'scale' in data:
                scale = data['scale']
            else:
                scale = 1.0

            # run network
            img = torch.FloatTensor(data['img'])
            img = img.permute(2, 0, 1)
            msk_npy = data['mask'][np.newaxis]
            msk = torch.LongTensor(data['mask'][np.newaxis])
            annot = np.array(data['annot'][np.newaxis])
            #print('annot', annot.dtype, annot.shape)
            if annot.shape[1]>0:
                annot = torch.FloatTensor(annot)
            else:
                annot = torch.ones((1, 1, 5)) * -1

            if use_gpu:
                img = img.cuda()
                msk = msk.cuda()
                annot = annot.cuda()

            classifications, regressions, anchors, semantic_logits, scores, labels, boxes =\
                model(img.float().unsqueeze(dim=0))

            # SEMANTIC SEGMENTATION
            semantic_loss = loss_func_semantic_xe(semantic_logits, msk) #/ nelements
            ## CONVERT LOGITS TO PROBABLILITIES
            semantic_prob = nn.Softmax2d()(semantic_logits)
            semantic_prob = semantic_prob.detach()#.cpu().numpy()
            iou_ = losses.sparse_iou_pt(msk, semantic_prob, reduce=False).cpu().detach().tolist() 
            results_semantic.append({'image_id': dataset.image_ids[index],
                                     'iou':iou_})
            ##
            classification_loss, regression_loss =\
                loss_func_bbox(classifications, regressions, 
                               anchors, annot)
            classification_loss = float(classification_loss.cpu().detach())
            regression_loss = float(regression_loss.cpu().detach())
            semantic_loss = float(semantic_loss.cpu().detach())

            loss = w_class * classification_loss + \
                   w_regr * regression_loss + \
                   w_sem * semantic_loss

            mean_loss_total = upd_mean(mean_loss_total, loss, index)
            mean_loss_class = upd_mean(mean_loss_class, classification_loss, index)
            mean_loss_regr  = upd_mean(mean_loss_regr, regression_loss, index)
            mean_loss_sem   = upd_mean(mean_loss_sem, semantic_loss, index)
            mean_ious        = [upd_mean(mu, float(iou__), index) for mu, iou__ in zip(mean_ious, iou_)]

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

        loss_summary_dict = OrderedDict([("loss_total", float(mean_loss_total)),
                                 ("loss_class", float(mean_loss_class)),
                                 ("loss_regr", float(mean_loss_regr)),
                                 ("loss_sem", float(mean_loss_sem)),])
        loss_summary_dict.update( {("iou_%d"%(ii+1)):vv for ii,vv in enumerate(mean_ious)} )

        logstr = [ "Loss:\tTotal: {:.4f}\tClass: {:.4f}\tRegr: {:.4f}\tSemantic: {:.4f}" ] +\
                ["\tIOU#{:d}: {{:.3f}}".format(n+1) for n in range(num_classes)]
        logstr = "".join(logstr)
        print(logstr.format(
               mean_loss_total, mean_loss_class, mean_loss_regr, mean_loss_sem, *mean_ious)
               )

        if save:
            # write output
            json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)
            json.dump(results_semantic, open('{}_semantic_results.json'.format(dataset.set_name), 'w'),
                      indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(results)
        #coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        model.train()

        if returntype == 'dict':
            if coco_header is None:
                coco_header = get_header(coco_eval)
            apar_summary_dict = OrderedDict(zip(coco_header, coco_eval.stats))
            loss_summary_dict.update(apar_summary_dict)
            return loss_summary_dict
        else:
            return coco_eval, loss_summary_dict


class CSVLogger():
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        #super(CSVLogger, self).__init__()
        self.on_train_begin()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            if isinstance(logs, OrderedDict):
                self.keys = logs.keys()
            elif isinstance(logs, dict):
                self.keys = sorted(logs.keys())

        print("keys", self.keys)
        print("logs.keys", logs.keys())
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        print(row_dict)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def __call__(self, *args, **kwargs):
        return self.on_epoch_end(*args, **kwargs)

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

    def __del__(self):
        try:
            self.on_train_end()
        except:
            pass

