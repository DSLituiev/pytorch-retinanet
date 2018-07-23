import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model_w_semsegm as model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
from coco_eval import upd_mean
from collections import OrderedDict
from attrdict import AttrDict
#from viz import plot_bboxes

assert int(torch.__version__.split('.')[1]) >= 3

use_gpu = torch.cuda.is_available()
print('CUDA available: {}'.format(use_gpu))


#def main(args=None):
if __name__ == '__main__':
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--load', help='model checkpoint')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--batch-size', help='batch size', type=int, default=2)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--n_train_eval', help='Number training samples to evaluate AP/AR metrics on', type=int,
    default=20)
    parser.add_argument('--w-sem', help='weight for semantic segmentation branch', type=float, default=0.0)
    parser.add_argument('--w-class', help='weight for classification segmentation branch', type=float, default=1.0)
    parser.add_argument('--w-regr', help='weight for regression segmentation branch', type=float, default=1.0)
    parser.add_argument('--lr', help='initial learning rate', type=float, default=1e-5)

    parser = parser.parse_args()
#    parser = parser.parse_args(args)
    parser = AttrDict(parser.__dict__)
    parser.add_git()
    arghash = parser.md5
    print("argument hash:", arghash)
    

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                   transform=transforms.Compose([Normalizer(), Augmenter(), 
                                #Resizer()
                                ]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', 
                                  transform=transforms.Compose([Normalizer(), 
                                  #Resizer()
                                  ]))

        classes = []
 
    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')


        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

    print(dataset_train[0].keys())
    print('-'*20)
    for iter_num, data in enumerate(dataloader_train):
        break
    print(data.keys())
    print('-'*20)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')        

    if parser.load:
        if use_gpu:
            retinanet = torch.load(parser.load)
        else:
            retinanet = torch.load(parser.load, map_location='cpu')
        #retinanet.load_state_dict(torch.load(parser.load))

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_func_bbox = losses.FocalLoss()
    loss_func_semantic_xe = nn.CrossEntropyLoss(reduce=True, size_average=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    logdir = "checkpoints/{}".format(arghash)
    os.makedirs(logdir, exist_ok=True)
    parser.to_yaml(os.path.join(logdir, 'checkpoint.info'))

    ndigits = int(np.ceil(np.log10(len(dataloader_train))))
    logstr = '''Ep#{} | Iter#{:%d}/{:%d} || Losses | Class: {:1.4f} | Regr: {:1.4f} | Sem: {:1.5f} | Running: {:1.4f}'''  % ( ndigits, ndigits) 
    logfile_train = os.path.join(logdir, "progress-train.csv")
    logfile_val = os.path.join(logdir, "progress-val.csv")
    epoch_logger_train = coco_eval.CSVLogger(logfile_train)
    epoch_logger_val = coco_eval.CSVLogger(logfile_val)
    
    coco_header = None
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.freeze_bn()   
        epoch_loss = []
        mean_loss_total = 0.0
        mean_loss_class = 0.0
        mean_loss_regr  = 0.0
        mean_loss_sem   = 0.0
        mean_ious       = [0.0]*dataset_train.num_classes()
        
        for iter_num, data in enumerate(dataloader_train):
            #if iter_num>1: break
            try:
                optimizer.zero_grad()

                #print("="*10, 0)
                img = data['img']
                msk = data['mask']
                nelements = msk.shape[-1] * msk.shape[-2]
                annot = data['annot']
                #print("ANNOT", annot.shape)
                if use_gpu:
                    img = img.cuda()
                    msk = msk.cuda()
                    annot = annot.cuda()
                classifications, regressions, anchors, semantic =\
                    retinanet(img)
                
                semantic_loss = loss_func_semantic_xe(semantic, msk)
                iou_ = losses.sparse_iou_pt(msk, semantic, reduce=False).cpu().detach().tolist()
                classification_loss, regression_loss =\
                    loss_func_bbox(classifications, regressions, 
                               anchors, annot)

                loss = parser.w_class * classification_loss + \
                       parser.w_regr * regression_loss + \
                       parser.w_sem * semantic_loss
                mean_loss_total = upd_mean(mean_loss_total, loss, iter_num)
                mean_loss_class = upd_mean(mean_loss_class, classification_loss, iter_num)
                mean_loss_regr  = upd_mean(mean_loss_regr, regression_loss, iter_num)
                mean_loss_sem   = upd_mean(mean_loss_sem, semantic_loss, iter_num)
                mean_ious       = [upd_mean(mu, float(iou__), iter_num) for mu, iou__ in zip(mean_ious, iou_)]
                
                if bool(loss == 0):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print( logstr.format(epoch_num, iter_num, len(dataloader_train),
                    float(classification_loss),
                    float(regression_loss), float(semantic_loss),
                    np.mean(loss_hist)))
            except Exception as e:
                raise e
                print(e)
#                break
        
        # print(epoch_num, float(mean_loss_total), float(mean_loss_class), float(mean_loss_regr), float(mean_loss_sem), *mean_ious, sep=',')
        train_loss_summary_dict = OrderedDict([("loss_total", float(mean_loss_total)),
         ("loss_class", float(mean_loss_class)),
         ("loss_regr", float(mean_loss_regr)),
         ("loss_sem", float(mean_loss_sem)),])
        train_loss_summary_dict.update( {("iou_%d"%(ii+1)):vv for ii,vv in enumerate(mean_ious)} )
        retinanet.eval()
        print("EVAL ON TRAIN SET ({:d} samples)".format(parser.n_train_eval))
        print("=" * 30)
        train_apar_summary = coco_eval.evaluate_coco(dataset_train, retinanet, 
                                use_gpu=use_gpu, use_n_samples=parser.n_train_eval,
                                save=False,
                                returntype='dict', coco_header=coco_header,
                                **{kk:vv for kk,vv in parser.__dict__.items() if str(kk).startswith('w_')})
        
        if coco_header is None:
            coco_header = list(set((train_apar_summary.keys())) - set(train_loss_summary_dict.keys()))
#        print(train_apar_summary)
        train_apar_summary.update(train_loss_summary_dict)
        epoch_logger_train(epoch_num, train_apar_summary)
        
        if parser.dataset == 'coco':
            print("EVAL ON VALIDATION SET")
            val_summary = coco_eval.evaluate_coco(dataset_val, retinanet,
                                    use_gpu=use_gpu, save=False, returntype='dict',
                                    **{kk:vv for kk,vv in parser.__dict__.items() if kk.startswith('w_')})
#            print(val_summary)
            epoch_logger_val(epoch_num, val_summary)
            
        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            eval_csv(dataloader_val, retinanet, total_loss,)
            
        scheduler.step(np.mean(epoch_loss))    

        torch.save(retinanet, '{}/retinanet_{}.pt'.format(logdir, epoch_num))

    retinanet.eval()

    torch.save(retinanet, '{}/model_final.pt'.format(logdir, epoch_num))

#if __name__ == '__main__':
#    main()
