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

import utils
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
    parser.add_argument('--train-tag', help='', default = 'train2017')
    parser.add_argument('--val-tag', help='', default = 'val2017')
    parser.add_argument('--train-json', help='', default = None)
    parser.add_argument('--val-json', help='', default = None)
    parser.add_argument('--image-dir', help='', default = None)
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
    
    parser.add_argument('--no-rpn', help='train only semantic segmentation, NOT RPN', action='store_true')
    parser.add_argument('--bypass-semantic', help='train RPN, by feeding semantic segmentation into it', action='store_true')
    parser.add_argument('--w-sem', help='weight for semantic segmentation branch', type=float, default=0.0)
    parser.add_argument('--weight-decay', help='weight decay', type=float, default=0.0)
    parser.add_argument('--decoder-dropout', help='add droupout to the decoder', type=float, default=0.0)
    parser.add_argument('--decoder-activation', help='add droupout to the decoder', default='relu')
    parser.add_argument('--encoder-activation', help='add droupout to the encoder', default='relu')
    parser.add_argument('--batch-norm', help='batch norm for decoder', action='store_true')
    parser.add_argument('--w-class', help='weight for classification segmentation branch', type=float, default=1.0)
    parser.add_argument('--w-regr', help='weight for regression segmentation branch', type=float, default=1.0)
    parser.add_argument('--lr', help='initial learning rate', type=float, default=1e-5)
    parser.add_argument('--overwrite', help='overwrite existing folder',
                        dest='overwrite', action='store_true')
    parser.add_argument('--class-feature-sizes', nargs='+', type=int, default=[256]*3)
    parser.add_argument('--regr-feature-sizes', nargs='+', type=int, default=[256]*3)
   
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

        dataset_train = CocoDataset(parser.coco_path, set_name=parser.train_tag,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), 
                                #Resizer()
                                ]))
        dataset_val = CocoDataset(parser.coco_path, set_name=parser.val_tag, 
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
        dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=collater, batch_sampler=sampler_val)

    if parser.decoder_activation in (None, 'relu'):
        decoder_activation = nn.ReLU() 
    elif parser.decoder_activation.lower() == 'selu':
        decoder_activation = nn.SELU()
    else:
        raise NotImplementedError()

    if parser.encoder_activation in (None, 'relu'):
        encoder_activation = nn.ReLU() 
    elif parser.encoder_activation.lower() == 'selu':
        encoder_activation = nn.SELU()
    else:
        raise NotImplementedError()

    model_kws = dict(decoder_dropout = parser.decoder_dropout,
                     decoder_activation = decoder_activation,
                     encoder_activation = encoder_activation,
                     batch_norm = parser.batch_norm,
                     bypass_semantic = parser.bypass_semantic,
                     regr_feature_sizes=[256]*3,
                     class_feature_sizes=[256]*3,
                     )
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, **model_kws)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, **model_kws)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, **model_kws)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, **model_kws)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, **model_kws)
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

    ndigits = int(np.ceil(np.log10(len(dataloader_train))))
    if parser.no_rpn:
        retinanet.no_rpn = True
        logstr = '''Ep#{} | Iter#{:%d}/{:%d} || Losses | Sem: {:1.5f} | Running: {:1.4f} |'''  % ( ndigits, ndigits) 
    else:
        retinanet.no_rpn = False
        logstr = '''Ep#{} | Iter#{:%d}/{:%d} || Losses | Class: {:1.4f} | Regr: {:1.4f} | Sem: {:1.5f} | Running: {:1.4f}'''  % ( ndigits, ndigits) 

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(),
                           lr=parser.lr,
                           weight_decay = parser.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_func_bbox = losses.FocalLoss()
    loss_func_semantic_xe = nn.CrossEntropyLoss(reduce=True, size_average=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    logdir = "checkpoints/{}".format(arghash)
    if (not parser.overwrite) and os.path.exists(logdir) and \
            sum((1 for x in os.scandir(logdir) if x.name.endswith('.pt'))):
        raise RuntimeError("directory exists and non empty:\t%s" % logdir)
    os.makedirs(logdir, exist_ok=True)
    parser.to_yaml(os.path.join(logdir, 'checkpoint.info'))

    
    logfile_train = os.path.join(logdir, "progress-train.csv")
    logfile_val = os.path.join(logdir, "progress-val.csv")
    epoch_logger_train = coco_eval.CSVLogger(logfile_train)
    epoch_logger_val = coco_eval.CSVLogger(logfile_val)
    
    num_channels = len(dataset_train.coco_labels)
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

                if parser.bypass_semantic:
                    img = utils.sparse_to_onehot(msk, num_channels=1+num_channels)

                classifications, regressions, anchors, semantic_logits =\
                    retinanet(img)
             
                if not parser.bypass_semantic:
                    semantic_loss, iou_ = losses.get_semantic_metrics(
                                                semantic_logits, msk,
                                                loss_func_semantic_xe=loss_func_semantic_xe)
                else:
                    semantic_loss, iou_ = 0.0, [0.0] * num_channels

                if not retinanet.no_rpn:
                    classification_loss, regression_loss =\
                        loss_func_bbox(classifications, regressions, 
                                   anchors, annot)
                else:
                    if use_gpu:
                        classification_loss = regression_loss = torch.tensor(0.0).cuda()
                    else:
                        classification_loss = regression_loss = torch.tensor(0.0)

                loss = parser.w_class * classification_loss + \
                       parser.w_regr * regression_loss + \
                       parser.w_sem * semantic_loss

                mean_loss_total = upd_mean(mean_loss_total, loss, iter_num)
                mean_loss_class = upd_mean(mean_loss_class, classification_loss, iter_num)
                mean_loss_regr  = upd_mean(mean_loss_regr, regression_loss, iter_num)
                if not parser.bypass_semantic:
                    mean_loss_sem   = upd_mean(mean_loss_sem, semantic_loss, iter_num)
                    mean_ious       = [upd_mean(mu, float(iou__), iter_num) for mu, iou__ in zip(mean_ious, iou_)]
                
                if bool(loss == 0):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                if parser.no_rpn:
                    print(logstr.format(epoch_num, iter_num, len(dataloader_train),
                                         float(semantic_loss),
                                         np.mean(loss_hist),
                                         ))
                else:
                    print(logstr.format(epoch_num, iter_num, len(dataloader_train),
                        float(classification_loss),
                        float(regression_loss), float(semantic_loss),
                        np.mean(loss_hist)))
                    
            except Exception as e:
                raise e
                print(e)
#                break
        del data, msk, img
        # print(epoch_num, float(mean_loss_total), float(mean_loss_class), float(mean_loss_regr), float(mean_loss_sem), *mean_ious, sep=',')
        train_loss_summary_dict = OrderedDict([("loss_total", float(mean_loss_total)),
         ("loss_class", float(mean_loss_class)),
         ("loss_regr", float(mean_loss_regr)),
         ("loss_sem", float(mean_loss_sem)),])
        for ii,vv in enumerate(mean_ious):
            train_loss_summary_dict[("iou_%d"%(ii+1))]=vv
        retinanet.eval()
        print("EVAL ON TRAIN SET ({:d} samples)".format(parser.n_train_eval))
        print("=" * 30)
        if not retinanet.no_rpn:
            train_apar_summary = coco_eval.evaluate_coco(dataset_train, retinanet, 
                                    use_gpu=use_gpu, use_n_samples=parser.n_train_eval,
                                    save=False,
                                    returntype='dict', coco_header=coco_header,
                                    **{kk:vv for kk,vv in parser.__dict__.items() if str(kk).startswith('w_')})
            if coco_header is None:
                coco_header = list(set((train_apar_summary.keys())) - set(train_loss_summary_dict.keys()))
            train_apar_summary.update(train_loss_summary_dict)
        else:
            train_apar_summary = train_loss_summary_dict
            print("\t".join(['{}:  {:.4f}'.format(kk,vv) for kk,vv in train_apar_summary.items()]))
            
        epoch_logger_train(epoch_num, train_apar_summary)
        
        print("EVAL ON VALIDATION SET")
        if not retinanet.no_rpn:
            val_summary = coco_eval.evaluate_coco(dataset_val, retinanet,
                                    use_gpu=use_gpu, save=False, returntype='dict',
                                    **{kk:vv for kk,vv in parser.__dict__.items() if kk.startswith('w_')})
#            print(val_summary)
        else:
            with torch.no_grad():
                mean_loss_sem   = 0.0
                mean_ious       = [0.0]*dataset_train.num_classes()
                retinanet.eval()
                for iter_num, data in enumerate(dataloader_val):
                    optimizer.zero_grad()
                    #if iter_num>1: break
                    img = data['img']
                    msk = data['mask']
                    if use_gpu:
                        img = img.cuda()
                        msk = msk.cuda()
                    _,_,_, semantic_logits =\
                        retinanet(img)
                    del _

                    semantic_loss, iou_ = losses.get_semantic_metrics(semantic_logits, msk)

                    mean_loss_sem   = upd_mean(mean_loss_sem, semantic_loss, iter_num)
                    mean_ious       = [upd_mean(mu, float(iou__), iter_num) for mu, iou__ in zip(mean_ious, iou_)]
                val_summary = OrderedDict([
                    ("loss_sem", float(mean_loss_sem)),
                    ])
                for ii,vv in enumerate(mean_ious):
                    val_summary[("iou_%d"%(ii+1))]=vv
                print("\t".join(['{}:  {:.4f}'.format(kk,vv) for kk,vv in val_summary.items()]))
        epoch_logger_val(epoch_num, val_summary)
            
            
        scheduler.step(np.mean(epoch_loss))    

        torch.save(retinanet, '{}/retinanet_{}.pt'.format(logdir, epoch_num))

    retinanet.eval()

    torch.save(retinanet, '{}/model_final.pt'.format(logdir, epoch_num))

#if __name__ == '__main__':
#    main()
