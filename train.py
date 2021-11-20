# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
from pickle import FALSE
import random
import argparse
from cv2 import LMEDS, Tonemap, log, polarToCart
import numpy as np
import datetime

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from nni.utils import merge_parameter

from torch.utils.data import DataLoader
from imageio import imsave
from core.networks import *
from core.loss import SP_CAM_Loss

from core.datasets import *
from tools.general.Q_util import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

#import evaluate
import evaluator

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime

import  core.models as fcnmodel

#import evaluate
#from tools.ai import evaluator

import nni

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def get_params():
    parser = argparse.ArgumentParser()
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_dir', default='VOC2012/VOCdevkit/VOC2012/', type=str)

    ###############################################################################
    # Network
    ###############################################################################
    parser.add_argument('--architecture', default='SANET_Model_new_base', type=str)
    parser.add_argument('--backbone', default='resnest50', type=str)

    ###############################################################################
    # Hyperparameter
    ###############################################################################
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=12, type=int)#***********调

    parser.add_argument('--lr', default=0.01, type=float)#***********调
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)###keyitiao
    parser.add_argument('--tao', default=0.4, type=float)

    ###############################################################################
    # others
    ###############################################################################
    parser.add_argument('--withQ', default=True, type=str2bool)#***********改
    parser.add_argument('--Qmodelpath', default='experiments/models/bestQ.pth', type=str)#***********改
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--tag', default='pub_withq_lr_t', type=str)

    ###############################################################################
    ## parse for model fusion
    ###############################################################################
    parser.add_argument('--lr2', default=400, type=int)
    parser.add_argument('--lr3', default=10, type=int)
    parser.add_argument('--fgORall', default=True, type=str2bool)

    args, _ = parser.parse_known_args()
    return args


def main(args):
    set_seed(0)

    ###################################################################################
    # Arguments
    ###################################################################################
    
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args["tag"]}/{TIMESTAMP}/')   
    log_tag=create_directory(f'./experiments/logs/{args["tag"]}/')
    data_tag=create_directory(f'./experiments/data/{args["tag"]}/')
    model_tag=create_directory(f'./experiments/models/{args["tag"]}/')
    log_path = log_tag+ f'/{start_time}.txt'
    data_path = data_tag + f'/{start_time}.json'
    model_path = model_tag + f'/{start_time}.pth'
    
    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args["tag"]))
    log_func(str(args))
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transforms = [
        RandomResize_For_Segmentation(args["min_image_size"], args["max_image_size"]),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args["image_size"]),
    ]
    
    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    

    meta_dic = read_json('./data/VOC_2012.json')
    
    train_dataset = VOC_Dataset_For_MNSS(
        args["data_dir"], 'VOC2012/VOCdevkit/VOC2012/saliency_map/' ,'train_aug',train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], num_workers=args["num_workers"], shuffle=True, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    

    val_iteration = int(len(train_loader))
    log_iteration = int(val_iteration * args["print_ratio"])
    max_iteration = args["max_epoch"] * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    if(args['withQ']):
        Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        Q_model.load_state_dict(torch.load(args['Qmodelpath']))
        Q_model = nn.DataParallel(Q_model)
        Q_model.eval()
    else:
        Q_model=None

    ###################################################################################
    # Network
    ###################################################################################
    
    if(args['withQ']):
        model = SP_CAM_Model(args["backbone"], num_classes=meta_dic['classes'] + 1)
    else:
        model = CAM_Model(args["backbone"], num_classes=meta_dic['classes'] + 1,)
    
    model = model.cuda()
    model.train()
    log_func('[i] Architecture is {}'.format(args["architecture"]))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)
    
    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    if(args['withQ']):
        evaluatorA=evaluator.evaluator(withQ=True,refine_list=[0,20])
    else:
        evaluatorA=evaluator.evaluator(domain='train' ,withQ=False,refine_list=[0])

    
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    if(args["withQ"]):
        param_groups = model.get_parameter_groups1()
        params = [
            {'params': param_groups[0], 'lr': args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[1], 'lr': 2*args["lr"], 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[3], 'lr': 20*args["lr"], 'weight_decay': 0},
            {'params': param_groups[4], 'lr': args["lr2"]*args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[5], 'lr': 2*args["lr2"]*args["lr"], 'weight_decay': 0},
            {'params': param_groups[6], 'lr': args["lr3"]*args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[7], 'lr': 2*args["lr3"]*args["lr"], 'weight_decay': 0},
        ]
    else:
        param_groups = model.get_parameter_groups()
        params = [
            {'params': param_groups[0], 'lr': args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[1], 'lr': 2*args["lr"], 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*args["lr"], 'weight_decay': args["wd"]},
            {'params': param_groups[3], 'lr': 20*args["lr"], 'weight_decay': 0},
          ]
    optimizer = PolyOptimizer(params, lr=args["lr"], momentum=0.9, weight_decay=args["wd"], max_step=max_iteration, nesterov=args["nesterov"])
    
    lossfn=SP_CAM_Loss(args=args)
    lossfn = torch.nn.DataParallel(lossfn).cuda()
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss','cls_loss','sal_loss'])

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)
    best_valid_mIoU =-1
    for iteration in range(max_iteration):
        images, imgids,labels,masks,sailencys= train_iterator.get()
        images = images.cuda()
        labels = labels.cuda()
        sailencys = sailencys.cuda().view(sailencys.shape[0],1,sailencys.shape[1],sailencys.shape[2])/255.0
        prob=None
        if(args['withQ']):
            prob = Q_model(images)

        if(args['withQ']):
            logits = model(images,prob)
        else:
            logits = model(images)

        loss_list=lossfn(logits,prob,sailencys,labels)
        cls_loss=torch.mean(loss_list[0])
        sal_loss=torch.mean(loss_list[1])

        loss= cls_loss+args["alpha"]*(sal_loss) 

        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'sal_loss' : sal_loss.item(), 
            'cls_loss' : cls_loss.item()
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss,sal_loss,cls_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'cls_loss' : cls_loss, 
                'sal_loss' : sal_loss, 
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                sal_loss={sal_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        #################################################################################################
        # Evaluation
        #################################################################################################
        # val_iteration=1
        if (iteration + 1) % val_iteration == 0:
            mIoU,para = evaluatorA.evaluate(model,Q_model)[0]
            refine_num,threshold=para
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'refine_num' : refine_num,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                threshold={threshold:.2f}%,\
                refine_num={refine_num:.0f},\
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
            nni.report_intermediate_result(mIoU)

    nni.report_intermediate_result(best_valid_mIoU)
    
    write_json(data_path, data_dic)
    writer.close()

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise



        