# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
from pickle import FALSE, TRUE
import sys
import copy
import shutil
import random
import argparse
from cv2 import LMEDS, Tonemap, log, polarToCart
import numpy as np
import datetime

import cv2 as cv
from numpy.core.fromnumeric import shape

import torch
from torch import tensor
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from nni.utils import merge_parameter

from torch.utils.data import DataLoader
from imageio import imsave
from core.networks import *
#import core.spnetworks

from core.datasets import *
from tools.general.Q_util import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

#import evaluate
import evaluatorqcam

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
#evaluatorA=evaluator.evaluator()
#evaluatorA.evaluate('/media/ders/mazhiming/PCAM/experiments/model/alpha1/2021-10-04 10:44:24.pth','/media/ders/zhangyumin/PuzzleCAM/experiments/models/train_Q_relu.pth')

import nni

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def get_params():
    parser = argparse.ArgumentParser()
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_dir', default='VOC2012/VOCdevkit/VOC2012/', type=str)

    ###############################################################################
    # Network
    ###############################################################################
    parser.add_argument('--architecture', default='Seg_Model', type=str)
    parser.add_argument('--backbone', default='resnest101', type=str)
    parser.add_argument('--mode', default='fix', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)
    #"backbone": {"_type":"choice","_value":["resnet50","resnet101","resnest50","resnest101"]},

    ###############################################################################
    # Hyperparameter
    ###############################################################################
    parser.add_argument('--batch_size', default=32, type=int)
    #parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)#***********调

    parser.add_argument('--lr', default=0.01, type=float)#***********调
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)###keyitiao
    parser.add_argument('--ksalq', default=1, type=float)

    # parser.add_argument('--alpha_q', default=0.5, type=float)
    parser.add_argument('--sal_th', default=0.001, type=float)
    # parser.add_argument('--sal_or_q', default=False, type=str2bool)
    parser.add_argument('--loss_mask', default=1.0, type=float)
    parser.add_argument('--tao', default=0.4, type=float)

    ###############################################################################
    # others
    ###############################################################################
    parser.add_argument('--withQ', default=False, type=str2bool)#***********改
    parser.add_argument('--Qmodelpath', default='/media/ders/zhangyumin/PuzzleCAM/experiments/models/bestQ.pth', type=str)#***********改
    parser.add_argument('--Qloss_rtime', default=0, type=int)

    parser.add_argument('--print_ratio', default=0.1, type=float)

    parser.add_argument('--tag', default='Qcam_batch8', type=str)

    ###############################################################################
    ## parse for model fusion
    ###############################################################################
    parser.add_argument('--ch_mid', default=512, type=int)  #1024
    parser.add_argument('--ch_q', default=64, type=int)
    parser.add_argument('--process1', default=0,type=int) 
    parser.add_argument('--process2', default=4, type=int)
    parser.add_argument('--with_se', default=0, type=int)
    parser.add_argument('--ratio', default=1, type=int)
    parser.add_argument('--conv_mode', default=0, type=int)
    parser.add_argument('--se_ratio', default=16, type=int)
    parser.add_argument('--process', default=4, type=int)
    
    # parser.add_argument('--covn', default=1, type=int)

    args, _ = parser.parse_known_args()
    return args
from torch.nn.modules.loss import _Loss

class SetLoss(_Loss):

  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SetLoss, self).__init__(size_average, reduce, reduction)
    self.args=args
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,logits,prob,sailencys,labels):
        if(self.args['withQ']):
     
            sailencys = poolfeat(sailencys, prob, 16, 16).cuda()
            
        b, c, h, w = logits.size()
        sailencys = F.interpolate(sailencys, size=(h, w))
        if( self.args['sal_th']>0.01):
            sailencys = (sailencys > self.args['sal_th']).float()

        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#
        loss_cls = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        if(True):
            cam=logits
            sailencys = F.interpolate(sailencys.float(), size=(h, w))

            label_map = labels[:,1:].view(b, 20, 1, 1).expand(size=(b, 20, h, w)).bool()#label_map_bg[0,:,0,0]
            # Map selection
            label_map_fg = torch.zeros(size=(b, 21, h, w)).bool().cuda()
            label_map_bg = torch.zeros(size=(b, 21, h, w)).bool().cuda()

            label_map_bg[:, 0] = True
            label_map_fg[:,1:] = label_map.clone()

            sal_pred = F.softmax(cam, dim=1) 

            iou_saliency = (torch.round(sal_pred[:, 1:].detach()) * torch.round(sailencys)).view(b, 20, -1).sum(-1) / \
                        (torch.round(sal_pred[:, 1:].detach()) + 1e-04).view(b, 20, -1).sum(-1)

            valid_channel = (iou_saliency > self.args["tao"]).view(b, 20, 1, 1).expand(size=(b, 20, h, w))
            
            label_fg_valid = label_map & valid_channel

            label_map_fg[:, 1:] = label_fg_valid
            label_map_bg[:, 1:] = label_map & (~valid_channel)

            # Saliency loss
            fg_map = torch.zeros_like(sal_pred).cuda()
            bg_map = torch.zeros_like(sal_pred).cuda()

            fg_map[label_map_fg] = sal_pred[label_map_fg]
            bg_map[label_map_bg] = sal_pred[label_map_bg]

            fg_map = torch.sum(fg_map, dim=1, keepdim=True)
            bg_map = torch.sum(bg_map, dim=1, keepdim=True)
    
            bg_map = torch.sub(1, bg_map) #label_map_fg[1,:,0,0] torch.sum(fg_map[7][0]>0.5) F.mse_loss(2*fg_map,sailencys) 
            sal_pred = fg_map * 0.5 + bg_map * (1 - 0.5) 

            sal_loss =F.mse_loss(sal_pred,sailencys)
       
        q_loss =torch.tensor(0.0).cuda()
    
        if(self.args['withQ'] and False ):
            # label_map = labels[:,1:].view(16, 20, 1, 1).expand(size=(16, 20, h, w)).bool()#label_map_bg[0,:,0,0]
                reconstr_feat5= torch.zeros(b,5,h,w).float() #reconstr_feat5.reshape(b,5,-1).max(dim=2)
                for ii in range(b):
                    cur=0
                    for jj in range(21):
                        if(labels[ii,jj]==1):
                            reconstr_feat5[ii,cur]=logits[ii,jj]
                            cur+=1
                            if(cur>=5):break
                    pass
                reconstr_feat=F.softmax(reconstr_feat5, dim=1).cuda()
                refinecam=reconstr_feat
                for i in range(self.args['Qloss_rtime']):
                        refinecam= upfeat(refinecam,prob)
                        refinecam= poolfeat(refinecam,prob)
                q_loss =F.mse_loss(reconstr_feat,refinecam)
        return loss_cls,sal_loss,q_loss

def main(args):

    ###################################################################################
    # Arguments
    ###################################################################################
    if(args['withQ']):
        evaluatorA=evaluatorqcam.evaluator(domain='train_600',fast_eval=True)
    else:
        import evaluator
        evaluatorA=evaluator.evaluator(domain='train',withQ=False)
    
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args["tag"]}/{TIMESTAMP}/')   

    # log_path = log_dir + f'{args["tag"]}{start_time}.txt'
    # data_path = data_dir + f'{args["tag"]}{start_time}.json'
    # model_path = model_dir + f'{args["tag"]}{start_time}.pth'
    log_tag=create_directory(f'./experiments/logs/{args["tag"]}/')
    data_tag=create_directory(f'./experiments/data/{args["tag"]}/')
    model_tag=create_directory(f'./experiments/models/{args["tag"]}/')

    log_path = log_tag+ f'/{start_time}.txt'
    data_path = data_tag + f'/{start_time}.json'
    model_path = model_tag + f'/{start_time}.pth'
    
    set_seed(args["seed"])
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args["tag"]))
    log_func()
    log_func(str(args))
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize_For_Segmentation(args["min_image_size"], args["max_image_size"]),
        RandomHorizontalFlip_For_Segmentation(),
        
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args["image_size"]),
    ]
    
    # if 'Seg' in args["architecture:
    #     if 'C' in args["architecture:
    #         train_transforms.append(Resize_For_Mask(args["image_size // 4))
    #     else:
    #         train_transforms.append(Resize_For_Mask(args["image_size // 8))

    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args["image_size"]),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])


    
    # train_dataset = VOC_Dataset_For_WSSS(args["data_dir, 'train_aug', 'VOC2012/VOCdevkit/VOC2012/saliency_map/', train_transform)
    train_dataset = VOC_Dataset_For_MNSS(
        args["data_dir"], 'VOC2012/VOCdevkit/VOC2012/saliency_map/' ,'train_aug',train_transform)
    # valid_dataset = VOC_Dataset_For_Segmentation(args["data_dir, 'train', test_transform)
    valid_dataset = VOC_Dataset_For_Testing_CAM(args["data_dir"], 'train', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], num_workers=args["num_workers"], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], num_workers=1, shuffle=False, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args["print_ratio"])
    max_iteration = args["max_epoch"] * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    setloss=SetLoss(args=args)
    
    setloss = torch.nn.DataParallel(setloss).cuda()
    if(args['withQ']):
        Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        Q_model.load_state_dict(torch.load(args['Qmodelpath']))
        Q_model = nn.DataParallel(Q_model)
        Q_model.eval()

    ###################################################################################
    # Network
    ###################################################################################
    # model = core.spnetworks.__dict__[args["architecture"]](args["backbone"], num_classes=meta_dic['classes'] + 1)

    model = core.networks.__dict__[args["architecture"]](args["backbone"], num_classes=meta_dic['classes'] + 1)
            # process1=args["process1"],process2=args["process2"],with_se=args["with_se"],conv_mode=args["conv_mode"],ratio=args["ratio"],se_ratio=args["se_ratio"])
    # if args["architecture"] == 'DeepLabv3+':
    #     model = DeepLabv3_Plus(args["backbone"], num_classes=meta_dic['classes'] + 1, mode=args["mode"], use_group_norm=args["use_gn"])
    # elif args["architecture"] == 'Seg_Model':
    #     model = SANET_Model(args["backbone"], num_classes=meta_dic['classes'] + 1)
    # elif args["architecture"] == 'CSeg_Model':
    #     model = CSeg_Model(args["backbone"], num_classes=meta_dic['classes'] + 1)


    param_groups = model.get_parameter_groups()
    params = [
        {'params': param_groups[0], 'lr': args["lr"], 'weight_decay': args["wd"]},
        {'params': param_groups[1], 'lr': 2*args["lr"], 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args["lr"], 'weight_decay': args["wd"]},
        {'params': param_groups[3], 'lr': 20*args["lr"], 'weight_decay': 0},
    ]
    
    model = model.cuda()
    model.train()
    # model.load_state_dict(torch.load('experiments/models/Q_cams_nni2/2021-10-17 17:44:07.pth'))
    #model.load_state_dict(torch.load('/media/ders/mazhiming/PCAM/experiments/models/train_10.1.pth'))
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

        # for sync bn
        # patch_replication_callback(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    # log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    # log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    # log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer(params, lr=args["lr"], momentum=0.9, weight_decay=args["wd"], max_step=max_iteration, nesterov=args["nesterov"])
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss','sal_loss','q_loss'])

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

        b,c,h,w= images.shape 

        if(args['withQ']):
            logits = model(images,prob)
        else:
            logits = model(images)

        # logits=F.interpolate(logits,(sailencys.shape[-2],sailencys.shape[-1]))
        # logits=poolfeat(logits,prob)

        lossret=setloss(logits,prob,sailencys,labels)
        loss_cls=torch.mean(lossret[0])
        sal_loss=torch.mean(lossret[1])
        q_loss=torch.mean(lossret[2])

        # sailencys=sailencys/sailencys.reshape(32,-1).max(dim=1)[0]
        #################################################################################################
        # Inference
        #################################################################################################
      
        ###############################################################################
        # The part is to calculate losses.
        # ###############################################################################

        #alpha=1
        # sal_start = args["alpha_sal"]
        # sal_final = args["alpha_final"]
        # q_start = 1-sal_start
        # q_final = 1-sal_final



        # alpha_sal=sal_start*(1-iteration / max_iteration ) + sal_final*iteration/max_iteration 
        # alpha_q= q_start*(1-iteration / max_iteration ) + q_final*iteration/max_iteration 

        loss= loss_cls+args["alpha"]*(sal_loss+args["ksalq"]*q_loss) 

        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss_cls.item(), 
            'sal_loss' : sal_loss.item(), 
            'q_loss' : q_loss.item()
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss,sal_loss,q_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'sal_loss' : sal_loss, 
                'q_loss' : q_loss, 
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                sal_loss={sal_loss:.4f}, \
                q_loss={q_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        #################################################################################################
        # Evaluation
        #################################################################################################
        #val_iteration=1
        # mIoU,re_th = evaluatorA.evaluate(model,args['Qmodelpath'])
        if (iteration + 1) % val_iteration == 0:
            #mIoU, threshold = evaluate(valid_loader)
            #best_mIoU,best_th = evaluate(valid_loader)
            mIoU,re_th = evaluatorA.evaluate(model,args['Qmodelpath'])
            refine,threshold=re_th
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
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
        #tuner_params = nni.get_next_parameter()
        #logger.debug(tuner_params)
        #params = vars(merge_parameter(get_params(), tuner_params))
        #print(params)
        #main(params)

        #refinecam= upfeat(refinecam,prob)
        img_path = '/media/ders/zhangyumin/DATASETS/dataset/newvoc/VOCdevkit/VOC2012/JPEGImages/'
        cam_path = "/media/ders/zhangyumin/PuzzleCAM/VOC2012/VOCdevkit/VOC2012/saliency_map"
        #path_list = os.listdir(path)
        save_path="/media/ders/zhangyumin/PuzzleCAM/experiments/result/png/LINEUP/sal_q_lineup/"
        Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        Q_model.load_state_dict(torch.load('/media/ders/zhangyumin/PuzzleCAM/experiments/models/bestQ.pth'))
        Q_model = nn.DataParallel(Q_model)
        Q_model.eval()
        with open('data/train.txt', 'r') as tf:
            train_list = tf.readlines()
        for filename in train_list:
            print(filename)

            img_id=filename.split('.')[0]
            img_id=img_id.split('\n')[0]
            image = cv.imread(os.path.join(img_path,img_id+ '.jpg'))
            cam = cv.imread(os.path.join(cam_path,img_id+ '.png'))
            H,W,_=shape(image)
            H_,W_=(H//16)*16,(W//16)*16

            HC,WC,_= shape(cam)
            HC_,WC_=(HC//16)*16,(WC//16)*16
            print(shape(cam))
            print(shape(image))
            resize_in=transforms.Resize([H_,W_])
            resize_in_C=transforms.Resize([H_//16,W_//16])
            #resize_out=transforms.Resize([H,W])
            image=torch.FloatTensor(np.array(image))
            image=image.transpose(2,0).transpose(1,2).unsqueeze(0)
            image=resize_in(image)
            #image.cuda()
            #image=image.float
            #image=image.to(device)
            #image=torch.FloatTensor(image)
            cam=torch.FloatTensor(np.array(cam))
            cam=cam.transpose(2,0).transpose(1,2).unsqueeze(0)
            cam=resize_in_C(cam)
            print(shape(cam))
            print(shape(image))

            prob = Q_model(image.cuda())
            #sp_image= poolfeat(cam.cuda(),prob)
            #sp_image=F.interpolate(sp_image, size=(H, W),mode='nearest')

            sp_image = upfeat(cam.cuda(),prob)
            #sp_image=resize_out(sp_image)
            sp_image=sp_image.squeeze(0).transpose(0,1).transpose(1,2)
            
            sp_image=sp_image.cpu()
            sp_image=sp_image.detach().numpy()
            cv.imwrite(os.path.join(save_path, img_id + '_sp.png'), sp_image)

    except Exception as exception:
        logger.exception(exception)
        raise


        