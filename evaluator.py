from operator import mod
import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from core.networks import *
from core.datasets import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.dataset.voc_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime
from core.spnetwork_new import SANET_Model_new_base
import core.models as fcnmodel

import dataset_root

parser = argparse.ArgumentParser()

###################################################################################
def get_params():
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--dataset', default='voc12', type=str, choices=['voc12', 'coco'])
    parser.add_argument('--domain', default='train', type=str)
    

    parser.add_argument(
        '--Qmodel_path', default="models_ckpt/Q_model_img.pth", type=str)  # 
    # parser.add_argument('--Qmodel_path', default='models_ckpt/Q_model_final.pth', type=str)#
    
    parser.add_argument(
        '--Cmodel_path', default='experiments/models/train_sp_cam_VOC_tile/2022-09-08 11:11:43.pth', type=str)  # 
    
    parser.add_argument('--savepng', default=False, type=str2bool)
    parser.add_argument('--savenpy', default=False, type=str2bool)
    
    parser.add_argument('--sp_cam', default=True, type=str2bool)
    
    parser.add_argument('--tag', default='evaluate_', type=str)
    parser.add_argument('--curtime', default='00', type=str)
    
    args = parser.parse_args()
    return args

class evaluator:
    def __init__(self, dataset='voc12',domain='train', SP_CAM=True, save_np_path=None,savepng_path=None,muti_scale=False,th_list=list(np.arange(0.2, 0.5, 0.1)),refine_list = range(0, 50, 10)) -> None:
        self.C_model = None
        self.Q_model = None
        self.SP_CAM = SP_CAM
        if(muti_scale):
        #   self.scale_list = [0.5, 1, 1.5, 2.0]  # - is flip
          self.scale_list = [0.5, 1, 1.5, 2.0, -0.5, -1, -1.5, -2.0]  # - is flip
        else:
          self.scale_list =  [1]   # - is flip
            
        self.th_list = th_list
        self.refine_list = refine_list
        self.parms = []
        for renum in self.refine_list:
            for th in self.th_list:
                self.parms.append((renum, th))
        class_num = 21 if dataset=='voc12' else 81
        self.meterlist = [Calculator_For_mIoU(
            class_num ) for x in self.parms]
        
        self.save_png_path = savepng_path
        self.save_np_path = save_np_path
        if(self.save_png_path!=None ):
            if not os.path.exists(self.save_png_path):
                os.mkdir(self.save_png_path)
                
                
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Transpose_For_Segmentation()
        ])
        if(dataset=='voc12'):
            valid_dataset = Dataset_For_Evaluation(
                dataset_root.VOC_ROOT,domain, test_transform,dataset)
        else:
            valid_dataset = Dataset_For_Evaluation(
                dataset_root.COCO_ROOT,domain, test_transform,'coco')
            
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    def get_cam(self, images, ids, Qs):
        with torch.no_grad():
            cam_list = []
            _, _, h, w = images.shape
            for s, q in zip(self.scale_list, Qs):
                target_size = (round(h * abs(s)), round(w * abs(s)))
                scaled_images = F.interpolate(
                    images, target_size, mode='bilinear', align_corners=False)
                H_, W_ = int(
                    np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
                scaled_images = F.interpolate(
                    scaled_images, (H_, W_), mode='bilinear', align_corners=False)
                if(s < 0):
                    scaled_images = torch.flip(
                        scaled_images, dims=[3])  # ?dims
                if(self.SP_CAM):
                    logits,x4 = self.C_model(scaled_images, q)
                else:
                    logits,x4  = self.C_model(scaled_images)

                # pred = F.softmax(logits, dim=1)
                pred=make_cam(logits)
                cam_list.append(pred)
        return cam_list

    def get_Q(self, images, ids):
        _, _, h, w = images.shape
        Q_list = []
        affmat_list = []

        for s in self.scale_list:
            target_size = (round(h * abs(s)), round(w * abs(s)))
            H_, W_ = int(
                np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
            scaled_images = F.interpolate(
                images, (H_, W_), mode='bilinear', align_corners=False)
            if(s < 0):
                scaled_images = torch.flip(scaled_images, dims=[3])  # ?dims
            pred = self.Q_model(scaled_images)
            Q_list.append(pred)
            affmat_list.append(calc_affmat(pred))
        return Q_list, affmat_list

    def get_mutiscale_cam(self, cam_list, Q_list, affmat_list, refine_time=0):
        _, _, h, w = Q_list[self.scale_list.index(1.0)].shape
        refine_cam_list = []
        for cam, Q, affmat, s in zip(cam_list, Q_list, affmat_list, self.scale_list):
                if(self.SP_CAM):
                    for i in range(refine_time):
                        cam = refine_with_affmat(cam, affmat)
                    cam = upfeat(cam, Q, 16, 16)

                cam = F.interpolate(cam, (int(h),int(w)), mode='bilinear', align_corners=False)
                if(s <0):
                   cam = torch.flip(cam, dims=[3])#?dims 
                refine_cam_list.append(cam)

        refine_cam = torch.sum(torch.stack(refine_cam_list),dim=0)
        return refine_cam

    def getbest_miou(self, clear=True):
        iou_list = []
        for parm, meter in zip(self.parms,self.meterlist):
           cur_iou, mIoU_foreground, IoU_list, FP, FN = meter.get(clear=clear,detail=True)
        #    log_func(str(cur_iou))
        #    log_func(str(IoU_list))
           
           iou_list.append((cur_iou, parm))
        iou_list.sort(key=lambda x: x[0], reverse=True)
        return iou_list

    def evaluate(self, C_model,Q_model=None):
            self.C_model, self.Q_model = C_model,Q_model
            self.C_model.eval()
            if(self.SP_CAM):
                self.Q_model.eval()
            with torch.no_grad():
                length = len(self.valid_loader)
                for step, (images, image_ids, tags, gt_masks) in enumerate( self.valid_loader ):
                    images = images.cuda()
                    gt_masks = gt_masks.cuda()
                    _,_,h,w = images.shape
                    if(self.SP_CAM):
                         Qs, affmats = self.get_Q(images,image_ids)
                    else:
                        Qs = [images for x in range(len(self.scale_list))]
                        affmats = [None for x in range(len(self.scale_list))]
                    cams_list = self.get_cam(images, image_ids,Qs)
                    mask = tags.unsqueeze(2).unsqueeze(3).cuda()

                    for renum in self.refine_list:
                        refine_cams = self.get_mutiscale_cam(cams_list, Qs,affmats,renum)
                        cams = (make_cam(refine_cams) * mask)
                        cams = F.interpolate(cams, (int(h),int(w)), mode='bilinear', align_corners=False)
                        if(self.save_np_path !=None):
                            np.save(os.path.join(self.save_np_path, image_ids[0]+'.npy'),cams.cpu().numpy())
                        for th in self.th_list:
                            cams[:,0] = th#predictions.max()
                            predictions = torch.argmax(cams,dim=1)
                            for batch_index in range(images.size()[0]):
                                pred_mask = get_numpy_from_tensor(
                                    predictions[batch_index])
                                gt_mask = get_numpy_from_tensor(#cv2.imwrite("1.png",pred_mask*10)
                                    gt_masks[batch_index])
                                gt_mask = cv2.resize(gt_mask,(pred_mask.shape[1],pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                                # self.getbest_miou(clear=False)
                                if (step==600) or (step==200):
                                    print(self.getbest_miou(clear=False))
                                self.meterlist[self.parms.index((renum, th))].add(pred_mask, gt_mask) #,self.meterlist[2].get(clear=False,detail=True)
                                if(self.save_png_path!=None):
                                    cur_save_path = os.path.join(self.save_png_path,str(th))
                                    if not os.path.exists(cur_save_path):
                                           os.mkdir(cur_save_path)
                                    cur_save_path = os.path.join(cur_save_path,str(renum))
                                    if not os.path.exists(cur_save_path):
                                           os.mkdir(cur_save_path)
                                    img_path = os.path.join(cur_save_path,image_ids[batch_index]+'.png')
                                    save_colored_mask(pred_mask, img_path)

                    sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                    sys.stdout.flush()
            self.C_model.train()
            if(self.save_png_path!=None):
                savetxt_path = os.path.join(self.save_png_path,"result.txt")
                with open(savetxt_path, 'wb') as f:
                            for parm, meter in zip(self.parms,self.meterlist):
                                cur_iou = meter.get(clear=False)[-2]
                                f.write('{:>10.2f} {:>10.2f} {:>10.2f}\n'.format(
                                cur_iou , parm[0], parm[1]).encode())
            ret = self.getbest_miou()

            return ret
if __name__ =="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"    
    
    args =get_params()
    time_string =  time.strftime("%Y-%m-%d %H:%M:%S")
    
    args.curtime=time_string

    log_tag = create_directory(f'./experiments/logs/{args.tag}/')
    log_path = log_tag + f'/{args.curtime}.txt'
    if( args.savepng or args.savenpy):
        prediction_tag = create_directory(f'./experiments/predictions/{args.tag}/')
        prediction_path =create_directory( prediction_tag + f'{args.curtime}/')
    
    
    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))
    
    class_num =21 if args.dataset=='voc12' else 81
    if(args.sp_cam):
        model = SP_CAM_Model2('resnet50', num_classes=class_num)
    else:
        model = CAM_Model('resnet50', num_classes=class_num)
        
    model = model.cuda()
    model.train()
    model.load_state_dict(torch.load(args.Cmodel_path))
    
    if(args.sp_cam):
        # network_data = torch.load(
        #     "models_ckpt/SpixelNet_bsd_ckpt.tar")
        # Q_model = fcnmodel.SpixelNet1l_bn(data=network_data).cuda()
        Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        # Q_model = nn.DataParallel(Q_model)
        
        Q_model.load_state_dict(torch.load(args.Qmodel_path))
        Q_model.eval()
    else:
        Q_model=None
    _savepng_path=None
    _savenpy_path=None
    if(args.savepng):
        _savepng_path=create_directory(prediction_path+'pseudo/')
    if(args.savenpy):
        _savenpy_path=create_directory(prediction_path+'camnpy/')
        
    
    evaluatorA = evaluator(dataset='voc12',domain=args.domain,muti_scale=True, SP_CAM=args.sp_cam,save_np_path=_savenpy_path,savepng_path=_savepng_path, refine_list=[0,10,20],th_list=[0.2,0.3,0.35])
    ret = evaluatorA.evaluate(model, Q_model)
    log_func(str(ret))
