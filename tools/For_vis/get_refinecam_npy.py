import os 
import sys
sys.path.append(r"/media/ders/zhangyumin/PuzzleCAM/")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from operator import mod
import os
from pickle import FALSE, NONE, TRUE
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *


from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime
import glob
from PIL import Image

sys.path.append(r"/media/ders/zhangyumin/superpixel_fcn")
import  core.models as fcnmodel
###################################################################################
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
palette_img_PIL = Image.open(r"VOC2012/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png")
palette = palette_img_PIL.getpalette()
tag_name='train'
SCAM_PATH='finalmodel/base_sal/our-resnest50/SCAMS/'
# OURCAM_PATH='/media/ders/zhangyumin/PuzzleCAM/experiments/res/train_ourcam'
# RECAM_PATH='/media/ders/zhangyumin/PuzzleCAM/experiments/res/train_recam'

class evaluator:
    def __init__(self,domain='train',withQ=True,save_np=False,savepng=False,fast_eval=False,first_check=(22320,20.5)) -> None:
        self.C_model = None
        self.Q_model = None
        self.proxy_Q_model =None
        self.with_Q =withQ
        self.first_check = first_check


        self.fast_eval =fast_eval#eval的时候会缩小尺寸，精度会有所偏差
        self.scale_list  = [0.5,1,1.5,2.0,-0.5,-1,-1.5,-2.0]#- is flip
        if(self.fast_eval):
            self.scale_list  = [0.5,1,1.5,2.0]#- is flip


        self.th_list = [0.25,0.3,0.35]
        #self.refine_list = [0]
        self.refine_list = [0,5,30]

        # self.th_list = [0.3]
        # self.refine_list = [20]
        self.parms=[]
        for renum in self.refine_list:
            for th in self.th_list:
                self.parms.append((renum,th))
        self.meterlist=[ Calculator_For_mIoU('./data/VOC_2012.json') for x in self.parms]


        self.flip   = True

        
        self.batch_size = 8
        self.Top_Left_Crop =False
        self.savept   = False
        self.ptsave_path=[None,None,None]
        self.savepng   = savepng
        self.save_path='experiments/res/cam_test2_qcam/'
        self.save_np=save_np
        self.save_np_path=None
        if not os.path.exists( self.save_path):
                os.mkdir(self.save_path)
        self.tag    = 'test'
        self.domain = domain

        test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(512),
        Transpose_For_Segmentation()
        ])
        if not self.Top_Left_Crop:
            test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Transpose_For_Segmentation()
            ])
            self.batch_size = 1 
        valid_dataset =VOC_Dataset_For_Evaluation('VOC2012/VOCdevkit/VOC2012/', self.domain, test_transform)
        self.valid_loader = DataLoader(valid_dataset, batch_size= self.batch_size, num_workers=1, shuffle=False, drop_last=True)
        pass

    def get_cam(self,images,ids,Qs,tags):
        with torch.no_grad():
            cam_list=[]
            if(type(self.C_model)==str):
                cam_list = torch.load(os.path.join(self.C_model,ids[0]+'.pt'))
            else:
                _,_,h,w = images.shape
                for s,q in zip(self.scale_list,Qs):
                    target_size = (round(h * abs(s)), round(w* abs(s)))
                    scaled_images = F.interpolate(images,target_size, mode='bilinear', align_corners=False)
                    if not self.Top_Left_Crop:
                        H_, W_  = int(np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
                        # scaled_images=nn.ZeroPad2d(padding=(0, W_-target_size[1], 0, H_-target_size[0]))(scaled_images)
                        scaled_images = F.interpolate(scaled_images, (H_,W_), mode='bilinear', align_corners=False)
                    if(s<0):
                        scaled_images =torch.flip(scaled_images,dims=[3])#?dims
                    logits=self.C_model(scaled_images,q)
                    pred=F.softmax(logits,dim=1)
                    #cams = (make_cam(refine_cam) * mask)
                    # for step, (images,image_ids, tags, gt_masks) in enumerate( self.valid_loader ):
                        #cams = self.getpse(cams,Qs,tags,ids)
                    pred_ = pred.clone()
                    mask=tags.unsqueeze(2).unsqueeze(3).cuda()
                    pred_ = (make_cam(pred_) * mask)


                    pred_o=pred_.cpu().numpy()
                    #pred_=np.asfarray(pred_)
                    #pred=pred[0]
                    #pred = np.argmax(pred, axis=0).astype(np.uint8)
                    # np.save(os.path.join('/media/ders/zhangyumin/PuzzleCAM/experiments/res/qcam_npy', ids[0] + '.npy'), pred_o)
                    ourcam=upfeat(pred,q)
                    ourcam_o=ourcam.cpu().numpy()
                    # np.save(os.path.join('/media/ders/zhangyumin/PuzzleCAM/experiments/res/ourcam_npy', ids[0] + '.npy'), ourcam_o)
                    cam_list.append(pred)
        if(self.ptsave_path[0]!=None):
            torch.save(cam_list,os.path.join(self.ptsave_path[0],ids[0]+'.pt'))

        return cam_list
    def get_Q(self,images,ids):
        _,_,h,w = images.shape
        Q_list=[]
        model =self.proxy_Q_model if self.proxy_Q_model!=None else self.Q_model
        if(type(model)==str):
                Q_list = torch.load(os.path.join(model,ids[0]+'.pt'))
        else:
            for s in self.scale_list:
                # if(type(model)==str)
                target_size = (round(h * abs(s)), round(w* abs(s)))
                H_, W_  = int(np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
                scaled_images = F.interpolate(images, (H_,W_), mode='bilinear', align_corners=False)
                if(s<0):
                    scaled_images =torch.flip(scaled_images,dims=[3])#?dims
                pred=model(scaled_images)
                Q_list.append(pred)
        if(self.proxy_Q_model==None):
            refine_Q=Q_list[self.scale_list.index(1.0)]
            if(self.ptsave_path[1]!=None):
                torch.save(Q_list,os.path.join(self.ptsave_path[1],ids[0]+'.pt'))
        else:
            H_, W_  = int(np.ceil(h/16.)*16), int(np.ceil(w/16.)*16)
            scaled_images = F.interpolate(scaled_images, (H_,W_), mode='bilinear', align_corners=False)
            refine_Q=model(scaled_images)
        return Q_list,refine_Q
    def getpse(self,cam_list,Q_list,tags,ids):
        _,_,h,w=Q_list[self.scale_list.index(1.0)].shape
        refine_cam_list=[]
        for cam,Q,s in zip(cam_list,Q_list,self.scale_list):
                if(self.with_Q):
                    cam=upfeat(cam,Q,16,16)
                    #save our cam
                cam = F.interpolate(cam,(int(h),int(w)), mode='bilinear', align_corners=False)
                if(s<0):
                   cam = torch.flip(cam,dims=[3])#?dims 
                refine_cam_list.append(cam)
      
        refine_cam=torch.sum(torch.stack(refine_cam_list),dim=0)


        return refine_cam
    def getbest_miou(self,clear=True):
        best_iou=0
        best_parm=None
        for parm,meter in zip(self.parms,self.meterlist):
            cur_iou=meter.get(clear=clear)[-2]
            if(cur_iou>best_iou):
                best_iou=cur_iou
                best_parm=parm
        return best_iou,best_parm


    def evaluate(self,C_model,Q_model,proxy_Q_model=None):
            if(proxy_Q_model!=None):
                 assert( type(proxy_Q_model) ==str) ,'proxy_Q_model必须是现成的'
            model_list=[ C_model,Q_model,proxy_Q_model]
            for i in range(len(model_list)):
                if(model_list[i]!=None):
                    if not(type(model_list[i])==str):
                        model_list[i].eval()
                    else:
                        modelpath=model_list[i]
                        modelpt_path=model_list[i][:-4]+'ptFOReval/'
                        if not os.path.exists(modelpt_path):
                            os.mkdir(modelpt_path)
                        path_files=glob.glob(pathname=modelpt_path+'*.pt') 
                        if(self.save_np and i==0):
                            self.save_np_path=model_list[i][:-4]+'npy/'
                            if not os.path.exists(self.save_np_path):
                                os.mkdir(self.save_np_path)
                        if(len(path_files)>= len(self.valid_loader)):
                            model_list[i]=modelpt_path
                        else:
                            if(i==0):
                                model_list[i] = SANET_Model('resnest50', num_classes=20 + 1)
                                model_list[i] = model_list[i].cuda()
                            elif(i==1):
                                model_list[i] = fcnmodel.SpixelNet1l_bn().cuda()
                            else:
                                assert False ,'proxy_Q_model必须是现成的'
                            model_list[i].load_state_dict(torch.load(modelpath))
                            model_list[i].eval()
                            if(self.savept):
                                self.ptsave_path[i]=modelpt_path
                          
            self.C_model,self.Q_model,self.proxy_Q_model =model_list

            with torch.no_grad():
                length = len(self.valid_loader)
                time_list=[0,0,0]
                good=True
                for step, (images,image_ids, tags, gt_masks) in enumerate( self.valid_loader ):
                    images = images.cuda()
                    gt_masks = gt_masks.cuda()
                    _,_,h,w= images.shape
                    torch.cuda.synchronize()
                    t1=time.time()
                    Qs,refinQ = self.get_Q(images,image_ids)
                    torch.cuda.synchronize()
                    t2=time.time()
                    cams = self.get_cam(images,image_ids,Qs,tags)
                    torch.cuda.synchronize()

                    mask=tags.unsqueeze(2).unsqueeze(3).cuda()


                    cams = self.getpse(cams,Qs,tags,image_ids)
                    t3=time.time()
   


                    # predictions = self.getpse(cams,Qs)
                    refine_cam = cams.clone()
                    mask=tags.unsqueeze(2).unsqueeze(3).cuda()

                    for renum in range(len(self.refine_list)):
                        if(self.with_Q):
                            refine_cam= refine_with_q(cams,refinQ,self.refine_list[renum])
                        cams = (make_cam(refine_cam) * mask)
                        #cv.imwrite(os.path.join(RECAM_PATH,refinetime,image_ids+'.png'),cams)
                        cams_=cams.cpu().numpy()
                        reqcam=cams.clone()
                            # _,_,H_,W_=reqcam.numpshape()
                        reqcam=poolfeat(reqcam, refinQ).cuda()
                        for batch_index in range(images.size()[0]):
                        #对cam做Q下采样，然后再其线性上采样
                                reqcam=reqcam.cpu().numpy()
                                # reqcam = reqcam * mask
                                retime=self.refine_list[renum]
                                if not os.path.exists(SCAM_PATH):
                                    os.mkdir(SCAM_PATH)
                                if not os.path.exists(SCAM_PATH+"CAM512_"+str(retime)):
                                    os.mkdir(SCAM_PATH+"CAM512_"+str(retime))
                                if not os.path.exists(SCAM_PATH+"QCAM32_"+str(retime)):
                                    os.mkdir(SCAM_PATH+"QCAM32_"+str(retime))
                                np.save(os.path.join(SCAM_PATH+"CAM512_"+str(retime), image_ids[batch_index]+'.npy'), cams_[batch_index])
                                np.save(os.path.join(SCAM_PATH+"QCAM32_"+str(retime), image_ids[batch_index]+'.npy'), reqcam[batch_index])




                    # self.getbest_miou()
                    torch.cuda.synchronize()
                    t4=time.time()
                    if(step==self.first_check[0]):
                        if(self.getbest_miou(clear=False)[0]<self.first_check[1]):
                            good=False
                            break

                    sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                    sys.stdout.flush()
                    time_list[0]+=t2-t1
                    time_list[1]+=t3-t2
                    time_list[2]+=t4-t3
            # print(time_list)
            for m in [self.C_model, self.Q_model]:
                if(m!=None):
                    if not(type(m)==str):
                        m.train()
            ret = self.getbest_miou()
            if not good:
                ret =(ret[0], self.first_check)

            return ret


if __name__ =='__main__':
    from core.spnetwork_new import *

    model = SANET_Model_new_base('resnest50', num_classes=20 + 1)
    model = model.cuda()
    model.train()
    model.load_state_dict(torch.load('finalmodel/base_sal/our-resnest50/2021-11-12 15:09:03.pth'))

    evaluatorA = evaluator(domain='train_aug',withQ=True, fast_eval=False,savepng=FALSE,save_np=False)
    # evaluatorA = psemakeForQ.evaluator(domain='train_aug',savepng=True,savept=True,th_bg=[0.05,0.1,0.03],th_step=[0.4,0.5,0.6],th_fg=[0.05,0.1,0.2])

    ret = evaluatorA.evaluate(model,'experiments/models/bestQ.pth')
    print(ret[0])
    print(ret[1])