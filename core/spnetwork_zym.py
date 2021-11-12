# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from hashlib import sha1
import math
from numpy.core.numeric import False_

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torchvision.transforms.transforms import ToPILImage
from .networks import Backbone
from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model
from .deeplab_utils import ASPP, Decoder
from .aff_utils import PathIndex
from .puzzle_utils import tile_features, merge_features
from tools.ai.torch_utils import resize_for_tensors
from tools.general.Q_util import *
from core.models.model_util import conv

#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

class SANET_Q_Attention(Backbone):
    def __init__(self, model_name, num_classes=21,process=0,kernel_size1=3,kernel_size2=1):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        pad1=int((kernel_size1-1)/2)
        pad2=int((kernel_size2-1)/2)
        self.process=process
        if process==0:   #(1,0)
            self.qcov33=nn.Sequential(
                    # conv_bn(True,34+9,128,kernel_size1),
                    # conv_bn(True,128 ,9  ,kernel_size2),
                    # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    # nn.Softmax(dim=1),
                    nn.Conv2d(34+9, int(128),kernel_size1, 1,pad1),
                    # nn.ReLU(),
                    nn.Conv2d(int(128), 9, kernel_size2, 1, pad2),
                    nn.Softmax(dim=1)
                    ) 
        elif process==1:     #(1,0)
            self.qcov55=nn.Sequential(
                conv_bn(True,34+25,128,kernel_size1),
                conv_bn(True,128 ,25  ,kernel_size2),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                # nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
                # # nn.BatchNorm2d(128),
                # # nn.ReLU(),
                # nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
                # # nn.BatchNorm2d(25),
                # # nn.ReLU(),
                nn.Softmax(dim=1)
                ) 
        elif process==2:   #(1,1)
            self.qcov33=nn.Sequential(
                    nn.Conv2d(34+9, int(128),kernel_size1, 1,pad1),
                    # nn.ReLU(),
                    nn.Conv2d(int(128), 9, kernel_size2, 1, pad2),
                    nn.Softmax(dim=1)
                    )   
            self.qcov55=nn.Sequential(
                nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
                nn.BatchNorm2d(25),
                nn.ReLU(),
                nn.Softmax(dim=1)
                )
            self.conv=nn.Conv2d(2048*2, 2048, 1,padding=0, bias=False)
        else:
            self.qcov55=nn.Sequential(
            conv_bn(True,34+25,128,kernel_size1),
            conv_bn(True,128 ,25  ,kernel_size2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
            # # nn.BatchNorm2d(128),
            # # nn.ReLU(),
            # nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
            # # nn.BatchNorm2d(25),
            # # nn.ReLU(),
            nn.Softmax(dim=1)
            ) 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    # def get_eu(self,features):
    #         b, c, h, w = features.shape
    #         feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
    #         diff_map_list=[]
    
    #         for i in range(3):
    #             for j in range(3):
    #                     abs_dist=F.pairwise_distance(feat_pd[:,:21,i:i+h,j:j+w],feat_pd[:,:21,2:2+h,2:2+w],2)
    #                     diff_map_list.append(abs_dist)
    #         ret = torch.stack(diff_map_list,dim=1)
    #         return ret
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        
       
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        with torch.no_grad():
            dist_f33=get_eu(x4,3)
            dist_f55=get_eu(x4,5)
        if self.process==0:
           
            aff22 = self.qcov33(torch.cat([dist_f33.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
       
        elif self.process==1:
           
            aff22 = self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
        else :
           
            aff22 = self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
        # elif self.process==2:       
        #     dist_f33=get_eu(x4,3)
        #     dist_f55=get_eu(x4,5)
        #     aff33 = self.qcov33(torch.cat([dist_f33.detach(),feat_q,aff_mat],dim=1))
        #     x33= upfeat(x5,aff33,1,1)
        #     aff55= self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
        #     x55= rewith_affmat(x5,aff55)
        #     x=torch.cat([x33,x55],dim=1)
        #     x=self.conv(x)
       
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
  
    

        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

     
    
    def forward(self, inputs):


        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        logits = self.classifier(x)
        
        return logits

class SENet(nn.Module):

    def __init__(self, in_chnls, out_chnls,ratio=16):
        super(SENet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        # self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, out_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

        
class SANET_Model_new(Backbone):
       def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        
        if process==0:       ##1
            self.qcov2=nn.Sequential(
                    nn.Conv2d(34+1024, int(256),3, 1,1),
                    # nn.ReLU(),
                    nn.Conv2d(int(256), 9, 1, 1, 0),
                    nn.Softmax(dim=1)
                    )   
    
            self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
            self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)
        elif process==1:    
            # self.qcov1=nn.Conv2d(9+25, 256, 1,padding=0, bias=False)
            # self.qcov2=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)
            # self.qcov1=nn.Conv2d(9, 256, 3,padding=1, bias=False)
            self.qcov1=nn.Conv2d(9+25, 128, 1,padding=0, bias=False)
            self.qcov2=nn.Conv2d(128, 256, 1,padding=0, bias=False)
        elif process==2:
            # self.qcov1=nn.Conv2d(9+25+3, 256, 1,padding=0, bias=False)
            # self.qcov2=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)
            self.qcov1=nn.Conv2d(9+25, 512, 1,padding=0, bias=False)
            self.qcov2=nn.Conv2d(512, 256, 1,padding=0, bias=False)
        elif process==3:
            # self.qcov=nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(256, int(256/16),1, 1, 0),
            #     nn.ReLU(),
            #     nn.Conv2d(int(256/16), 256, 1, 1, 0),
            #     nn.Sigmoid()
            #     )   
            self.qcov1=nn.Conv2d(9+25, 256, 1,padding=0, bias=False)
            self.qcov2=nn.Conv2d(256+2048, 256, 1,padding=0, bias=False)
        elif process==4:  ##1
            # self.qcov=nn.Sequential(
            #     nn.Conv2d(9+25, 256, 1,padding=0, bias=False),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU()
            # )
            self.qcov1=nn.Conv2d(9+25, 256, 1,padding=0, bias=False)
            # self.qcov=nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.ReLU(),
            #     nn.Conv2d(34, 256, 1, 1, 0),
            #     nn.Sigmoid()
            #     )   
            
          
       
          
        
        
        self.classifier = nn.Conv2d(512+2048, num_classes, 1, bias=False)
    
        def forward(self, inputs,probs):
            b,c,w,h=probs.shape
            x = self.stage1(inputs)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)       
            
            feat_q = getpoolfeatsum(probs)
            _,aff_mat = refine_with_q(None,probs,with_aff=True)
            if self.process==0:   #3单一1*1
                q=self.qcov(torch.cat([feat_q,aff_mat],dim=1))
                xq=torch.cat([x,q],dim=1)
                xq_se=self.SE(xq)
                xq=xq*xq_se
                xq=self.conv(xq)
                x=torch.cat([x,xq],dim=1)
                
            
            elif self.process==1:
                q=torch.cat([feat_q,aff_mat],dim=1)
                q=self.qcov1(q)
                q=self.qcov2(q)
                x=torch.cat([x,q],dim=1)
                # q=torch.cat([feat_q,aff_mat],dim=1) ##突破1
                # q=self.qcov1(q)
                # x=torch.cat([x,q],dim=1)
                # x=self.qcov2(x)
                self.qcov1=nn.Conv2d(3, 256, 1,padding=0, bias=False)
            elif self.process==2:
                # q=poolfeat(inputs,probs)
                # q=torch.cat([q,feat_q,aff_mat],dim=1)
                # q=self.qcov1(q)
                # x=torch.cat([x,q],dim=1)
                # if self.se:
                #     x_se=self.SE(x)
                #     x=x*x_se
                # x=self.qcov2(x)
                q=torch.cat([feat_q,aff_mat],dim=1)
                q=self.qcov1(q)
                q=self.qcov2(q)
                x=torch.cat([x,q],dim=1)
            elif self.process==3:
                q=torch.cat([feat_q,aff_mat],dim=1)
                q=self.qcov1(q)
                x=torch.cat([x,q],dim=1)
                x=self.qconv2(x)
            elif self.process==4:
                q=torch.cat([feat_q,aff_mat],dim=1)
                q=self.qcov1(q)
                x=torch.cat([x,q],dim=1)
                # if self.se:
                #     x_se=self.SE(x)
                #     x=x*x_se
            # x_se=self.SE(x)
            # x=x*x_se
        
            # x = self.qcov(x)
            logits = self.classifier(x)

class SANET_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
  
        all_h_coords = np.arange(0, 32, 1)
        all_w_coords = np.arange(0, 32, 1)
        curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

        self.coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])

        self.qcov=nn.Conv2d(2048+9+25, 256, 3,padding=1, bias=False)
        # self.qcov2=nn.Conv2d(1024, 256, 3,padding=1, bias=False)

        self.classifier = nn.Conv2d(256, num_classes, 1, bias=False)

     
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        with torch.no_grad:
            x = self.stage1(inputs)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
        

        # all_XY_feat = (torch.from_numpy(
        #     np.tile( self.coord_tensor, (inputs.shape[0], 1, 1, 1)).astype(np.float32)).cuda())
        # xy22= upfeat(all_XY_feat,probs)
        # xy22= F.interpolate(xy22,(32,32),mode='bilinear',align_corners=False)
        # all_sum =torch.ones((b,1,w,h))
        # all_sum = poolfeat(all_sum,probs)
        feat_q = getpoolfeatsum(probs)/16*16 #/16*16 #torch.sum(feat_q,dim=1)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        x=self.qcov(torch.cat([x,feat_q,aff_mat],dim=1))
        # x = self.qcov(x)
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


def get_eu(features,kernel_size=3):
        b, c, h, w = features.shape
        pad=int((kernel_size-1)/2)
        feat_pd = F.pad(features, (pad, pad, pad, pad), mode='constant', value=0)
        diff_map_list=[]

        for i in range(kernel_size):
            for j in range(kernel_size):
                    abs_dist=F.pairwise_distance(feat_pd[:,:21,i:i+h,j:j+w],feat_pd[:,:21,2:2+h,2:2+w],2)
                    diff_map_list.append(abs_dist)
        ret = torch.stack(diff_map_list,dim=1)
        return ret     
        
class SANET_Q_Attention(Backbone):
    def __init__(self, model_name, num_classes=21,process=0,kernel_size1=3,kernel_size2=1):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        pad1=int((kernel_size1-1)/2)
        pad2=int((kernel_size2-1)/2)
        self.process=process
        if process==0:   #(1,0)
            self.qcov33=nn.Sequential(
                    # conv_bn(True,34+9,128,kernel_size1),
                    # conv_bn(True,128 ,9  ,kernel_size2),
                    # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    # nn.Softmax(dim=1),
                    nn.Conv2d(34+9, int(128),kernel_size1, 1,pad1),
                    # nn.ReLU(),
                    nn.Conv2d(int(128), 9, kernel_size2, 1, pad2),
                    nn.Softmax(dim=1)
                    ) 
        elif process==1:     #(1,0)
            self.qcov55=nn.Sequential(
                conv_bn(True,34+25,128,kernel_size1),
                conv_bn(True,128 ,25  ,kernel_size2),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                # nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
                # # nn.BatchNorm2d(128),
                # # nn.ReLU(),
                # nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
                # # nn.BatchNorm2d(25),
                # # nn.ReLU(),
                nn.Softmax(dim=1)
                ) 
        elif process==2:   #(1,1)
            self.qcov33=nn.Sequential(
                    nn.Conv2d(34+9, int(128),kernel_size1, 1,pad1),
                    # nn.ReLU(),
                    nn.Conv2d(int(128), 9, kernel_size2, 1, pad2),
                    nn.Softmax(dim=1)
                    )   
            self.qcov55=nn.Sequential(
                nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
                nn.BatchNorm2d(25),
                nn.ReLU(),
                nn.Softmax(dim=1)
                )
            self.conv=nn.Conv2d(2048*2, 2048, 1,padding=0, bias=False)
        else:
            self.qcov55=nn.Sequential(
            conv_bn(True,34+25,128,kernel_size1),
            conv_bn(True,128 ,25  ,kernel_size2),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(34+25, int(128),kernel_size1, 1,pad1),
            # # nn.BatchNorm2d(128),
            # # nn.ReLU(),
            # nn.Conv2d(int(128), 25, kernel_size2, 1, pad2),
            # # nn.BatchNorm2d(25),
            # # nn.ReLU(),
            nn.Softmax(dim=1)
            ) 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    # def get_eu(self,features):
    #         b, c, h, w = features.shape
    #         feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
    #         diff_map_list=[]
    
    #         for i in range(3):
    #             for j in range(3):
    #                     abs_dist=F.pairwise_distance(feat_pd[:,:21,i:i+h,j:j+w],feat_pd[:,:21,2:2+h,2:2+w],2)
    #                     diff_map_list.append(abs_dist)
    #         ret = torch.stack(diff_map_list,dim=1)
    #         return ret
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        
       
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        with torch.no_grad():
            dist_f33=get_eu(x4,3)
            dist_f55=get_eu(x4,5)
        if self.process==0:
           
            aff22 = self.qcov33(torch.cat([dist_f33.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
       
        elif self.process==1:
           
            aff22 = self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
        else :
           
            aff22 = self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
            x= upfeat(x5,aff22,1,1)
        # elif self.process==2:       
        #     dist_f33=get_eu(x4,3)
        #     dist_f55=get_eu(x4,5)
        #     aff33 = self.qcov33(torch.cat([dist_f33.detach(),feat_q,aff_mat],dim=1))
        #     x33= upfeat(x5,aff33,1,1)
        #     aff55= self.qcov55(torch.cat([dist_f55.detach(),feat_q,aff_mat],dim=1))
        #     x55= rewith_affmat(x5,aff55)
        #     x=torch.cat([x33,x55],dim=1)
        #     x=self.conv(x)
       
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_fusion(Backbone):
    def __init__(self, model_name, num_classes=21,ch_mid=256,ch_q=64,process1=1,process2=1):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
    

    
        self.process1=process1
        self.process2=process2 
        

        self.prefusion=nn.Sequential(
                # conv(True, 9,   64, kernel_size=3, stride=2),
                # # conv(True, 64,  64, kernel_size=3),
                # conv(True, 64, 128, kernel_size=3, stride=2),
                # # conv(True, 64, 128, kernel_size=3, stride=2)
                    conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                    conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                    conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                    conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
            

        #
        ch_fusion1=9+25+ch_q*4
        
        ch_end1=ch_fusion1
        if self.process1==1:
            self.pre1=nn.Conv2d(ch_fusion1,ch_end1, 1,padding=0, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process1==2:
            self.pre1=nn.Sequential(
                nn.Conv2d(ch_fusion1,ch_end1, 3,padding=1, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
                nn.Conv2d(ch_end1,ch_end1, 1,padding=0, bias=False))##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process1==3:
            self.pre1=nn.Sequential(
            nn.Conv2d(ch_fusion1,ch_end1, 1,padding=0, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
            nn.Conv2d(ch_end1,ch_end1, 3,padding=1, bias=False)) ##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process1==4:
            self.pre1=nn.Sequential(
            nn.Conv2d(ch_fusion1,ch_end1, 3,padding=1, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
            nn.Conv2d(ch_end1,ch_end1, 3,padding=1, bias=False)) ##后续写成处理块{3*3，1*1，SEnet等}
        elif  self.process1==5:
            self.pre1=nn.Conv2d(ch_fusion1,ch_end1, 3,padding=1, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        else :
            self.pre1=nn.Conv2d(3,ch_end1, 1,padding=0, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        
        
        ch_fusuion=2048+ch_end1
        
        self.fusion_end_conv=nn.Conv2d(ch_fusuion,ch_mid, 1,padding=0, bias=False)
        self.qconv=nn.Conv2d(ch_fusuion,ch_mid, 3,padding=1, bias=False)
        # self.qcov2=nn.Conv2d(1024, 256, 3,padding=1, bias=False)
        ch_end=ch_mid
        if self.process2==1:
            self.pre_cls=nn.Conv2d(ch_mid,ch_end, 1,padding=0, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process2==2:
            self.pre_cls=nn.Sequential(
                nn.Conv2d(ch_mid,ch_end, 3,padding=1, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
                nn.Conv2d(ch_end,ch_end, 1,padding=0, bias=False))##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process2==3:
            self.pre_cls=nn.Sequential(
            nn.Conv2d(ch_mid,ch_end, 1,padding=0, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
            nn.Conv2d(ch_end,ch_end, 3,padding=1, bias=False)) ##后续写成处理块{3*3，1*1，SEnet等}
        elif self.process2==4:
            self.pre_cls=nn.Conv2d(ch_mid,ch_end, 3,padding=1, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        else:
            self.pre_cls=nn.Sequential(
            nn.Conv2d(ch_mid,ch_end, 3,padding=1, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
            nn.Conv2d(ch_end,ch_end, 3,padding=1, bias=False)) ##后续写成处理块{3*3，1*1，SEnet等}
        self.classifier1 = nn.Conv2d(ch_end, num_classes, 3,padding=1, bias=False)
        
        

    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        with torch.no_grad():
            x = self.stage1(inputs)
            x = self.stage2(x) 
            
            
            x = self.stage3(x) 

            x = self.stage4(x)
            x = self.stage5(x)              ## self.stage3(self.fusion_conv(fusion))  self.stage1(inputs).cpu().detach().numpy().max()
            
        
        feat_q = getpoolfeatsum(probs)/16*16 #/(16*16) #/16*16 #torch.sum(feat_q,dim=1)
        _,aff_mat = refine_with_q(inputs,probs,with_aff=True)
        if self.process1:   
            conv_q=self.prefusion(probs)
            q=torch.cat([feat_q,aff_mat,conv_q],dim=1)
            q=self.pre1(q) 
        else:
            # q=torch.cat([feat_q,aff_mat],dim=1)
            q=poolfeat(inputs,probs)
            # xq,aff_mat = refine_with_q(x,conv_q,with_aff=True)
            q=self.pre1(q)
            # xq=rewith_affmat(x,xq)
        x=torch.cat([x,q],dim=1) 

        if self.process2:
            
            # x_SE=self.SE_end_conv(x)
            # x=x_SE*xdx
            x=self.fusion_end_conv(x)
            x=self.pre_cls(x)
        else:

            # x_SE=self.SE_end_conv(x)
            # x=x_SE*x
            x=self.qconv(x)
        logits = self.classifier1(x)
    

        return logits  #logits.cpu().detach().numpy().max()
            # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
            

    

    
class SANET_Model_noconv(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+1024, int(256),3, 1,1),
                # nn.ReLU(),
                nn.Conv2d(int(256), 9, 1, 1, 0),
                nn.Softmax(dim=1)
                )   
 
        self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        self.classifier1 = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        
        # feat_q = getpoolfeatsum(probs)
        # _,aff_mat = refine_with_q(None,probs,with_aff=True)
            
           

        # aff22 = self.qcov2(torch.cat([x4,feat_q,aff_mat],dim=1))
        # x= upfeat(x5,aff22,1,1)
        #  x=rewith_affmat(x5,aff22)
        logits = self.classifier1(x5)
        logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits=poolfeat(logits,probs)
        
        return logits


class SANET_Model_new4(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process

        self.qcov_for_x4=nn.Sequential(
                nn.Conv2d(1024, int(512),3, 1,1),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                

        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                # nn.ReLU(),
                nn.Conv2d(int(128), 9, 1, 1, 0),
                nn.Softmax(dim=1)
                )   
 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        with torch.no_grad():
            x1 = self.stage1(inputs)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
            
           
        x4 =self.qcov_for_x4(x4.detach())
        aff22 = self.qcov2(torch.cat([x4,feat_q,aff_mat],dim=1))
        x= upfeat(x5,aff22,1,1)
        # x=rewith_affmat(x5,aff22)
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new5(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process

        self.qcov_for_x4=nn.Sequential(
                nn.Conv2d(1024+2048, int(512),3, 1,1),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                

        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 9, 1, 1, 0),
                nn.Softmax(dim=1)
                )   
 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        with torch.no_grad():
            x1 = self.stage1(inputs)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)       
        
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
            
           
        x4 =self.qcov_for_x4(torch.cat([x4.detach(),x5.detach()],1))
        aff22 = self.qcov2(torch.cat([x4,feat_q,aff_mat],dim=1))
        x= upfeat(x5,aff22,1,1)
        # x=rewith_affmat(x5,aff22)
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


def get_noliner(features):
            b, c, h, w = features.shape
            if(c==9):
                feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
            elif(c==25):
                feat_pd = F.pad(features, (2, 2, 2, 2), mode='constant', value=0)

            diff_map_list=[]
            nn=int(math.sqrt(c))
            for i in range(nn):
                for j in range(nn):
                        diff_map_list.append(feat_pd[:,i*nn+j,i:i+h,j:j+w])
            ret = torch.stack(diff_map_list,dim=1)
            return ret

class SANET_Model_new6(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov1=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 25, 1, 1, 0),
                nn.Softmax(dim=1)
                )   
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 25, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        # x33 =self.qcov_for_x3(x3.detach())
        # aff11 = self.qcov1(torch.cat([x33,feat_q,aff_mat],dim=1))



        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        # aff22=get_noliner(aff22)#torch.sum(aff22).min()
        logits=rewith_affmat(logits,aff22)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits
class SANET_Model_new6_1(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(1024),3, 1,1),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1024, int(256),3, 1,1),
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(int(256), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                # nn.ReLU(),
                nn.Conv2d(int(128), 25, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
 
        # self.qcov1=nn.Conv2d(34+1024, 256, 1,padding=0, bias=False)
        # self.conv=nn.Conv2d(256+2048, 512, 1,padding=0, bias=False)

          
        
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        # x33 =self.qcov_for_x3(x3.detach())
        # aff11 = self.qcov1(torch.cat([x33,feat_q,aff_mat],dim=1))



        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        # aff22=get_noliner(aff22)#torch.sum(aff22).min()
        logits=rewith_affmat(logits,aff22)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


class SANET_Model_new7(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new7(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

###8-10改内部结构 (11是senet)
class SANET_Model_new8(Backbone):  
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(1024),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(1024), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


class SANET_Model_new9(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(256),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(256), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits
class SANET_Model_new10(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
      
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.seconv=nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(34+128, (34+128)//16, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d((34+128)//16, 34+128, 1, 1, 0),
                nn.Sigmoid()
                )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        q=torch.cat([x55,feat_q,aff_mat],dim=1)
        q_se=self.seconv(q)
        q=q*q_se
        aff22 = self.qcov2(q)
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


class SANET_Model_new11(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new12(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


class SANET_Model_new13(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new14(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
                        
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        q=self.prefusion(probs)
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,q],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits
class SANET_Model_new15(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=128
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
                        
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.BatchNorm2d(18),
                nn.ReLU(),
                nn.Softmax(dim=1)

            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

        q=self.prefusion(probs) 
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,q],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits



class SANET_Model_new16(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
        self.seconv=nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(ch_q*4,+128, int((ch_q*4+128)//16), 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(int((ch_q*4+128)//16), ch_q*4,+128, 1, 1, 0),
                nn.Sigmoid()
                )                 
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

        q=self.prefusion(probs) 
        x55 = self.qcov_for_x5(x5.detach())
        q=torch.cat([x55,q],dim=1)
        q_se=self.seconv(q)
        q=q*q_se
        aff22 = self.qcov2(q)

        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits


class SANET_Model_new17(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
    
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(int(512), 128, 1, 1, 0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
        
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(34+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
        feat_q = getpoolfeatsum(probs)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
        b,c,w,h=probs.shape
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,feat_q,aff_mat],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new18(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
                        
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        q=self.prefusion(probs)
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,q],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new19(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
                        
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        q=self.prefusion(probs)
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,q],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class SANET_Model_new20(Backbone):
    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.prefusion=nn.Sequential(
                    # conv(True, 9,   64, kernel_size=3, stride=2),
                    # # conv(True, 64,  64, kernel_size=3),
                    # conv(True, 64, 128, kernel_size=3, stride=2),
                    # # conv(True, 64, 128, kernel_size=3, stride=2)
                        conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                        conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                        conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                        conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),)
                        
        self.process=process
        # self.qcov_for_x3=nn.Sequential(
        #             nn.Conv2d(512, int(256),3, 1,1),
        #             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #             # nn.Conv2d(512, int(256),3, 1,1),
        #             # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #             # nn.ReLU(),
        #             nn.Conv2d(int(256), 128, 1, 1, 0),
        #     )
        self.qcov_for_x5=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(512, int(256),3, 1,1),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # nn.ReLU(),
                nn.Conv2d(int(512), 128, 3, 1, 1),
        )
                
        self.qcov2=nn.Sequential(
                nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 3, 1, 1),
                nn.Softmax(dim=1)
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs,probs):
      
        b,c,w,h=probs.shape
        q=self.prefusion(probs)
        # with torch.no_grad():
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x4= upfeat(x4,aff11,1,1)

        x5 = self.stage5(x4)       
        

            
        x55 = self.qcov_for_x5(x5.detach())
        aff22 = self.qcov2(torch.cat([x55,q],dim=1))
        logits = self.classifier(x5)
        # logits= upfeat(logits,aff22,1,1)
        bg_aff=get_noliner(aff22[:,:9])#torch.sum(aff22).min()
        fg_aff=get_noliner(aff22[:,9:])#torch.sum(aff22).min()

        # logits=rewith_affmat(logits,aff22)
        bg= upfeat(logits[:,0:1],bg_aff,1,1)
        fg= upfeat(logits[:,1:],fg_aff,1,1)

        logits =torch.cat([bg,fg],dim=1)

        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits