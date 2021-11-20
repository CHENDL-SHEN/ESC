# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model
from tools.general.Q_util import *
from core.models.model_util import conv

#######################################################################
# Normalization
#######################################################################
class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

def conv_dilation(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,dilation=16):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, bias=False,dilation=dilation,padding_mode='circular'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True), 
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True,dilation=dilation,padding_mode='circular'),
            nn.ReLU(inplace=True), 
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

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



class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class CAM_Model(Backbone):
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
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits
    
class SP_CAM_Model(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=32
        self.outc=9*2

        self.get_qfeats=nn.Sequential(
                        conv_dilation(True,9,ch_q,  3, stride=1,dilation=16),
                        conv(True,ch_q, ch_q,  3, stride=2), 
                        conv(True,ch_q,ch_q*2, 3, stride=2),
                        conv(True,ch_q*2,ch_q*4, 3, stride=2),
                        conv(True,ch_q*4,ch_q*4, 3, stride=2),
                        )
                
        self.get_tran_conv_parms=nn.Sequential(
                conv(False,ch_q*4+2048, int(1024),3),
                conv(False,1024,256,3),
                conv(False,256,128,3),
                conv(False,128,  self.outc,1),
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def get_x5_features(self,inputs):
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return    x5
    

    def get_sp_cam(self,logits,deconv_para):
        bg= upfeat(logits[:,0:1],deconv_para[:,:9],1,1)
        fg= upfeat(logits[:,1:],deconv_para[:,9:],1,1)
        logits =torch.cat([bg,fg],dim=1)
        return logits

    def DRM(self,probs,x5):
        q=self.get_qfeats(probs) 
        deconv_parameters = self.get_tran_conv_parms(torch.cat([x5.detach(),q],dim=1))
        bg_para=get_noliner(F.softmax(deconv_parameters[:,:9],dim=1))#torch.sum(fg_aff).max()# fg_aff[0,:,10:20,10:20].detach().cpu().numpy()
        fg_para=get_noliner(F.softmax(deconv_parameters[:,9:],dim=1))#torch.sum(aff22).min()
        deconv_parameters= torch.cat([bg_para,fg_para],dim=1)
        return  deconv_parameters
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x5 =self.get_x5_features(inputs)
        logits = self.classifier(x5)
        deconv_parameters= self.DRM(probs,x5)
        logits = self.get_sp_cam(logits,deconv_parameters)
        return logits
   
    def get_parameter_groups1(self, print_fn=print):
        groups = ([], [], [], [],[],[],[],[])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'model' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)
                    
            # scracthed weights
            else:
                if('qfeats' in name ):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[4].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[5].append(value)
                elif('tran_conv' in name):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[6].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[7].append(value)
                else:
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[2].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[3].append(value)
        return groups

