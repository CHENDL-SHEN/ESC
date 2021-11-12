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

       



       

 ##10.28  fusion融合模型
class SANET_Model_Fusion(Backbone):
    def __init__(self, model_name, num_classes=21,ch_mid=256,ch_q=64,mid_fusion=False,end_process=False):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
        
        all_h_coords = np.arange(0, 32, 1)
        all_w_coords = np.arange(0, 32, 1)
        curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
        
        self.mid_fusion=mid_fusion
        self.end_process=end_process

        self.coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
        # self.prefusion1=nn.Sequential(
        #         # conv(True, 9,   64, kernel_size=3, stride=2),
        #         # # conv(True, 64,  64, kernel_size=3),
        #         # conv(True, 64, 128, kernel_size=3, stride=2),
        #         # # conv(True, 64, 128, kernel_size=3, stride=2)

        #         conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
        #         conv_bn(True,ch_q,ch_q*2, 3, stride=1),
        #     )
        # self.prefusion2=nn.Sequential(
        #         # conv(True, 128,   256, kernel_size=3, stride=2),
        #         # # conv(True, 64,  64, kernel_size=3),
        #         # conv(True, 256, 256, kernel_size=3, stride=2),
        #         # # conv(True, 64, 128, kernel_size=3, stride=2)

        #         conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
        #         conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),
        #     )
        # self.fusion_mid_conv=conv(True,256+ch_q*2,256, 1,1)
        self.prefusion=nn.Sequential(
                # conv(True, 9,   64, kernel_size=3, stride=2),
                # # conv(True, 64,  64, kernel_size=3),
                # conv(True, 64, 128, kernel_size=3, stride=2),
                # # conv(True, 64, 128, kernel_size=3, stride=2)

                conv_bn(True,9, ch_q,  3, stride=1), ##包含了卷积/正则化/Relu/maxpooling
                conv_bn(True,ch_q,ch_q*2, 3, stride=1),
                conv_bn(True,ch_q*2,ch_q*4, 3, stride=1),
                conv_bn(True,ch_q*4,ch_q*4, 3, stride=1),
            )
     
        
        self.fusion_end_conv=nn.Conv2d(2048+25+9+ch_q*4,ch_mid, 3,padding=1, bias=False)
        # self.qcov=nn.Conv2d(2048+25+9,ch_mid, 3,padding=1, bias=False)
        # self.qcov2=nn.Conv2d(1024, 256, 3,padding=1, bias=False)
        ch_end=ch_mid
        if self.end_process==1:
            self.pre_cls=nn.Conv2d(ch_mid,ch_mid, 1,padding=0, bias=False) ##后续写成处理块{3*3，1*1，SEnet等}
        if self.end_process==2:
            ch_end=int(ch_mid/2)
            self.pre_cls=nn.Sequential(
                nn.Conv2d(ch_mid,ch_end, 3,padding=1, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
                nn.Conv2d(ch_end,ch_end, 1,padding=0, bias=False), ##后续写成处理块{3*3，1*1，SEnet等}
            )
        self.classifier = nn.Conv2d(ch_end, num_classes, 3,padding=1, bias=False)
       
     
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape

        x = self.stage1(inputs)
        x = self.stage2(x) 
       
        # ##fusion on mid layer    
        # p_=self.prefusion1(20*probs)
        # if self.mid_fusion:
        #     fusion=torch.cat([x,p_],dim=1) #x.cpu().detach().numpy().max()
        #     x = self.fusion_mid_conv(fusion) 
        # #####end mid fusion
        x = self.stage3(x) 
        x = self.stage4(x)
        x = self.stage5(x)              ## self.stage3(self.fusion_conv(fusion))  self.stage1(inputs).cpu().detach().numpy().max()
        
        # p_=self.prefusion1(5*probs)
        p_2=self.prefusion(probs)      

        feat_q = getpoolfeatsum(probs)/16*16 #/(16*16) #/16*16 #torch.sum(feat_q,dim=1)
        _,aff_mat = refine_with_q(None,probs,with_aff=True)
       
        x=self.fusion_end_conv(torch.cat([x,feat_q,aff_mat,p_2],dim=1))
        if self.end_process:
            x=self.pre_cls(x)
        logits = self.classifier(x)
  

        return logits  #logits.cpu().detach().numpy().max()
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
    
###这个是用于测试网格结构，上面的用于调参       
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
        elif self.process1==0:
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
        self.classifier = nn.Conv2d(ch_end, num_classes, 3,padding=1, bias=False)
       
     
    
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape

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
        logits = self.classifier(x)
  

        return logits  #logits.cpu().detach().numpy().max()
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
    class SANET_Model_new_base2(Backbone):

    def __init__(self, model_name, num_classes=21,process=0):
        super().__init__(model_name, num_classes, mode='fix',segmentation=False)
        ch_q=64
        self.get_qfeats=nn.Sequential(
                        conv_dilation(True,9, 64,  3, stride=1,dilation=16),
                        conv(True,64, ch_q,  3, stride=2), ##包含了卷积/正则化/Relu/maxpooling
                        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        conv(True,ch_q,ch_q*2, 3, stride=2),
                        # conv_dilation(True,ch_q*2, ch_q*2,  3, stride=1,dilation=8),
                        conv(True,ch_q*2,ch_q*4, 3, stride=2),
                        conv(True,ch_q*4,ch_q*4, 3, stride=2),
                        )
        self.qcov_for_x5_tran_conv=nn.Sequential(
                nn.Conv2d(2048, int(512),3, 1,1),
                nn.Conv2d(int(512), 128, 1, 1, 0),
        )
                
        self.get_tran_conv=nn.Sequential(
        nn.Conv2d(ch_q*4+128, int(128),3, 1,1),
                nn.ReLU(),
                nn.Conv2d(int(128), 18, 1, 1, 0),
            )   
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def get_x5_features(self,inputs):
        x1 = self.stage1(inputs)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return    x5
    

    def get_tconv_cam(self,logits,tconv):
            bg_aff=get_noliner(F.softmax(tconv[:,:9],dim=1))#torch.sum(bg_aff,dim=1).max()
            fg_aff=get_noliner(F.softmax(tconv[:,9:],dim=1))#torch.sum(aff22).min()
            bg= upfeat(logits[:,0:1],bg_aff,1,1)
            fg= upfeat(logits[:,1:],fg_aff,1,1)
            logits =torch.cat([bg,fg],dim=1)
            return logits
    def forward(self, inputs,probs):
        b,c,w,h=probs.shape
        x5 =self.get_x5_features(inputs)

        logits = self.classifier(x5)

        q=self.get_qfeats(probs) 
        x55 = self.qcov_for_x5_tran_conv(x5)
        q=torch.cat([x55.detach(),q],dim=1)
        tconv = self.get_tran_conv(q)

        logits = self.get_tconv_cam(logits,tconv)
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
                if('tran_conv' in name ):
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[4].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[5].append(value)
                elif('qfeats' in name):
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

