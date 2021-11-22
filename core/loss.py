


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

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
from torch.nn.modules.loss import _Loss
class SP_CAM_Loss(_Loss):
  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               reduction='mean'):
    super(SP_CAM_Loss, self).__init__(size_average, reduce, reduction)
    self.args=args
    self.fg_c_num =20 if  args['dataset'] == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,logits,prob,sailencys,labels):
      
        #region cls_loss
        b, c, h, w = logits.size()
        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#
        cls_loss = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        #endregion
        #region sal_loss
        if(self.args['SP_CAM']):
            sailencys = poolfeat(sailencys, prob, 16, 16).cuda()
        sailencys = F.interpolate(sailencys, size=(h, w))
        sailencys = F.interpolate(sailencys.float(), size=(h, w))

        label_map = labels[:,1:].view(b,  self.fg_c_num , 1, 1).expand(size=(b,  self.fg_c_num , h, w)).bool()#label_map_bg[0,:,0,0]
        # Map selection
        label_map_fg = torch.zeros(size=(b,  self.fg_c_num + 1 , h, w)).bool().cuda()
        label_map_bg = torch.zeros(size=(b,  self.fg_c_num + 1 , h, w)).bool().cuda()

        label_map_bg[:, 0] = True
        label_map_fg[:,1:] = label_map.clone()

        sal_pred = F.softmax(logits, dim=1) 

        iou_saliency = (torch.round(sal_pred[:, 1:].detach()) * torch.round(sailencys)).view(b,  self.fg_c_num , -1).sum(-1) / \
                    (torch.round(sal_pred[:, 1:].detach()) + 1e-04).view(b,  self.fg_c_num , -1).sum(-1)

        valid_channel = (iou_saliency > self.args["tao"]).view(b,  self.fg_c_num , 1, 1).expand(size=(b,  self.fg_c_num , h, w))
        
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

        bg_map = torch.sub(1, bg_map) 
        sal_pred = fg_map * 0.5 + bg_map * (1 - 0.5) 

        sal_loss =F.mse_loss(sal_pred,sailencys)
        
    
        return cls_loss,sal_loss
      
      
class QLoss(_Loss):

  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               relu_t=0.9,
               reduction='mean'):
    super(QLoss, self).__init__(size_average, reduce, reduction)
    self.relu_t=relu_t
    self.relufn =nn.ReLU()
    self.args=args
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,prob,LABXY_feat_tensor,cams,imgids):

            loss_guip, loss_sem_guip, loss_pos_guip = compute_semantic_pos_loss( prob,LABXY_feat_tensor,
                                                        pos_weight= 0.003, kernel_size=16)

            # make superpixel segmentic pseudo label
            cur_masks_1hot_dw=poolfeat(cams,prob)
            cams_bg=cur_masks_1hot_dw.clone()
            cams_fg=cur_masks_1hot_dw.clone()
            cams_bg[:,0]=self.args.th_bg#predictions.max()
            cams_fg[:,0]=self.args.th_bg+self.args.th_step#predictions.max()
            predictions1=torch.argmax(cams_bg,dim=1)
            predictions2=torch.argmax(cams_fg,dim=1)
            fgsort = torch.sort(cur_masks_1hot_dw[:,1:],1,True)[0]
            ignore_masks = predictions1 != predictions2#fgsort[0][0][:,0,0]
            ignore_masks |= (self.args.th_fg*fgsort[:,0]<fgsort[:,1])&(predictions1>0)#
            predictions=predictions1.clone()
            predictions[ignore_masks] =21
            cur_masks_1hot_dw=label2one_hot_torch(predictions.unsqueeze(1), C=22)#masks.max()
            
            b, c, h, w = cur_masks_1hot_dw.shape
            feat_pd = F.pad(cur_masks_1hot_dw, (2, 2, 2, 2), mode='constant', value=0)
            sam_map_list=[]
            diff_map_list=[]
            feat_pd[:,0,:2,:]=1
            feat_pd[:,0,-2:,::]=1
            feat_pd[:,0,:,:2]=1
            feat_pd[:,0,:,-2:]=1

            for i in range(5):
                for j in range(5):
                        ignore_mat=(cur_masks_1hot_dw[:,21]==1)|(feat_pd[:,21,i:i+h,j:j+w]==1)
                        abs_dist=torch.max(torch.abs(feat_pd[:,:21,i:i+h,j:j+w]-feat_pd[:,:21,2:2+h,2:2+w]),dim=1)[0]
                        diff_mat=(abs_dist>0.9)&(~ignore_mat)
                        diff_map_list.append(diff_mat)
                        same_mat=(abs_dist<0.01)&(~ignore_mat)
                        sam_map_list.append(same_mat)
            sam_map=torch.stack(sam_map_list,dim=1)
            center_mask_map_55=torch.zeros((b,5,5,h,w)).bool()
            center_mask_map_55[:,1:4,1:4,:,:]=True
            diff_map=torch.stack(diff_map_list,dim=1).detach()
            center_relu_map= (torch.sum(sam_map,dim=1)>15)#diff_map.sum()/8
            affmat=calc_affmat(prob)
            same_loss= torch.sum(self.relufn(((affmat[:,12]-self.args.relu_t)*center_relu_map)))/b
            diff_loss=torch.sum(self.relufn(((torch.sum(affmat*diff_map,dim=1)))))/b

            return loss_guip, same_loss,2*diff_loss
    