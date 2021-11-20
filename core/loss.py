


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    self.fg_c_num =20
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,logits,prob,sailencys,labels):
      
        #region cls_loss
        b, c, h, w = logits.size()
        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#
        cls_loss = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        #endregion
        #region sal_loss
        if(self.args['withQ']):
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
        #endregion
        
    
        return cls_loss,sal_loss