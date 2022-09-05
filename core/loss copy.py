


from weakref import ref
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
    self.fg_c_num =20 if  args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,logits,prob,sailencys,labels,imgmin_mask):#sailencys.max()
      
        #region cls_loss
        b, c, h, w = logits.size()
        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#
        cls_loss = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        #endregion
        #region sal_loss
        mask=labels[:,:].unsqueeze(2).unsqueeze(3).cuda()
        if(sailencys.shape[1]==1):
          if(self.args.SP_CAM):
              sailencys = poolfeat(sailencys, prob, 16, 16).cuda()
          sailencys =torch.cat([sailencys,1-sailencys],dim=1)
        # else:
        #   affmat=calc_affmat(prob).detach()
        #   crf_inference()
        #   for i in range(20):
        #     sailencys=refine_with_affmat(sailencys,affmat)
        sailencys = F.interpolate(sailencys.float(), size=(h, w))


        # fg_cam=F.softmax(logits,dim=1)*mask
        cam_mask=logits.detach()[:,1:].max(1,True)[0]>0
        fg_cam=make_cam(logits[:,1:])*mask[:,1:]
        bg=1-torch.max(fg_cam,dim=1,keepdim=True)[0]**1
        probs=torch.cat([bg,fg_cam],dim=1)
        # probs=fg_cam
        b,c,h,w=probs.shape
        
        origin_f=F.normalize(sailencys.detach(),dim=1)

        # imgmin_mask=imgmin_mask*cam_mask.detach()#cam_mask.sum()
        probs=probs*imgmin_mask
        f_min=pool_feat_2(probs,origin_f)
        up_f=up_feat_2(probs,f_min)
        # up_f=F.normalize(up_f,dim=1)
        # aaa=(origin_f*up_f).sum(dim=1,keepdim=True)#aaa.min()
        # aaa=torch.clamp(aaa+0.2,0.01,0.99)
        # sal_loss=-torch.log(aaa+1e-5)#*lbl.unsqueeze(-1).unsqueeze(-1).cuda()
        sal_loss =F.mse_loss(up_f,origin_f,reduce=False)
        
        sal_loss=(sal_loss*imgmin_mask).sum()/(torch.sum(imgmin_mask)+1e-5)
        
    
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
  def forward(self,prob,LABXY_feat_tensor,imgids):

            loss_guip, loss_sem_guip, loss_pos_guip = compute_semantic_pos_loss( prob,LABXY_feat_tensor,
                                                        pos_weight= 0.003, kernel_size=16)

            return loss_guip, loss_guip,loss_guip
    