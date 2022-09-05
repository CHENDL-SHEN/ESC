


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
  def forward(self,fg_cam,sailencys):#sailencys.max()
      
        #region cls_loss
        b, c, h, w = fg_cam.size()
        imgmin_mask=sailencys.sum(1,True)!=0
        #endregion
        #region sal_loss
        # else:
        #   affmat=calc_affmat(prob).detach()
        #   crf_inference()
        #   for i in range(20):
        #     sailencys=refine_with_affmat(sailencys,affmat)
        sailencys = F.interpolate(sailencys.float(), size=(h, w))


        # fg_cam=F.softmax(logits,dim=1)*mask
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
        
    
        return sal_loss
      
      
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

            # # make superpixel segmentic pseudo label
            # cur_masks_1hot_dw=poolfeat(cams,prob)
            # cams_bg=cur_masks_1hot_dw.clone()
            # cams_fg=cur_masks_1hot_dw.clone()
            # cams_bg[:,0]=self.args.th_bg
            # cams_fg[:,0]=self.args.th_bg+self.args.th_step
            # predictions1=torch.argmax(cams_bg,dim=1)
            # predictions2=torch.argmax(cams_fg,dim=1)
            # fgsort = torch.sort(cur_masks_1hot_dw[:,1:],1,True)[0]
            # ignore_masks = predictions1 != predictions2#fgsort[0][0][:,0,0]
            # ignore_masks |= (self.args.th_fg*fgsort[:,0]<fgsort[:,1])&(predictions1>0)#
            # predictions=predictions1.clone()
            # predictions[ignore_masks] =21
            # cur_masks_1hot_dw=label2one_hot_torch(predictions.unsqueeze(1), C=22)#masks.max()
            
            # b, c, h, w = cur_masks_1hot_dw.shape
            # feat_pd = F.pad(cur_masks_1hot_dw, (2, 2, 2, 2), mode='constant', value=0)
            # sam_map_list=[]
            # diff_map_list=[]
            # feat_pd[:,0,:2,:]=1
            # feat_pd[:,0,-2:,::]=1
            # feat_pd[:,0,:,:2]=1
            # feat_pd[:,0,:,-2:]=1

            # for i in range(5):
            #     for j in range(5):
            #             ignore_mat=(cur_masks_1hot_dw[:,21]==1)|(feat_pd[:,21,i:i+h,j:j+w]==1)
            #             abs_dist=torch.max(torch.abs(feat_pd[:,:21,i:i+h,j:j+w]-feat_pd[:,:21,2:2+h,2:2+w]),dim=1)[0]
            #             diff_mat=(abs_dist>0.9)&(~ignore_mat)
            #             diff_map_list.append(diff_mat)
            #             same_mat=(abs_dist<0.01)&(~ignore_mat)
            #             sam_map_list.append(same_mat)
            # sam_map=torch.stack(sam_map_list,dim=1)
            # center_mask_map_55=torch.zeros((b,5,5,h,w)).bool()
            # center_mask_map_55[:,1:4,1:4,:,:]=True
            # diff_map=torch.stack(diff_map_list,dim=1).detach()
            # center_relu_map= (torch.sum(sam_map,dim=1)>15)#diff_map.sum()/8
            # affmat=calc_affmat(prob)
            # press_loss= torch.sum(self.relufn(((affmat[:,12]-self.args.relu_t)*center_relu_map)))/b
            # diff_loss=torch.sum(self.relufn(((torch.sum(affmat*diff_map,dim=1)))))/b

            return loss_guip, loss_guip,loss_guip
    