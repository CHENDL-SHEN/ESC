# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *
from core.sync_batchnorm.batchnorm import _unsqueeze_ft

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
from nni.utils import merge_parameter
import nni

os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,1,2,3"

import  core.models as fcnmodel
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
start_time=datetime.now().strftime('%Y-%m-%d%H:%M:%S')

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

parser = argparse.ArgumentParser()
class DottableDict(dict): 
  def __init__(self, *args, **kwargs): 
    dict.__init__(self, *args, **kwargs) 
    self.__dict__ = self 
  def allowDotting(self, state=True): 
    if state: 
      self.__dict__ = self 
    else: 
      self.__dict__ = dict() 
def get_params():
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--data_dir', default='VOC2012/VOCdevkit/VOC2012/', type=str)
        
    ###############################################################################
    # Network
    ###############################################################################
    parser.add_argument('--architecture', default='DeepLabv3+', type=str)
    parser.add_argument('--backbone', default='resnest50', type=str)
    parser.add_argument('--mode', default='fix', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)

    ###############################################################################
    # Hyperparameter
    ###############################################################################
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epoch', default=15, type=int) #***********调#@3

    parser.add_argument('--lr', default=0.001, type=float) #***********调#@3
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)

    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--downsize', default=16, type=int)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--th_bg', default=0.1, type=float) #@1 *6  #0.03,0.05,0.1
    parser.add_argument('--th_step', default=0.4, type=float)#0.4,0.5,0.6
    parser.add_argument('--th_fg', default=0.05, type=float)#0.1,0.05,0.03
    parser.add_argument('--relu_t', default=0.75, type=float)#0.75,0.7,0.8
    parser.add_argument('--K_same', default=2, type=float)#1,2,4,8                     1
    parser.add_argument('--relu_diff', default=2, type=int) #0,1,2,4,8                 3
    parser.add_argument('--domain', default='train_aug', type=str)#***********调#@2
    parser.add_argument('--pretrain', default=True, type=str2bool)#***********调#@4
    parser.add_argument('--pse_path', default='experiments/models/baseline_new_eval/2021-10-14 09:59:48npy', type=str)#***********调#@5

    parser.add_argument('--tag', default='train_Q_withPse_SP', type=str)

    parser.add_argument('--label_name', default='AffinityNet@Rresnest269@Puzzle@train@beta=10@exp_times=8@rw@crf=0@color', type=str)
    args = parser.parse_args()
    return args
from torch.nn.modules.loss import _Loss
class SetLoss(_Loss):

  def __init__(self,
               args,
               size_average=None,
               reduce=None,
               relu_t=0.9,
               reduction='mean'):
    super(SetLoss, self).__init__(size_average, reduce, reduction)
    self.relu_t=relu_t
    self.relufn =nn.ReLU()
    self.args=args
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()
  def forward(self,prob,LABXY_feat_tensor,cams,imgids):



            loss_guip, loss_sem_guip, loss_pos_guip = compute_semantic_pos_loss( prob,LABXY_feat_tensor,
                                                        pos_weight= 0.003, kernel_size=16)
            # return loss_pos_guip,loss_sem_guip,loss_sem_guip*0

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

            refine_masks_1hot,affmat=refine_with_q(cur_masks_1hot_dw,prob,10,with_aff= True)
            # refine_masks_1hot_up=upfeat(refine_masks_1hot,prob)
            
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
                        ignore_mat=(cur_masks_1hot_dw[:,21]>0.9)|(feat_pd[:,21,i:i+h,j:j+w]>0.9)
                        abs_dist=torch.max(torch.abs(feat_pd[:,:21,i:i+h,j:j+w]-feat_pd[:,:21,2:2+h,2:2+w]),dim=1)[0]
                        diff_mat=(abs_dist>0.9)&(~ignore_mat)
                        diff_map_list.append(diff_mat)
                        same_mat=(abs_dist<0.01)&(~ignore_mat)
                        sam_map_list.append(same_mat)
            sam_map=torch.stack(sam_map_list,dim=1)
            # center_mask_map_55=torch.zeros((b,5,5,h,w)).bool()
            # center_mask_map_55[:,1:4,1:4,:,:]=True
            # center_mask_map=center_mask_map_55.reshape((b,25,h,w)).cuda()

            diff_map=torch.stack(diff_map_list,dim=1).detach()

            # relu_loss_diff2=torch.sum(affmat[diff_map])
            # diff_map[center_mask_map]=False   

            # cv2.imwrite('1.png',cams[5][0].detach().cpu().numpy()*255)   # cv2.imwrite('2.png',ignore_masks[5].detach().cpu().numpy()*255)
            # cv2.imwrite('4.png',(torch.sum(diff_map[5],dim=0)>1).detach().cpu().numpy()*255) # poolfeat(cur_masks_1hot,prob)[5,:,16,15]  

            # sam_map_bf=(sam_map*(cur_masks_1hot_dw[:,21:]==0))#affmat[0,:,10,10].reshape(b,5,5,h,w)[:,1:4,1:4])
            # sam_map*=center_mask_map
            center_relu_map= (torch.sum(sam_map,dim=1)>15)#diff_map.sum()/8
            center_diff_relu_map= (torch.sum(diff_map,dim=1)>self.args.relu_diff) #cv2.imwrite('1.png',center_diff_relu_map[5].detach().cpu().numpy()*100)  
            # relu_loss_sam=torch.sum(self.relufn(((0.001-affmat)*sam_map)))#torch.sum(affmat,dim=1)affmat[:,12].max()

            loss3= torch.sum(self.relufn(((affmat[:,12]-self.args.relu_t)*center_relu_map)))/b
            pse_loss=torch.sum(self.relufn(((torch.sum(affmat*diff_map,dim=1))*center_diff_relu_map)))/b
            # nofeat=~cur_masks_1hot_dw.bool()
            #  =torch.sum(nofeat*refine_masks_1hot)/b
            # nofeat[:,21]=False 
            # refine_masks_1hot+=refine_masks_1hot[:,21:].repeat(1,22,1,1)*cur_masks_1hot_dw.bool()# refine_masks_1hot[0,:21,10,10].sum(dim=1).max() predictions[0,10,10].max()
            # logit = torch.log(refine_masks_1hot[:, :21, :, :] + 1e-8)
            # pse_loss = - torch.sum(logit * cur_masks_1hot_dw[:, :21, :, :]) / b


            return loss_guip,loss3*self.args.K_same,pse_loss*self.args.K_same
            
def main(args):
    # evaluatorA=evaluator.evaluator(domain='train_100')
    ###################################################################################
    # Arguments
    ###################################################################################
    # log_dir = create_directory(f'./experiments/logs/')
    # data_dir = create_directory(f'./experiments/data/')
    # model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/{TIMESTAMP}/')   
    pred_dir = './experiments/preditcions/{}/'.format(args.label_name)
    
    log_tag=create_directory(f'./experiments/logs/{args.tag}/')
    data_tag=create_directory(f'./experiments/data/{args.tag}/')
    model_tag=create_directory(f'./experiments/models/{args.tag}/')

    log_path = log_tag+ f'/{start_time}.txt'
    data_path = data_tag + f'/{start_time}.json'
    model_path = model_tag + f'/{start_time}.pth'
    
    
    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))

    log_func()
    setloss=SetLoss(args=args)
    setloss = torch.nn.DataParallel(setloss).cuda()
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    


    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    # train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train_aug', pred_dir, train_transform)
    cfg =(args.th_bg,args.th_step,args.th_fg)
    # pse_path='/media/ders/zhangyumin/PuzzleCAM/experiments/res/psefortrainQ/'+str(cfg)+'/'
    # pse_path='experiments/models/baseline_new_eval/2021-10-14 09:59:48npy'

    train_dataset = VOC_Dataset_For_trainQ_withcam(
        args.data_dir, 'VOC2012/VOCdevkit/VOC2012/saliency_map/',args.pse_path,args.domain,train_transform)
    valid_dataset = VOC_Dataset_For_Segmentation(args.data_dir, 'train', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    nn = 4 if(args.domain=='train') else 1
    val_iteration =4*len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    network_data = torch.load('/media/ders/zhangyumin/superpixel_fcn/result/VOCAUG/SpixelNet1l_bn_adam_3000000epochs_epochSize6000_b32_lr5e-05_posW0.003_21_09_15_21_42/model_best.tar')
    model = fcnmodel.SpixelNet1l_bn(data=network_data).cuda()
    if(args.pretrain):
        model.load_state_dict(torch.load('experiments/models/modelbest18.pth'))


    model = torch.nn.DataParallel(model).cuda()

    #=========== creat optimizer, we use adam by default ==================
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': 0},
                    {'params': model.module.weight_parameters(), 'weight_decay': 0}]
    optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(0.9, 0.999))


    # model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        the_number_of_gpu = len(use_gpu.split(','))  

    except KeyError:
        use_gpu =0
    
    # if the_number_of_gpu > 1:
    #     log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
    #     model = nn.DataParallel(model)

        # for sync bn
        # patch_replication_callback(model)



    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################

    
    # optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer() #torch.cuda.device_count() 



    eval_timer = Timer()

    train_meter = Average_Meter(['loss','sem_loss','pos_loss','relu_loss'])

    best_valid_mIoU = -1
    spixelID, XY_feat_stack = init_spixel_grid(args)
    val_spixelID,  val_XY_feat_stack = init_spixel_grid(args, b_train=False)
    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter = IOUMetric(21) 

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):  
                images = images.cuda()
                _,_,w,h= images.shape
                labels = labels.cuda()
                inuptfeats=labels.clone()
                aff_feats=labels.clone()
                aff_feats[aff_feats==255]=21
                inuptfeats[inuptfeats==255]=0  
                inuptfeats=label2one_hot_torch(inuptfeats.unsqueeze(1), C=21)
                inuptfeats=F.interpolate(inuptfeats.float(), size=(12,12),mode='bilinear', align_corners=False)
                inuptfeats=F.interpolate(inuptfeats.float(), size=(w, h),mode='bilinear', align_corners=False)
                prob = model(images)
                inuptfeats,affmat=refine_with_q(inuptfeats,prob,20,with_aff=True)

                # cur_masks_1hot=label2one_hot_torch(aff_feats.unsqueeze(1), C=22)
                # cur_masks_1hot_dw=poolfeat(cur_masks_1hot,prob)
                # b, c, h, w = cur_masks_1hot_dw.shape
                # feat_pd = F.pad(cur_masks_1hot_dw, (2, 2, 2, 2), mode='constant', value=0)
                # diff_map_list=[]

                # for i in range(5):
                #     for j in range(5):
                #         ignore_mat=(cur_masks_1hot_dw[:,21]>0.9)|(feat_pd[:,21,i:i+h,j:j+w]>0.9)

                #         # if(i==0 or i==4 or j==0 or j==4):
                #         diff_mat=(torch.max(torch.abs(feat_pd[:,:21,i:i+h,j:j+w]-feat_pd[:,:21,2:2+h,2:2+w]),dim=1)[0]>0.9)&(~ignore_mat)
                #         diff_map_list.append(diff_mat)
                #         # else:
                #         #     diff_map_list.append(torch.zeros((b,h,w)).bool().cuda())
                # diff_map=torch.stack(diff_map_list,dim=1)
                # affmat[diff_map]=0#torch.sum(diff_map)
                # for i in range(19):
                #     cur_masks_1hot_dw=rewith_affmat(cur_masks_1hot_dw,affmat)
                # inuptfeats= upfeat(cur_masks_1hot_dw,prob)


                predictions =torch.argmax(inuptfeats,dim=1)
                predictions[predictions==21]=255
            
                # for visualization
                if step == 0:
                    disp=refine_with_q(XY_feat_stack,prob,50) #
                    disp= disp - XY_feat_stack#get_numpy_from_tensor(rgb[0]-rgb[1])
                    b = abs(disp[:,0])-disp[:,0]
                    g = abs(disp[:,0])+disp[:,0]
                    r = abs(disp[:,1])
                    
                    rgb= torch.stack([r,g,b],dim=1) 
                    rgb=make_cam(rgb)
                    mask_fg=(labels>0)
                    mask_fg=mask_fg.unsqueeze(1).expand(rgb.shape)
                    rgb = 200*(1-make_cam(rgb))#rgb[0].cpu().numpy()rgb*mask_fg
                    for b in range(args.batch_size):
                        image = get_numpy_from_tensor(images[b])
                        pred_mask = get_numpy_from_tensor(predictions[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        image_gt = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)[..., ::-1]
                       
                        disp = rgb[b].transpose(0,1).transpose(1,2)
                        displacement =  cv2.addWeighted(image, 0.0, get_numpy_from_tensor(disp).astype(np.uint8), 0.8, 0)[..., ::-1]
                        #displacement
                        image_gt = image_gt.astype(np.float32) / 255.
                        displacement = displacement.astype(np.float32) / 255.

                        writer.add_image('Mask/{}'.format(args.batch_size*step+b + 1), image_gt, iteration, dataformats='HWC')
                        writer.add_image('disp/{}'.format(args.batch_size*step+b + 1), displacement, iteration, dataformats='HWC')
                
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    meter.add_batch(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        _,_,_,_,_,_,_,mIoU, _=meter.evaluate()
        return mIoU*100,_
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)
    # mIoU,re_th = evaluatorA.evaluate('experiments/models/baseline_new_eval/2021-10-14 09:59:48.pth',model)
    # print(mIoU,re_th)
    for iteration in range(max_iteration):
        # mIoU, _ = evaluate(valid_loader) 
        images, imgids,tags,sailencys,cams= train_iterator.get()
        tags = tags.cuda()
        b,c,w,h=images.shape
        #################################################################################################
        # Inference
        #################################################################################################
        prob = model(images)

        ###############################################################################
        # The part is to calculate losses.
        ###############################################################################

            # print(labels.size(), labels.min(), labels.sum())

        #
        sailencys = sailencys.cuda().view(sailencys.shape[0],1,sailencys.shape[1],sailencys.shape[2])/255.0
        labels =(sailencys>0.2).long() 
        cams = cams.cuda()
         #cv2.imwrite('1.png',labels[9][0].detach().cpu().numpy()*255)
         #cv2.imwrite('2.png',masks[9][0].detach().cpu().numpy()*255)

        label_1hot = label2one_hot_torch(labels, C=2) # set C=50 as SSN does

        # label_1hot = label2one_hot_torch(labels, C=2) # set C=50 as SSN does
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat_stack)  # B* (50+2 )* H * W


        reloss=setloss(prob,LABXY_feat_tensor,cams,imgids)
        loss_s=torch.mean(reloss[0])
        loss_p=torch.mean(reloss[1])
        relu_loss=torch.mean(reloss[2])

        # prob[:,4]=args.relu_t- relufn(args.relu_t -prob[:,4]) #prob[:,4].max()
        # relu_loss= 1/(torch.sum(prob,dim=1).mean()**2)
       

        # loss_s=torch.sum(torch.tensor(loss_s_list))
        # relu_loss=torch.sum(torch.tensor(loss_relu_list))


        loss=loss_s+ loss_p+relu_loss
        # relu_loss=torch.tensor(0)
        # loss = class_loss_fn(logits, labels)
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'sem_loss' : loss_s.item(), 
            'pos_loss' : loss_p.item(), 
            'relu_loss' : relu_loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss,sem_loss,pos_loss, relu_loss= train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'sem_loss' :sem_loss,
                'pos_loss' :pos_loss,
                'relu_loss' :relu_loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                sem_loss={sem_loss:.4f}, \
                pos_loss={pos_loss:.4f}, \
                relu_loss={relu_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % (val_iteration) == 0:
        # # for test
        # if (iteration + 1)>0:
        #     mIoU=evaluate(valid_loader) 
        # # end test
            mIoU, _ = evaluate(valid_loader)
            # mIoU,re_th = evaluatorA.evaluate('experiments/models/baseline_new_eval/2021-10-14 09:59:48.pth',model)

            # continue
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
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
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
            nni.report_intermediate_result(mIoU)

    nni.report_intermediate_result(best_valid_mIoU)
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        params=DottableDict(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise