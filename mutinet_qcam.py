from  evaluatorqcam_copy2 import *
import psemakeForQ

# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
from cv2 import LMEDS, log
import numpy as np

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from imageio import imsave
from core.spnetwork_new import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime
import sys 
# import models  


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
model = SANET_Model_new_base('resnest50', num_classes=20 + 1,process=32,fgORall=True)
model = model.cuda()
model.train()
model.load_state_dict(torch.load('experiments/models/Scam_batch16_upfeat_uss/2021-11-18 17:49:37.pth'))

# network_data = torch.load('/media/ders/zhangyumin/superpixel_fcn/result/VOCAUG/SpixelNet1l_bn_adam_3000000epochs_epochSize6000_b32_lr5e-05_posW0.003_21_09_15_21_42/model_best.tar')
# print("=> using pre-trained model '{}'".format(network_data['arch']))
# Q_model = models.__dict__[network_data['arch']]( data = network_data).cuda()
# Q_model.load_state_dict(torch.load('/media/ders/zhangyumin/PuzzleCAM/experiments/models/train_Q_relu.pth'))
# Q_model = nn.DataParallel(Q_model)
# Q_model.eval()
evaluatorA = evaluator(domain='train',withQ=True, fast_eval=False,savepng=False,save_np=False,save_path='/media/ders/zhangyumin/PuzzleCAM/finalmodel/uss/pseudo/')
# evaluatorA = psemakeForQ.evaluator(domain='train_aug',savepng=True,savept=True,th_bg=[0.05,0.1,0.03],th_step=[0.4,0.5,0.6],th_fg=[0.05,0.1,0.2])

ret = evaluatorA.evaluate(model,'experiments/models/train_Q_withPse_uss/2021-11-1723:05:33.pth')
print(ret[0])
print(ret[1])