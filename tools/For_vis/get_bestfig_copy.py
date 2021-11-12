from torch._C import dtype
from tools.ai.evaluate_utils import *
from operator import mod
from PIL import Image
import os
import cv2

import sys
sys.path.append(r"/media/ders/zhangyumin/PuzzleCAM/")
palette_img_PIL = Image.open(r"VOC2012/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png")
palette = palette_img_PIL.getpalette()

our_path='/media/ders/zhangyumin/PuzzleCAM/experiments/res/train_SCAMnew/QCAM32_50/'
eps_path='/media/ders/zhangyumin/PuzzleCAM/experiments/res/EPSCAM/'
mask_path='VOC2012/VOCdevkit/VOC2012/SegmentationClassAug/'


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def save_fig(index):
    mask_data=mask_path+test_list[index][:-1]+'.png' 
    ours_data=our_path+test_list[index][:-1]+'.png'
    eps_data =eps_path+test_list[index][:-1]+'.png'

    mask = Image.open(mask_data)
    ours =Image.open(ours_data).convert('L')
    eps=Image.open(eps_data).convert('L')
    mask=np.array(mask)
    mask=Image.fromarray(mask)
    mask.putpalette(palette)
    mask.save('experiments/demo/ne/'+test_list[index][:-1]+'_mask.png')
    eps=np.array(eps)
    eps=Image.fromarray(eps)
    eps.putpalette(palette)
    eps_=cv2.imread(eps_data)
    cv2.imwrite(os.path.join('experiments/demo/ne/'+test_list[index][:-1]+'_eps.png'),eps_)
    #eps.save('experiments/demo/ne/'+test_list[index][:-1]+'_eps.png')
    ours=np.array(ours)
    ours=Image.fromarray(ours)
    ours.putpalette(palette)
    ours_=cv2.imread(ours_data)
    cv2.imwrite(os.path.join('experiments/demo/ne/'+test_list[index][:-1]+'_ours.png'),ours_)
    #ours.save('experiments/demo/ne/'+test_list[index][:-1]+'_ours.png')

if __name__ == '__main__':
    
    with open('data/train.txt', 'r') as tf:
          test_list = tf.readlines()
    diff=np.zeros([len(test_list),1]) 
    mIou_eps=np.zeros([len(test_list),1]) 
    mIou_our=np.zeros([len(test_list),1]) 
    diff.astype(np.float64)
    for n in range(len(test_list)):          ###后续可以改成crf形式
        mask_data=mask_path+test_list[n][:-1]+'.png' 
        ours_data=our_path+test_list[n][:-1]+'.png'
        eps_data =eps_path+test_list[n][:-1]+'.png'
      
        mask = Image.open(mask_data)
        ours =Image.open(ours_data).convert('L')
        eps=Image.open(eps_data).convert('L')
        #mask=cv2.imread(mask_data)
        #eps=cv2.imread(eps_data)
        #ours=cv2.imread(ours_data)
        h, w= np.shape(eps)
        our=ours.resize((w,h))
        ours=our
        #ours=cv2.resize(ours, (int(h),int(w)))  
        # if n<10:
        #     mask=np.array(mask)
        #     mask=Image.fromarray(mask)
        #     mask.putpalette(palette)
        #     mask.save('experiments/demo/'+test_list[n][:-1]+'_mask.png')
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_mask.png',mask )
            
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_eps.png',np.array(eps))
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_ours.png',np.array(ours))
        mIou_eps[n]=calculate_mIoU(eps,mask)
        mIou_our[n]=calculate_mIoU(our,mask)
        diff[n]=mIou_our[n]-mIou_eps[n]
    print(diff.max())
    index = indices(diff, lambda x: x > 10)
    print(index) 
    print('-------------') 
    for n in range(len(index)):    
        if mIou_eps[index[n]]>550:
            fig_index=index[n]
            print(test_list[fig_index][:-1])
            save_fig(fig_index)
           




    
    