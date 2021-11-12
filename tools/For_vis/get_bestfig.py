from torch._C import dtype
from tools.ai.evaluate_utils import *
from operator import mod
from PIL import Image
import os
import cv2
palette_img_PIL = Image.open(r"VOC2012/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png")
palette = palette_img_PIL.getpalette()

our_path='/media/ders/zhangyumin/PuzzleCAM/experiments/res/eps_cam_101/'
eps_path='/media/ders/zhangyumin/EPS-1/result/voc12_eps_pret/result/cam_png_aug/'
mask_path='VOC2012/VOCdevkit/VOC2012/SegmentationClassAug/'


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def save_fig(index):
    mask_data=mask_path+test_list[index][:-1]+'.png' 
    ours_data=our_path+test_list[index][:-1]+'.png'
    eps_data =eps_path+test_list[index][:-1]+'.png'

    mask = Image.open(mask_data)
    ours =Image.open(ours_data)
    eps=Image.open(eps_data)
    mask=np.array(mask)
    mask=Image.fromarray(mask)
    mask.putpalette(palette)
    mask.save('experiments/demo/new'+test_list[index][:-1]+'_mask.png')
    eps=np.array(eps)
    eps=Image.fromarray(eps)
    eps.putpalette(palette)
    eps.save('experiments/demo/new'+test_list[index][:-1]+'_eps.png')
    ours=np.array(ours)
    ours=Image.fromarray(ours)
    ours.putpalette(palette)
    ours.save('experiments/demo/new'+test_list[index][:-1]+'_ours.png')

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
        ours =Image.open(ours_data)
        eps=Image.open(eps_data)
        # if n<10:
        #     mask=np.array(mask)
        #     mask=Image.fromarray(mask)
        #     mask.putpalette(palette)
        #     mask.save('experiments/demo/'+test_list[n][:-1]+'_mask.png')
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_mask.png',mask )
            
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_eps.png',np.array(eps))
        #     # cv2.imwrite('experiments/demo/'+test_list[n][:-1]+'_ours.png',np.array(ours))
        mIou_eps[n]=calculate_mIoU(eps,mask)
        mIou_our[n]=calculate_mIoU(ours,mask)
        diff[n]=mIou_our[n]-mIou_eps[n]
    print(diff.max())
    index = indices(diff, lambda x: x > 40)
    print(index) 
    print('-------------') 
    for n in range(len(index)):    
        if mIou_eps[index[n]]>55:
            fig_index=index[n]
            print(test_list[fig_index][:-1])
            save_fig(fig_index)
           




    
    