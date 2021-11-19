import cv2 as cv
import numpy as np
import os




path = "/media/ders/zhangyumin/PuzzleCAM/finalmodel/base_sal/eps-resnest50/CAMs/resnest50/CAM5120"
path_list = os.listdir(path)
save_path="/media/ders/zhangyumin/PuzzleCAM/experiments/result/png/EPS_CAM/CAM5120_BIG"


path_list.sort(key=lambda x:int(x.split('.')[0]))
#print(path_list)

cls_mat=np.array([[0,0,0],
         [128,0,0],
         [0,128,0],
         [128,128,0],
         [0,0,128],
         [128,0,128],
         [0,128,128],
         [128,128,128],
         [64,0,0],
         [192,0,0],
         [64,128,0],
         [192,128,0],
         [64,0,128],
         [192,0,128],
         [64,128,128],
         [192,128,128],
         [0,64,0],
         [128,64,0],
         [0,192,0],
         [128,192,0],
         [0,64,128]])

for filename in path_list:
    npy = np.load(os.path.join(path,filename))

    _,H,W=np.shape(npy)
    img_id=filename.split('.')[0]
    print(img_id)

    cam = np.zeros((H, W), dtype=np.uint8)
    cam = cv.cvtColor(cam, cv.COLOR_GRAY2BGR)

    bgr=np.ones((H,W),dtype=np.uint8)
    bgr_img = cv.cvtColor(bgr,cv.COLOR_GRAY2BGR)

    cls_img=np.ones((H,W),dtype=np.uint8)

    #print(np.shape(np.squeeze(npy[:,1,:,:])))

    for i in range(21):
        color = cls_mat[i, :]
        #print(i)
        bgr_img[:, :, 0] = color[2]#r
        bgr_img[:, :, 1] = color[1]#b
        bgr_img[:, :, 2] = color[0]#g
        result_cam=np.ones((H,W),dtype=np.uint8)
        result_cam=cv.cvtColor(result_cam,cv.COLOR_GRAY2BGR)

        for j in range(3):
            #npy_=np.squeeze(npy[:,i,:,:])
            #bgr_img_=bgr_img[:,:,j]
            #result_cam[:,:,j]=cv.multiply(npy_,bgr_img_,dst=None, scale=None, dtype=None)
            result_cam[:,:,j]=(npy[i,:,:]*bgr_img[:,:,j])
            cam[:,:,j]=cv.add(cam[:,:,j],result_cam[:,:,j])
    #cv.imshow('result_cam',cam)
    #cv.waitKey(0)
    a=cv.imwrite(os.path.join(save_path, img_id + '.png'), cam)
    if a: 
        print('done')