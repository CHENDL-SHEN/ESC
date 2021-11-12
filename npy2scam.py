import cv2 as cv
import numpy as np
import os




path = '/media/ders/zhangyumin/PuzzleCAM/experiments/res/qcam_npy/'
path_list = os.listdir(path)
save_path="/media/ders/zhangyumin/PuzzleCAM/experiments/res/train_SCAM/"


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

    _,_,H,W=np.shape(npy)
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
            npy_=np.squeeze(npy[:,i,:,:])
            npy_fg=(npy_>0.2).astype(np.int_)
            white_small=(npy_<0.2).astype(np.int_)
            white_big=( npy_> 0.15).astype(np.int_)
            #print(white_big)
            white=white_big * white_small
            result_cam[:,:,j]=cv.add(npy_fg*bgr_img[:,:,j],white*255)
            cam[:,:,j]=cv.add(cam[:,:,j],result_cam[:,:,j])
    #cv.imshow('result_cam',cam)
    #cv.waitKey(0)
    cv.imwrite(os.path.join(save_path, img_id + '.png'), cam)
    