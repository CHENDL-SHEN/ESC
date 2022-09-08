import numpy as np
import cv2
import os
from PIL import Image
def palette_convert(imgs_directory: str):
    path = os.walk(imgs_directory)
    palette_img_PIL = Image.open(r"/media/ders/zhangyumin/irn-master/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png")
    palette = palette_img_PIL.getpalette()
    for  files in path:
        for file in files[2]:
            # print(file)
            if(file.endswith(".png") or file.endswith(".jpg") ):

                    img_path=os.path.join(imgs_directory,file)
                    # img=cv2.imread(img_path)
                    # cv2.normalize(img,img)
                    # cv2.imwrite(img_path,img*255)
                    # try:
                    img_pil2=Image.open(img_path)
                    img_pil2.putpalette(palette)
                    # np.asarray(img_pil2).max()
                    img_path=os.path.join('VOC2012/VOCdevkit/VOC2012/hed2',file)
                    
                    img_pil2.save(img_path)
                    # except:
                    #     print(img_path)

if(__name__=="__main__"):
    # for x in [0,1,10,20,30]:
        # palette_convert(R"experiments/predictions/train_kmeans66@val@scale=0.5,1.0,1.5,2.0@iteration=0"+str(x)+'/')
        palette_convert(R"VOC2012/VOCdevkit/VOC2012/hed/")
        
        pass    