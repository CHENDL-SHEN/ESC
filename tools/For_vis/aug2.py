import pathlib

import sys
sys.path.append(r"/media/ders/zhangyumin/PuzzleCAM/")
image_id_list = [image_id.strip() for image_id in open(
    './data/%s.txt' % "train_aug2").readlines()]

for id in image_id_list:
    path = pathlib.Path('/media/ders/zhangyumin/PuzzleCAM/VOC2012/VOCdevkit/VOC2012/saliency_unsupervised_model/'+id +'.png')
    if (path.exists()) : # True/False  判断路径是否存在
        print(id)