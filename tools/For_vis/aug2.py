import pathlib

import sys
sys.path.append(r"/media/ders/zhangyumin/PuzzleCAM/")
image_id_list = [image_id.strip() for image_id in open(
    './data/coco/%s.txt' % "train").readlines()]

for id in image_id_list:
    path = pathlib.Path('/media/ders/zhangyumin/PuzzleCAM/finalmodel/COCO/pseudo/0.2/30/'+id +'.png')
    if not (path.exists()) : # True/False  判断路径是否存在
        print(id)