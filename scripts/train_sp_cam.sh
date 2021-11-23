GPU=0,2
TAG="voc_sp_cam"
cur_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo $cur_time
# Default setting
DATASET=voc12

Qmodel_path='./experiments/models/Q_model_trained.pth'

# 4. train classification network for SP-CAM 
CUDA_VISIBLE_DEVICES=${GPU} python3 train.py \
    --dataset ${DATASET} \
    --tag ${TAG}"_train_sp_cam" \
    --batch_size 32\
    --curtime ${cur_time}\
    --SP_CAM  true\
    --Qmodel_path  $Qmodel_path

Cmodel_path='./experiments/models/'${TAG}'_train_sp_cam/'${cur_time}'.pth'
echo $Cmodel_path
# 5. evaluate sp_cams and make pseudo labels
CUDA_VISIBLE_DEVICES=${GPU} python3 evaluator.py \
    --Cmodel_path  $Cmodel_path\
    --dataset ${DATASET} \
    --tag ${TAG}"_evaluater_sp_cam" \
    --curtime ${cur_time}\
    --sp_cam true\
    --savepng true

camnpy_path='./experiments/models/'${TAG}'_evaluater_sp_cam/'${cur_time}'/pseudo/'
echo  "the pseudo labels path is: "$camnpy_path