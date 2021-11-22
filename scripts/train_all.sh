# NEED TO SET
GPU=0,2
TAG="voc_all"
cur_time=$(date "+%Y-%m-%d_%H:%M:%S")
echo $cur_time
DATASET=voc12


# 1. train classification network with EPS
CUDA_VISIBLE_DEVICES=${GPU} python3 train.py \
    --dataset ${DATASET} \
    --SP_CAM  false \
    --tag ${TAG}"_train_cam" \
    --batch_size 32\
    --curtime ${cur_time}
# 1. train classification network with EPS
Cmodel_path='./experiments/models/'${TAG}'_train_cam/'${cur_time}'.pth'
echo $Cmodel_path

# 2. evaluate cams and save cam as npy
CUDA_VISIBLE_DEVICES=${GPU} python3 evaluator.py \
    --Cmodel_path  $Cmodel_path\
    --dataset ${DATASET} \
    --tag ${TAG}"_evaluater_cam" \
    --curtime ${cur_time}\
    --sp_cam false\
    --savenpy true

camnpy_path='./experiments/models/'${TAG}'_evaluater_cam/'${cur_time}'/camnpy/'
echo $camnpy_path

# 3. train FCN for PSAM&ASAM

CUDA_VISIBLE_DEVICES=${GPU} python3 train_Q.py \
    --dataset ${DATASET} \
    --cam_npy_path $camnpy_path\
    --tag ${TAG}"_train_Q" \
    --batch_size 32\
    --curtime ${cur_time}

Qmodel_path='./experiments/models/'${TAG}'_train_Q/'${cur_time}'.pth'

# 4. train classification network for SP-CAM 
CUDA_VISIBLE_DEVICES=${GPU} python3 train.py \
    --dataset ${DATASET} \
    --tag ${TAG}"_train_sp_cam" \
    --batch_size 32\
    --curtime ${cur_time}
    --SP_CAM  true\
    --Qmodel_path  $Qmodel_path\

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