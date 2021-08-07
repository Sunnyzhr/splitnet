DATASET="gibson"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Copy this file to log location for tracking the flags used.
LOG_LOCATION="logs/pointnav/"${DATASET}"/imagenet_pretrain_supervised"
mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

python train_splitnet.py \
    --algo supervised \
    --encoder-network-type ShallowVisualEncoder \
    --log-prefix ${LOG_LOCATION} \
    --eval-interval 250000 \
    --lr 2.5e-4 \
    --value-loss-coef 0.5 \
    --dataset ${DATASET} \
    --data-subset train \
    --num-processes 12 \
    --num-forward-rollout-steps 128 \
    --task pointnav \
    --pytorch-gpu-ids 0,1,2,3 \
    --render-gpu-ids 1,2,3 \
    --end-to-end \
    --no-motion-loss \
    --no-visual-loss \
    --freeze-visual-decoder-features \
    --freeze-motion-decoder-features \
    --freeze-encoder-features \
    #--new-variable-prefix base.visual_encoder.module \
    #--saved-variable-prefix module.visual_encoder \
