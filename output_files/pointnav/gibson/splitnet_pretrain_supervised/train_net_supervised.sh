cp "$(readlink -f $0)" \
    logs/HabitatPointNav/shallow/gibson/pretrain_supervised \

#export GLOG_minloglevel=2

python reinforcement_learning/ppo_habitat_env.py \
    --algo supervised \
    --log-prefix shallow/gibson/pretrain_supervised \
    --eval-interval 250000 \
    --use-gae \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --dataset gibson \
    --data-subset train \
    --entropy-coef 0.1 \
    --num-processes 18 \
    --num-steps 256 \
    --resnet-size -1 \
    --task pointnav \
    --pytorch-gpu-ids 0,1,2,3 \
    --render-gpu-ids 1,2,3 \
    --freeze-visual-features \
    #--use-egomotion-loss \
    #--end-to-end \
    #--blind \
    #--freeze-planning-features \
    #--num-train-scenes 10 \
    #--use-egomotion-loss \
    #--clear-weights \
    #--use-curiosity-reward \
    #--no-weight-update \
    #--no-tensorboard \
    #--use-multithreading \
