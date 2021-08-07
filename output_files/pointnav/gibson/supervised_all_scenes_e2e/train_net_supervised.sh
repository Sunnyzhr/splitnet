cp "$(readlink -f $0)" \
    logs/HabitatPointNav/shallow/gibson/supervised_all_scenes_e2e/ \

#export GLOG_minloglevel=2

python reinforcement_learning/ppo_habitat_env.py \
    --algo supervised \
    --log-prefix shallow/gibson/supervised_all_scenes_e2e \
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
    --num-processes 16 \
    --num-steps 128 \
    --resnet-size -1 \
    --task pointnav \
    --pytorch-gpu-ids 0,1,2,3 \
    --render-gpu-ids 0,1,2,3 \
    --end-to-end \
    --freeze-visual-features \
    #--num-train-scenes 1 \
    #--use-egomotion-loss \
    #--blind \
    #--freeze-planning-features \
    #--use-egomotion-loss \
    #--clear-weights \
    #--use-curiosity-reward \
    #--no-weight-update \
    #--no-tensorboard \
    #--use-multithreading \
