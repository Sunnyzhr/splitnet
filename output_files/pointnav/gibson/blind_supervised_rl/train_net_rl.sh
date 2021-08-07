cp "$(readlink -f $0)" \
    logs/HabitatPointNav/shallow/gibson/blind_supervised_rl \

#export GLOG_minloglevel=2

python reinforcement_learning/ppo_habitat_env.py \
    --algo ppo \
    --log-prefix shallow/gibson/blind_supervised_rl \
    --eval-interval 250000 \
    --resnet-size -1 \
    --use-gae \
    --lr 2.5e-4 \
    --clip-param 0.1 \
    --value-loss-coef 0.5 \
    --use-linear-lr-decay \
    --use-linear-clip-decay \
    --entropy-coef 0.01 \
    --num-processes 16 \
    --num-steps 64 \
    --num-mini-batch 1 \
    --dataset gibson \
    --data-subset train \
    --task pointnav \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0,1,2,3 \
    --freeze-visual-features \
    --end-to-end \
    --blind \
    #--freeze-planning-features \
    #--num-train-scenes 10 \
    #--end-to-end \
    #--use-egomotion-loss \
    #--clear-weights \
    #--no-tensorboard \
    #--use-multithreading \
