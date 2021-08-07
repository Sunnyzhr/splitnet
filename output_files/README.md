# SplitNet Weights Files
This includes weights for all of the experiments mentioned in the paper.
Note that each folder contains a Pytorch weight file and a bash script that was used for training, but the bash parameters may have changed names.
0. `imagenet_pretrain` Weights from pretraining our architecture on ImageNet.
0. `pointnav`
    0. `gibson` Weights for SplitNet and baselines trained on Gibson scenes. Also includes the baselines trained on only a subset of the scenes.
    0. `mp3d`  Weights for SplitNet and baselines trained on MP3D scenes. Also includes the baselines trained on only a subset of the scenes.
    0. `sim2sim` Weights for SplitNet and baselines trained on <mp3d> and tested on <mp3d, gibson>.
