#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Tuple

import gym
import habitat
import numpy as np
import torch
from dg_util.python_utils import misc_util
from dg_util.python_utils import pytorch_util as pt_util
from habitat import SimulatorActions

from arguments import get_args
from networks import networks
from networks import optimizers
from networks.networks import VisualPolicy, VisualPolicy_PhaseTwo
from reinforcement_learning.get_config import get_dataset_config
from reinforcement_learning.nav_rl_env import make_env_fn, PointnavRLEnv, ExplorationRLEnv, RunAwayRLEnv
from utils.env_util import VecPyTorch, HabitatVecEnvWrapper

ACTION_SPACE = [SimulatorActions.MOVE_FORWARD,
                SimulatorActions.TURN_LEFT,
                SimulatorActions.TURN_RIGHT]
HABITAT_ACTION_TO_ACTION_SPACE = {val: ii for ii, val in enumerate(ACTION_SPACE)}
ACTION_SPACE = np.array(ACTION_SPACE, dtype=np.int64)

SIM_ACTION_TO_NAME = {
    SimulatorActions.MOVE_FORWARD: "Forward",
    SimulatorActions.TURN_LEFT: "Rotate Left",
    SimulatorActions.TURN_RIGHT: "Rotate Right",
    SimulatorActions.STOP: "Stop",
}


def make_task_envs(env_types, nav_configs, nav_datasets, shell_args):
    data_keys = list(nav_datasets.keys())
    """
    Modifed the nav_datasets.
    """
    if shell_args.data_subset == "train": # data_keys[0] == "train"
        nav_datasets = [{'train': nav_datasets['train'][ii]} for ii in range(len(nav_datasets[data_keys[0]]))]
    elif shell_args.data_subset == "val":
        # data_keys == ['train', 'val']
        # nav_datasets == {'train': [<habitat.datasets.pointnav.pointnav_dataset.PointNavDatasetV1 object at 0x7f13dc3872b0>], 'val': None}
        nav_datasets = [{key: nav_datasets[key][ii] for key in data_keys} for ii in range(len(nav_datasets[data_keys[0]]))]
    else:
        print("ZHR: this is not the original train process or val process.")
        nav_datasets = [{'train': nav_datasets['train'][ii]} for ii in range(len(nav_datasets[data_keys[0]]))]
    
    env_fn_args: Tuple[Tuple] = tuple(
        zip(env_types, nav_configs, nav_datasets, range(shell_args.seed, shell_args.seed + len(nav_configs)))
    )
    # print(env_types[0]._episode_success(env_types[0]))
    # print(env_types[0],dir(env_types[0]))
    if shell_args.use_multithreading:
        envs = habitat.ThreadedVectorEnv(make_env_fn, env_fn_args)
    else:
        envs = habitat.VectorEnv(make_env_fn, env_fn_args, multiprocessing_start_method="forkserver")
    envs = HabitatVecEnvWrapper(envs)
    return envs


class BaseHabitatRLRunner(object):
    def __init__(self, create_decoder):
        self.shell_args = get_args()
        self.torch_devices = None
        self.device = None
        self.configs = None
        self.gym_action_space = None
        self.time_str = None
        self.optimizer = None
        self.agent = None
        self.agent_PhaseTwo = None
        self.start_iter = None
        self.observation_space = None
        self.env_types = None
        self.datasets = None
        self.envs = None
        self.checkpoint_dir = None
        self.compute_surface_normals = None

        self.setup(create_decoder)
        self.restore()
        self.create_envs()

    def setup_device(self):
        self.shell_args.cuda = not self.shell_args.no_cuda and torch.cuda.is_available()
        if self.shell_args.cuda:
            torch_devices = [int(gpu_id.strip()) for gpu_id in self.shell_args.pytorch_gpu_ids.split(",")]
            device = "cuda:" + str(torch_devices[0])
        else:
            torch_devices = None
            device = "cpu"
        self.torch_devices = torch_devices
        self.device = device

    def restore(self):
        if self.shell_args.load_model:
            # print("\n++++++++ Now load the network of PhaseOne ++++++++"*3)
            # self.start_iter = pt_util.restore_from_folder(
            #     self.agent, # VisualPolicy(.....)
            #     os.path.join(self.shell_args.log_prefix, self.shell_args.checkpoint_dirname,"*"),
            #     # '/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/   checkpoints/    *'
            #     self.shell_args.saved_variable_prefix, # ''
            #     self.shell_args.new_variable_prefix, # ''
            # )# zhr: start with zhr_weights
            # print("++++++++ Loaded network of PhaseOne ++++++++\n"*3)
            print("\n++++++++ Now load the network of PhaseTwo ++++++++"*3)
            # ZHR:debug-1 
            pt_util.restore_from_folder(
                self.agent_PhaseTwo,
                os.path.join("/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/checkpoints/best_weights", "*"),
                self.shell_args.saved_variable_prefix,
                self.shell_args.new_variable_prefix,
            )
            print("++++++++ Loaded network of PhaseTwo ++++++++\n"*3)
        else:
            print("Randomly initializing parameters")
            pt_util.reset_module(self.agent)

    def save_checkpoint(self, num_to_keep=-1, iteration=0):
        if self.shell_args.save_checkpoints and not self.shell_args.no_weight_update:
            pt_util.save(self.agent, self.checkpoint_dir, num_to_keep=num_to_keep, iteration=iteration)
            #ZHR:debug-1
            # pt_util.save(self.agent_PhaseTwo, self.checkpoint_dir, num_to_keep=num_to_keep, iteration=iteration)

    def setup(self, create_decoder):
        self.setup_device()
        render_gpus = [int(gpu_id.strip()) for gpu_id in self.shell_args.render_gpu_ids.split(",")]
        self.configs = []
        self.env_types = []
        for proc in range(self.shell_args.num_processes): 
            extra_task_sensors = set()

            extra_agent_sensors = set()
            if self.shell_args.record_video or self.shell_args.update_encoder_features:
                extra_agent_sensors.add("DEPTH_SENSOR")

            if "SEMANTIC_SENSOR" in extra_agent_sensors:
                extra_task_sensors.append("CLASS_SEGMENTATION_SENSOR")

            if self.shell_args.dataset == "mp3d":
                data_path = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
            elif self.shell_args.dataset == "gibson":
                # data_path = "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
                data_path = "data/datasets/pointnav/gibson/v1/{split}/zhr.json.gz"
            # Attention: both eval_.. and base_.. have data_path configuration!!!!!!!!!!
            else:
                raise NotImplementedError("No rule for this dataset.")

            config = get_dataset_config(
                data_path,
                self.shell_args.data_subset,
                self.shell_args.max_episode_length,
                render_gpus[proc % len(render_gpus)],
                list(extra_task_sensors),
                list(extra_agent_sensors),
            )# zhr: habitat_nav_task_config.yaml is contained in config
            config.TASK.NUM_EPISODES_BEFORE_JUMP = self.shell_args.num_processes

            if self.shell_args.blind and not self.shell_args.record_video:
                config.SIMULATOR.RGB_SENSOR.HEIGHT = 2
                config.SIMULATOR.RGB_SENSOR.WIDTH = 2
            if self.shell_args.task == "pointnav":
                config.TASK.SUCCESS_REWARD = 2
                config.TASK.SUCCESS_DISTANCE = 0.2
                config.TASK.COLLISION_REWARD = -0.1 #ZHR
                config.TASK.ENABLE_STOP_ACTION = False
                if self.shell_args.task == "pointnav":
                    self.env_types.append(PointnavRLEnv)# zhr: PointnavRLEnv is the key class
            elif self.shell_args.task == "exploration":
                config.TASK.GRID_SIZE = 1
                assert config.TASK.GRID_SIZE >= config.SIMULATOR.FORWARD_STEP_SIZE
                config.TASK.NEW_GRID_CELL_REWARD = 0.1
                config.TASK.COLLISION_REWARD = 0  # -0.1
                config.TASK.RETURN_VISITED_GRID = self.shell_args.record_video
                config.ENVIRONMENT.MAX_EPISODE_STEPS = 250
                config.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = False
                self.env_types.append(ExplorationRLEnv)
                config.TASK.NUM_EPISODES_BEFORE_JUMP = 5
            elif self.shell_args.task == "flee":
                config.TASK.COLLISION_REWARD = 0  # -0.1
                config.ENVIRONMENT.MAX_EPISODE_STEPS = 250
                config.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = False
                self.env_types.append(RunAwayRLEnv)
                config.TASK.NUM_EPISODES_BEFORE_JUMP = 5
            else:
                raise NotImplementedError("Unknown task type")

            if self.shell_args.record_video:
                config.TASK.NUM_EPISODES_BEFORE_JUMP = -1
                config.TASK.STEP_SIZE = config.SIMULATOR.FORWARD_STEP_SIZE
                config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = config.ENVIRONMENT.MAX_EPISODE_STEPS
                config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250

            config.TASK.OBSERVE_BEST_NEXT_ACTION = self.shell_args.algo == "supervised"

            self.configs.append(config)
        if self.shell_args.debug:
            print("Config\n", self.configs[0])

        self.shell_args.cuda = not self.shell_args.no_cuda and torch.cuda.is_available()

        if self.shell_args.blind:
            decoder_output_info = []
        else:
            decoder_output_info = [("reconstruction", 3), ("depth", 1), ("surface_normals", 3)] # zhr: auxiliary visual task

        if self.shell_args.encoder_network_type == "ShallowVisualEncoder":
            encoder_type = networks.ShallowVisualEncoder# zhr: CNN 
        elif self.shell_args.encoder_network_type == "ResNetEncoder":
            encoder_type = networks.ResNetEncoder
        else:
            raise NotImplementedError("Unknown network type.")

        self.gym_action_space = gym.spaces.discrete.Discrete(len(ACTION_SPACE))
        target_vector_size = None
        if self.shell_args.task == "pointnav":
            # target_vector_size = 2
            target_vector_size = 0 #ZHR:debug0
        elif self.shell_args.task == "exploration" or self.shell_args.task == "flee":
            target_vector_size = 0
        self.agent = VisualPolicy( # zhr: Key class
            self.gym_action_space, #<class 'gym.spaces.discrete.Discrete'> such as Discrete(3).
            base=networks.RLBaseWithVisualEncoder, # zhr: Key class
            base_kwargs=dict(
                encoder_type=encoder_type,# zhr: CNN, that is, ShallowVisualEncoder
                decoder_output_info=decoder_output_info,# zhr: auxiliary visual task [("reconstruction", 3), ("depth", 1), ("surface_normals", 3)]
                recurrent=True,
                end_to_end=self.shell_args.end_to_end,
                hidden_size=256,
                target_vector_size=target_vector_size,# zhr: 2
                action_size=len(ACTION_SPACE),
                gpu_ids=self.torch_devices,
                create_decoder=create_decoder,# zhr: True
                blind=self.shell_args.blind,
            ),# zhr: construct network
        )

        #ZHR:debug-1
        self.agent_PhaseTwo = VisualPolicy_PhaseTwo( # zhr: Key class
            self.gym_action_space, #<class 'gym.spaces.discrete.Discrete'> such as Discrete(3).
            base=networks.RLBase_PhaseTwo, # zhr: Key class
            base_kwargs=dict(
                encoder_type=encoder_type,# zhr: CNN, that is, ShallowVisualEncoder
                decoder_output_info=decoder_output_info,# zhr: auxiliary visual task [("reconstruction", 3), ("depth", 1), ("surface_normals", 3)]
                recurrent=True,
                end_to_end=self.shell_args.end_to_end,
                hidden_size=256,
                target_vector_size=target_vector_size,# zhr: 2
                action_size=len(ACTION_SPACE),
                gpu_ids=self.torch_devices,
                create_decoder=create_decoder,# zhr: True
                blind=self.shell_args.blind,
            ),# zhr: construct network
        )

        """
        # zhr: networks.RLBaseWithVisualEncoder.__init__()
        #       define the layers of (GRU, ShallowVisualEncoder, visual_projection, ego_motion, motion_model, critic)
        # zhr: networks.RLBaseWithVisualEncoder.forward()
        #       image->encoder->[rollouts,128,8,8]->projection->[rollouts,256]->cat target,motion->[rollouts,261]->rl_layer->[rollouts,256]
        #       ->critic[rollouts,1]     x[rollouts,256]     rnn_hxs[1,256]
        """

        if self.shell_args.debug:
            print("actor critic", self.agent)
        self.agent.to(self.device)
        
        #ZHR:debug-1
        self.agent_PhaseTwo.to(self.device)
        """!!!!!!!!!!!!!!!!!!
        Otherwise, it will encounter error: "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same"
        """


        self.time_str = misc_util.get_time_str()

        visual_layers = self.agent.base.visual_encoder.module
        if self.shell_args.freeze_encoder_features:
            # Not necessary, but probably lets pytorch be more space efficient.
            for param in visual_layers.encoder.parameters():
                param.requires_grad = False

        if self.shell_args.freeze_visual_decoder_features:
            if hasattr(visual_layers, "bridge"):
                for param in visual_layers.bridge.parameters():
                    param.requires_grad = False
            if hasattr(visual_layers, "decoder"):
                for param in visual_layers.decoder.parameters():
                    param.requires_grad = False
            if hasattr(visual_layers, "out"): 
                for param in visual_layers.out.parameters():
                    param.requires_grad = False
            if hasattr(visual_layers, "class_pred_layer"):
                if visual_layers.class_pred_layer is not None:
                    for param in visual_layers.class_pred_layer.parameters():
                        param.requires_grad = False

        if self.shell_args.freeze_motion_decoder_features and self.shell_args.freeze_policy_decoder_features:
            for param in self.agent.base.visual_projection.parameters():
                param.requires_grad = False
        if self.shell_args.freeze_motion_decoder_features:
            for param in self.agent.base.egomotion_layer.parameters():
                param.requires_grad = False
            for param in self.agent.base.motion_model_layer.parameters():
                param.requires_grad = False
        if self.shell_args.freeze_policy_decoder_features:
            for param in self.agent.base.gru.parameters():
                param.requires_grad = False
            for param in self.agent.base.rl_layers.parameters():
                param.requires_grad = False
            for param in self.agent.base.critic_linear.parameters():
                param.requires_grad = False
            for param in self.agent.dist.parameters():
                param.requires_grad = False

        if self.shell_args.freeze_motion_decoder_features and self.shell_args.freeze_policy_decoder_features:
            for param in self.agent_PhaseTwo.base.visual_projection.parameters():
                param.requires_grad = False
        if self.shell_args.freeze_motion_decoder_features:
            for param in self.agent_PhaseTwo.base.egomotion_layer.parameters():
                param.requires_grad = False
            for param in self.agent_PhaseTwo.base.motion_model_layer.parameters():
                param.requires_grad = False
        if self.shell_args.freeze_policy_decoder_features:
            for param in self.agent_PhaseTwo.base.gru.parameters():
                param.requires_grad = False
            for param in self.agent_PhaseTwo.base.rl_layers.parameters():
                param.requires_grad = False
            for param in self.agent_PhaseTwo.base.critic_linear.parameters():
                param.requires_grad = False
            for param in self.agent_PhaseTwo.dist.parameters():
                param.requires_grad = False

        if self.shell_args.algo == "ppo":
            self.optimizer = optimizers.VisualPPO( # zhr: key class
                self.agent,
                self.shell_args.clip_param,
                self.shell_args.ppo_epoch,
                self.shell_args.num_mini_batch,
                self.shell_args.value_loss_coef,
                self.shell_args.entropy_coef,
                lr=self.shell_args.lr,
                eps=self.shell_args.eps,
                max_grad_norm=self.shell_args.max_grad_norm,
            )
        elif self.shell_args.algo == "supervised":
            self.optimizer = optimizers.BehavioralCloningOptimizer(
                self.agent,
                self.shell_args.clip_param,
                self.shell_args.ppo_epoch,
                self.shell_args.num_mini_batch,
                self.shell_args.value_loss_coef,
                self.shell_args.entropy_coef,
                lr=self.shell_args.lr,
                eps=self.shell_args.eps,
            )
        else:
            raise NotImplementedError("No such algorithm")

        height = self.configs[0].SIMULATOR.RGB_SENSOR.HEIGHT
        width = self.configs[0].SIMULATOR.RGB_SENSOR.WIDTH
        #ZHR:debug3
        self.observation_space = {
            "zhr_new_input": ((1,), np.dtype(np.float32)), #ZHR:debug3
            "pointgoal": ((2,), np.dtype(np.float32)),
            "prev_action_one_hot": ((len(ACTION_SPACE),), np.dtype(np.float32)),
            "prev_action": ((1,), np.dtype(np.int64)),
        }
        self.compute_surface_normals = self.shell_args.record_video or self.shell_args.update_encoder_features
        if self.shell_args.algo == "supervised":
            self.observation_space["best_next_action"] = ((len(ACTION_SPACE),), np.dtype(np.float32))
        if self.shell_args.update_encoder_features:
            self.observation_space["depth"] = ((1, height, width), np.dtype(np.float32))
            if self.compute_surface_normals:
                self.observation_space["surface_normals"] = ((3, height, width), np.dtype(np.float32))
        if not self.shell_args.end_to_end:
            self.observation_space["visual_encoder_features"] = (
                (self.agent.base.num_output_channels, 256 // 2 ** 5, 256 // 2 ** 5),
                np.dtype(np.float32),
            )

        # Send dummy batch through to allocate memory before vecenv
        # print("Feeding dummy batch")
        # dummy_start = time.time()

        # # zhr_tmp1 = torch.rand(self.shell_args.num_processes, self.agent.recurrent_hidden_state_size)
        # # zhr_tmp2 = torch.rand(self.shell_args.num_processes, 1)
        # # zhr_tmp1 = zhr_tmp1.to(self.device)
        # # zhr_tmp2 = zhr_tmp2.to(self.device)
        # # # https://stackoverflow.com/questions/61149598/pytorchs-input-type-torch-floattensor-and-weight-type-torch-cuda-floattensor
        # zhr_value, zhr_action, zhr_action_log_probs, zhr_rnn_hxs=self.agent.act(
        # # self.agent.act( # zhr: this is the original
        #     {
        #         "zhr_new_input": torch.rand(self.shell_args.num_processes, 1).to(self.device), #ZHR:debug3
        #         "images": torch.rand(
        #             (
        #                 self.shell_args.num_processes,
        #                 3,
        #                 self.configs[0].SIMULATOR.RGB_SENSOR.HEIGHT,
        #                 self.configs[0].SIMULATOR.RGB_SENSOR.WIDTH,
        #             )
        #         ).to(self.device),
        #         "target_vector": torch.rand(self.shell_args.num_processes, target_vector_size).to(self.device),
        #         "prev_action_one_hot": torch.rand(self.shell_args.num_processes, self.gym_action_space.n).to(
        #             self.device
        #         ),
        #     },
        #     torch.rand(self.shell_args.num_processes, self.agent.recurrent_hidden_state_size).to(self.device),
        #     torch.rand(self.shell_args.num_processes, 1).to(self.device),
        #     # zhr_tmp1,
        #     # zhr_tmp2,
        # )
        # print("Done feeding dummy batch %.3f" % (time.time() - dummy_start))
        self.start_iter = 0
        self.checkpoint_dir = os.path.join(
            self.shell_args.log_prefix, self.shell_args.checkpoint_dirname, self.time_str
        )

    def create_envs(self):
        start_t = time.time()
        envs = make_task_envs(self.env_types, self.configs, self.datasets, self.shell_args)
        print("Envs created in %.3f" % (time.time() - start_t))
        self.envs = VecPyTorch(envs, self.device)
