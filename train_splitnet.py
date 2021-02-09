#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.
# import sys
# sys.path.append("/home/u/Desktop/habitat-lab/habitat")

# import habitat_sim
# path = habitat_sim.ShortestPath()
# sim_cfg = habitat_sim.SimulatorConfiguration()
# sim_cfg.scene.id = "/home/u/Desktop/17DRP5sb8fy/Eudora.glb"
# agent_cfg = habitat_sim.agent.AgentConfiguration()
# sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg])) #zhr : it takes 1 seconds !!!
 

import sys
sys.path.insert(0, './yolov5')
from yolov5.zhr_detect_3 import hayo
object_detector = hayo()
# import cv2
import os
os.system('play -nq -t alsa synth 0.05 sine 300')

   

import datetime
import os
import random
import time
from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from a2c_ppo_acktr.utils import update_linear_schedule
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils import tensorboard_logger
from habitat.datasets import make_dataset
from habitat import SimulatorActions
from habitat.utils.visualizations.utils import images_to_video

from base_habitat_rl_runner import ACTION_SPACE, SIM_ACTION_TO_NAME
from eval_splitnet import HabitatRLEvalRunner, REWARD_SCALAR
from utils import draw_outputs
from utils.storage import RolloutStorageWithMultipleObservations


class HabitatRLTrainAndEvalRunner(HabitatRLEvalRunner):
    def __init__(self, create_decoder=True):
        self.detect_msg = None # This is from yolov5
        self.zhr_collision_flag = None
        self.zhr_get_distance = None
        self.zhr_prev_action = None

        self.rollouts = None
        self.logger = None
        self.train_stats = None
        super(HabitatRLTrainAndEvalRunner, self).__init__(create_decoder)

    def setup(self, create_decoder=True):
        assert self.shell_args.update_policy_decoder_features or self.shell_args.update_encoder_features
        super(HabitatRLTrainAndEvalRunner, self).setup(create_decoder)
        self.shell_args.cuda = not self.shell_args.no_cuda and torch.cuda.is_available()

        print("Starting make dataset")
        start_t = time.time()
        config = self.configs[0]
        dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
        observation_shape_chw = (3, config.SIMULATOR.RGB_SENSOR.HEIGHT, config.SIMULATOR.RGB_SENSOR.WIDTH) # zhr: [c,h,w]==[3,256,256]
        print("made dataset")

        assert len(dataset.episodes) > 0, "empty datasets"
        if self.shell_args.num_train_scenes > 0:
            scene_ids = sorted(dataset.scene_ids)
            random.seed(0)
            random.shuffle(scene_ids)
            used_scene_ids = set(scene_ids[: self.shell_args.num_train_scenes])
            dataset.filter_episodes(lambda x: x.scene_id in used_scene_ids)

        if self.shell_args.record_video:
            random.shuffle(dataset.episodes)

        datasets = dataset.get_splits(
            self.shell_args.num_processes, remove_unused_episodes=True, collate_scene_ids=True
        )

        print("Dataset creation time %.3f" % (time.time() - start_t))

        self.rollouts = RolloutStorageWithMultipleObservations(
            self.shell_args.num_forward_rollout_steps,
            self.shell_args.num_processes,
            observation_shape_chw,
            self.gym_action_space,
            self.agent.recurrent_hidden_state_size,
            self.observation_space,
            "rgb",
        )
        self.rollouts.to(self.device)

        print("Feeding dummy batch")
        dummy_start = time.time()
        
        # (
        # zhr_loss_total_epoch,
        # zhr_value_loss_epoch,
        # zhr_action_loss_epoch,
        # zhr_dist_entropy_epoch,
        # zhr_visual_loss_value,
        # zhr_visual_losses,
        # zhr_egomotion_loss_value, 
        # zhr_feature_prediction_loss_value
        # )=
        self.optimizer.update(self.rollouts, self.shell_args) # zhr: key method !!! excute PPO algorithm
        # self.optimizer.update(self.rollouts, self.shell_args) # zhr: this is the original code
        print("Done feeding dummy batch %.3f" % (time.time() - dummy_start))

        self.logger = None
        if self.shell_args.tensorboard:
            self.logger = tensorboard_logger.Logger(
                os.path.join(self.shell_args.log_prefix, self.shell_args.tensorboard_dirname, self.time_str + "_train")
            ) # create folder of tensorboard

        self.datasets = {"train": datasets, "val": self.eval_datasets}

        self.train_stats = dict(
            num_episodes=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            num_steps=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            reward=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            spl=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            visited_states=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            success=np.zeros(self.shell_args.num_processes, dtype=np.int32),
            end_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            start_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            delta_geodesic_distance=np.zeros(self.shell_args.num_processes, dtype=np.float32),
            distance_from_start=np.zeros(self.shell_args.num_processes, dtype=np.float32),
        )

    def train_model(self):
        zhr_flag_next_zero = False # To set zero some variables at next turn
        from habitat.utils.visualizations import maps
        from matplotlib import pyplot as plt
        from matplotlib import use as matplotlib_use
        from PIL import Image
        from matplotlib import patches

        episode_rewards = deque(maxlen=10)
        current_episode_rewards = np.zeros(self.shell_args.num_processes)
        episode_lengths = deque(maxlen=10)
        current_episode_lengths = np.zeros(self.shell_args.num_processes)
        current_rewards = np.zeros(self.shell_args.num_processes)

        total_num_steps = self.start_iter
        fps_timer = [time.time(), total_num_steps]
        timers = np.zeros(3)
        egomotion_loss = 0

        video_frames = []
        num_episodes = 0
        # self.evaluate_model()

        obs = self.envs.reset()
        current_episode_rewards = np.zeros(1) #ZHR
        current_episode_lengths = np.zeros(1) #ZHR

        if self.compute_surface_normals:
            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        if self.shell_args.algo == "supervised":
            obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE])
        self.rollouts.copy_obs(obs, 0)
        distances = pt_util.to_numpy(obs["goal_geodesic_distance"])
        self.train_stats["start_geodesic_distance"][:] = distances
        previous_visual_features = None
        egomotion_pred = None
        prev_action = None
        prev_action_probs = None
        num_updates = (
            int(self.shell_args.num_env_steps) // self.shell_args.num_forward_rollout_steps
        ) // self.shell_args.num_processes

        try:
            zhr_iter_count = 0
            for iter_count in range(num_updates): # num_updates == 1e8 // 32 == 312500
                zhr_iter_count += 1
                self.zhr_collision_flag = False
                self.zhr_get_distance = 0.0
                self.zhr_prev_action = -1
                # if self.shell_args.tensorboard: # ZHR:debug1
                if False:
                    if iter_count % 500 == 0:
                        print("Logging conv summaries")
                        self.logger.network_conv_summary(self.agent, total_num_steps)
                    elif iter_count % 100 == 0:
                        print("Logging variable summaries")
                        self.logger.network_variable_summary(self.agent, total_num_steps)

                if self.shell_args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    update_linear_schedule(self.optimizer.optimizer, iter_count, num_updates, self.shell_args.lr)

                if self.shell_args.algo == "ppo" and self.shell_args.use_linear_clip_decay:
                    self.optimizer.clip_param = self.shell_args.clip_param * (1 - iter_count / float(num_updates))

                if hasattr(self.agent.base, "enable_decoder"):
                    if self.shell_args.record_video:
                        self.agent.base.enable_decoder()
                    else:
                        self.agent.base.disable_decoder()

                for step in range(self.shell_args.num_forward_rollout_steps):
                    with torch.no_grad():
                        start_t = time.time()
                        ### 
                        # tmp1 = 1.0 if self.zhr_collision_flag else 0.0
                        # tmp2 = 1.0 if self.zhr_prev_action == 0 else 0.0
                        # tmp3 = self.zhr_get_distance
                        # # tmp2 = 1.0 if 1 == self.rollouts.additional_observations_dict["prev_action_one_hot"][step][0][0] else 0.0

                        # zhr_replace = torch.from_numpy(np.array([[tmp1,tmp2,tmp3]],dtype="int")).to("cuda:0")
                        ###
                        value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                            {
                                "images": self.rollouts.obs[step],
                                "target_vector": 0.00000000 * self.rollouts.additional_observations_dict["pointgoal"][step],
                                "prev_action_one_hot": self.rollouts.additional_observations_dict["prev_action_one_hot"][step],
                                "zhr_new_input": self.rollouts.additional_observations_dict["zhr_new_input"][step], #ZHR:debug3
                            },
                            self.rollouts.recurrent_hidden_states[step],
                            self.rollouts.masks[step],
                        )
                        self.zhr_prev_action = action
                        # ### zhr: randomly choose action:
                        # tmp =  iter_count*self.shell_args.num_forward_rollout_steps + step
                        # if tmp != 0 and tmp % 20 == 0:
                        #     action = action*0 + random.randint(0,2)

                        action_cpu = pt_util.to_numpy(action.squeeze(1))
                        translated_action_space = ACTION_SPACE[action_cpu]# zhr: ACTION_SPACE == [SimulatorActions.MOVE_FORWARD,SimulatorActions.TURN_LEFT,SimulatorActions.TURN_RIGHT]
                        if not self.shell_args.end_to_end:
                            self.rollouts.additional_observations_dict["visual_encoder_features"][
                                self.rollouts.step
                            ].copy_(self.agent.base.visual_encoder_features)

                        if self.shell_args.use_motion_loss:
                            if self.shell_args.record_video:
                                if previous_visual_features is not None:
                                    egomotion_pred = self.agent.base.predict_egomotion(
                                        self.agent.base.visual_features, previous_visual_features
                                    )
                            previous_visual_features = self.agent.base.visual_features.detach()

                        timers[1] += time.time() - start_t

                        if self.shell_args.record_video:
                            # Copy so we don't mess with obs itself
                            draw_obs = OrderedDict()
                            for key, val in obs.items():
                                draw_obs[key] = pt_util.to_numpy(val).copy()
                            best_next_action = draw_obs.pop("best_next_action", None)

                            if prev_action is not None:
                                draw_obs["action_taken"] = pt_util.to_numpy(self.agent.last_dist.probs).copy()
                                draw_obs["action_taken"][:] = 0
                                draw_obs["action_taken"][np.arange(self.shell_args.num_processes), prev_action] = 1
                                draw_obs["action_taken_name"] = SIM_ACTION_TO_NAME[draw_obs['prev_action'].item()]
                                draw_obs["action_prob"] = pt_util.to_numpy(prev_action_probs).copy()
                            else:
                                draw_obs["action_taken"] = None
                                draw_obs["action_taken_name"] = SIM_ACTION_TO_NAME[SimulatorActions.STOP]
                                draw_obs["action_prob"] = None
                            prev_action = action_cpu
                            prev_action_probs = self.agent.last_dist.probs.detach()
                            if (
                                hasattr(self.agent.base, "decoder_outputs")
                                and self.agent.base.decoder_outputs is not None
                            ):
                                min_channel = 0
                                for key, num_channels in self.agent.base.decoder_output_info:
                                    outputs = self.agent.base.decoder_outputs[
                                        :, min_channel : min_channel + num_channels, ...
                                    ]
                                    draw_obs["output_" + key] = pt_util.to_numpy(outputs).copy()
                                    min_channel += num_channels
                            draw_obs["rewards"] = current_rewards.copy()
                            draw_obs["step"] = current_episode_lengths.copy()
                            draw_obs["method"] = self.shell_args.method_name
                            if best_next_action is not None:
                                draw_obs["best_next_action"] = best_next_action
                            if self.shell_args.use_motion_loss:
                                if egomotion_pred is not None:
                                    draw_obs["egomotion_pred"] = pt_util.to_numpy(
                                        F.softmax(egomotion_pred, dim=1)
                                    ).copy()
                                else:
                                    draw_obs["egomotion_pred"] = None
                            images, titles, normalize = draw_outputs.obs_to_images(draw_obs)
                            if self.shell_args.algo == "supervised":
                                im_inds = [0, 2, 3, 1, 9, 6, 7, 8, 5, 4]
                            else:
                                im_inds = [0, 2, 3, 1, 6, 7, 8, 5]
                            height, width = images[0].shape[:2]
                            subplot_image = drawing.subplot(
                                images,
                                2,
                                5,
                                titles=titles,
                                normalize=normalize,
                                order=im_inds,
                                output_width=max(width, 320),
                                output_height=max(height, 320),
                            )
                            video_frames.append(subplot_image)

                        # save dists from previous step or else on reset they will be overwritten
                        distances = pt_util.to_numpy(obs["goal_geodesic_distance"])

                        start_t = time.time()
                        obs, rewards, dones, infos = self.envs.step(translated_action_space)# zhr: when getting to the max-episode-length
                        obs["zhr_new_input"] = torch.from_numpy(np.array([1.0 if infos[0]["zhr_collision_flag"] else 0.0])) #ZHR:debug3
                        
                        self.zhr_collision_flag = infos[0]["zhr_collision_flag"]
                        self.zhr_get_distance = infos[0]["zhr_get_distance"] 
                        # ## to print out useful infomation
                        # tmp_cnt = 0
                        # for tmp in range(len(infos[0]["zhr_shortest_ego_start"])-1):
                        #     tmp_cnt += np.linalg.norm(infos[0]["zhr_shortest_ego_start"][tmp+1] - infos[0]["zhr_shortest_ego_start"][tmp])
                        # print(infos[0]["zhr_accumulate_path"],infos[0]["zhr_get_distance"],tmp_cnt)
                        # print(infos[0]["zhr_prev_position"],infos[0]["zhr_ego_position"])
                        ### to satisify underlying codes
                        if infos[0]["zhr_prev_distance"] is None: # for the first time, the api will return None:     info["zhr_prev_distance"] = self._zhr_prev_distance
                            infos[0]["zhr_prev_distance"] = 0.0
                        if infos[0]["zhr_get_distance"] is None:
                            infos[0]["zhr_get_distance"] = 0.0
                        infos[0]["spl"] = -2.333
                        if len(dones)!=1:
                            raise NotImplemented("The length of 'dones' is not 1 !!!")
                        ### zhr: yolov5
                        zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                        self.detect_msg = object_detector.forward(obs=zhr_rgb)
                        success,box,zhr_rgb,conf = self.detect_msg["success"],self.detect_msg["box"],self.detect_msg["zhr_rgb"],self.detect_msg["conf"]
                        dones = [success]
                        ### zhr: boudary condition
                        if zhr_flag_next_zero:
                            zhr_flag_next_zero = False # To set zero some variables at next turn
                            self.zhr_get_distance = 0
                            infos[0]["zhr_get_distance"] = 0
                            infos[0]["zhr_prev_distance"] = 0
                            current_episode_rewards = np.zeros(1) 
                            current_episode_lengths = np.zeros(1)
                        if infos[0]["zhr_episode_over"]: 
                            zhr_flag_next_zero = True # To set zero some variables at next turn
                            print("This step will be skipped")
                            break # break the rollouts


                        success_rate = num_episodes/(1e-5+iter_count)
                        zhr_validity_index = infos[0]['zhr_get_distance']/(1e-5+infos[0]['zhr_accumulate_path'])
                        ### zhr: design new rewards
                        penalty_time = -0.01
                        reward_success = 3 * zhr_validity_index*0.01*self.shell_args.max_episode_length if dones[0] else 0.0
                        # penalty_rotate = -0.01 if action_cpu[0] == self.zhr_prev_action and action_cpu[0] != 0 else 0.0 #when init: zhr_prev_action == 0
                        penalty_collision = -0.04 if infos[0]["zhr_collision_flag"] else 0.0
                        reward_forward = 0.015 if action_cpu[0] == 0 else 0.0
                        # reward_flee = 1.0*(infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"])
                        rewards = reward_success + penalty_time  + reward_forward + penalty_collision
                        rewards = torch.from_numpy(np.array(rewards, dtype='float'))
                        print(f"Act:{action_cpu[0]}.", f"{success_rate*100:.1f}%Done:{success}.", f"iter:{iter_count+self.start_iter}.",\
                            f"Epi_id:{infos[0]['episode_id']}. Step:{step}. Acc_R:{current_episode_rewards[0]:.5f}. Cur_R:{rewards:.5f}.",\
                            f"{zhr_validity_index*100:.1f}%Acc_path{infos[0]['zhr_accumulate_path']:.5f}",\
                            f'delta_distance:{infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"]:.5f}',\
                            f"Collision:{self.zhr_collision_flag}")


                        # rewards = torch.from_numpy(np.array(-0.01, dtype='float')) # time penalty
                        # if infos[0]["zhr_prev_distance"] is None: # for the first time, the api will return None:     info["zhr_prev_distance"] = self._zhr_prev_distance
                        #     infos[0]["zhr_prev_distance"] = 0.0
                        #     print("infos[0][\"zhr_get_distance\"] is None")
                        # delta_distance = infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"]
                        # rewards = rewards + delta_distance # flee stimuli # 4=1/0.25
                        # if infos[0]["zhr_collision_flag"]:
                        #     rewards = rewards -0.1
                        #     print("collision!")
                        # if success:
                        #     rewards = torch.from_numpy(np.array(0.05*self.shell_args.max_episode_length))
                        # print("Action:",action_cpu[0], "Succeed:",success,\
                        #     f"Step:{step}. AccReward:{current_episode_rewards}. CurReward:{rewards}.",\
                        #         f"delta_distance:{delta_distance}")
                        
                        


                        """
                        zhr:                        
                        obs['rgb']                          torch.Size([1, 3, 256, 256])
                        obs['pointgoal']                    tensor([[2.1168, 1.2231]])
                        obs['heading']                      tensor([0.4476], dtype=torch.float64)
                        obs['prev_action']                  tensor([2])
                        obs['prev_action_one_hot']          tensor([[0., 0., 1., 0., 0., 0.]])
                        obs['goal_geodesic_distance']       tensor([2.6609], dtype=torch.float64)
                        reward                              tensor([[-0.0100]])
                        dones                               False
                        infos[0]["episode_id"]              '37366'
                        infos[0]["top_down_map"]["map"]                 It has a shape of (65, 50), contents are numbers 0~7
                        infos[0]["top_down_map"]["agent_map_coord"]     (24, 26)
                        infos[0]["top_down_map"]["agent_angle"]         -1.123239118675821
                        """
                        
                        # ### zhr: Show the ego information
                        if False:
                        # if True:           
                            matplotlib_use('TkAgg')
                            tmp=infos[0]["top_down_map"]["map"]
                            top_down_map = maps.colorize_topdown_map(infos[0]["top_down_map"]["map"])   
                            zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                            rgb_img = Image.fromarray(zhr_rgb, mode="RGB")
                            plt.ion()
                            plt.clf()
                            ax = plt.subplot(2, 1, 1)
                            ax.set_title("rgb")
                            a=infos[0]['zhr_ego_position']
                            temp=f"current_position:{infos[0]['zhr_ego_position']}. Episode_id:{infos[0]['episode_id']}"
                            plt.text(-280,-70,temp,fontsize=10)
                            temp="Action_probs:[{:.3f},{:.3f},{:.3f}]. Choose action:{:d}. critic value:{:.3f}".format(
                                self.agent.last_dist.probs.cpu().numpy()[0][0],
                                self.agent.last_dist.probs.cpu().numpy()[0][1],
                                self.agent.last_dist.probs.cpu().numpy()[0][2],
                                action.cpu().numpy()[0][0],
                                value.cpu().numpy()[0][0]
                                )
                            plt.text(-280,-40,temp,fontsize=10)

                            
                            if box is not None:
                                rect=patches.Rectangle(xy=(int(box[0]),int(box[1])),width=int(box[2])-int(box[0]),height=int(box[3])-int(box[1]),linewidth=2,fill=False,edgecolor='r')
                                ax.add_patch(rect)

                            plt.imshow(rgb_img)
                            ax = plt.subplot(2, 1, 2)
                            # ax.set_title(infos[0]["scene_id"].split('/')[-1])
                            ax.set_title(infos[0]['episode_id'])
                            plt.imshow(top_down_map)
                            plt.show()
                            plt.pause(0.001)
                            plt.ioff()
                        
                        timers[0] += time.time() - start_t
                        obs["reward"] = rewards
                        if self.shell_args.algo == "supervised":
                            obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE]).to(
                                torch.float32
                            )
                        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                        rewards *= REWARD_SCALAR
                        rewards = np.clip(rewards, -10, 10)

                        if self.shell_args.record_video and not dones[0]:
                            obs["top_down_map"] = infos[0]["top_down_map"]

                        if self.compute_surface_normals:
                            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))

                        current_rewards = pt_util.to_numpy(rewards)
                        current_episode_rewards += pt_util.to_numpy(rewards).squeeze()
                        current_episode_lengths += 1 # zhr: with a maxium of 500
                        for ii, done_e in enumerate(dones):
                            if done_e:
                                num_episodes += 1
                                os.system('play -nq -t alsa synth 0.05 sine 500')
                                if self.shell_args.record_video:
                                    final_rgb = draw_obs["rgb"].transpose(0, 2, 3, 1).squeeze(0)
                                    if self.shell_args.task == "pointnav":
                                        if infos[ii]["spl"] > 0:
                                            draw_obs["action_taken_name"] = "Stop. Success"
                                            draw_obs["reward"] = [self.configs[0].TASK.SUCCESS_REWARD]
                                            final_rgb[:] = final_rgb * np.float32(0.5) + np.tile(
                                                np.array([0, 128, 0], dtype=np.uint8),
                                                (final_rgb.shape[0], final_rgb.shape[1], 1),
                                            )
                                        else:
                                            draw_obs["action_taken_name"] = "Timeout. Failed"
                                            final_rgb[:] = final_rgb * np.float32(0.5) + np.tile(
                                                np.array([128, 0, 0], dtype=np.uint8),
                                                (final_rgb.shape[0], final_rgb.shape[1], 1),
                                            )
                                    elif self.shell_args.task == "exploration" or self.shell_args.task == "flee":
                                        draw_obs["action_taken_name"] = "End of episode."
                                    final_rgb = final_rgb[np.newaxis, ...].transpose(0, 3, 1, 2)
                                    draw_obs["rgb"] = final_rgb

                                    images, titles, normalize = draw_outputs.obs_to_images(draw_obs)
                                    im_inds = [0, 2, 3, 1, 6, 7, 8, 5]
                                    height, width = images[0].shape[:2]
                                    subplot_image = drawing.subplot(
                                        images,
                                        2,
                                        5,
                                        titles=titles,
                                        normalize=normalize,
                                        order=im_inds,
                                        output_width=max(width, 320),
                                        output_height=max(height, 320),
                                    )
                                    video_frames.extend(
                                        [subplot_image]
                                        * (self.configs[0].ENVIRONMENT.MAX_EPISODE_STEPS + 30 - len(video_frames))
                                    )

                                    if "top_down_map" in infos[0]:
                                        video_dir = os.path.join(self.shell_args.log_prefix, "videos")
                                        if not os.path.exists(video_dir):
                                            os.makedirs(video_dir)
                                        im_path = os.path.join(
                                            self.shell_args.log_prefix, "videos", "total_steps_%d.png" % total_num_steps
                                        )
                                        from habitat.utils.visualizations import maps
                                        import imageio

                                        top_down_map = maps.colorize_topdown_map(infos[0]["top_down_map"]["map"])
                                        imageio.imsave(im_path, top_down_map)

                                    images_to_video(
                                        video_frames,
                                        os.path.join(self.shell_args.log_prefix, "videos"),
                                        "total_steps_%d" % total_num_steps,
                                    )
                                    video_frames = []

                                if self.shell_args.task == "pointnav":
                                    print(
                                        "FINISHED EPISODE %d Length %d Reward %.3f SPL %.4f"
                                        % (
                                            num_episodes,
                                            current_episode_lengths[ii],
                                            current_episode_rewards[ii],
                                            infos[ii]["spl"],
                                        )
                                    )
                                    self.train_stats["spl"][ii] = infos[ii]["spl"]
                                    # self.train_stats["success"][ii] = self.train_stats["spl"][ii] > 0
                                    self.train_stats["success"][ii] = success #ZHR
                                    self.train_stats["end_geodesic_distance"][ii] = (
                                        distances[ii] - self.configs[0].SIMULATOR.FORWARD_STEP_SIZE
                                    )
                                    self.train_stats["delta_geodesic_distance"][ii] = (
                                        self.train_stats["start_geodesic_distance"][ii]
                                        - self.train_stats["end_geodesic_distance"][ii]
                                    )
                                    self.train_stats["num_steps"][ii] = current_episode_lengths[ii]
                                elif self.shell_args.task == "exploration":
                                    print(
                                        "FINISHED EPISODE %d Reward %.3f States Visited %d"
                                        % (num_episodes, current_episode_rewards[ii], infos[ii]["visited_states"])
                                    )
                                    self.train_stats["visited_states"][ii] = infos[ii]["visited_states"]
                                elif self.shell_args.task == "flee":
                                    print(
                                        "FINISHED EPISODE %d Reward %.3f Distance from start %.4f"
                                        % (num_episodes, current_episode_rewards[ii], infos[ii]["distance_from_start"])
                                    )
                                    self.train_stats["distance_from_start"][ii] = infos[ii]["distance_from_start"]

                                self.train_stats["num_episodes"][ii] += 1
                                self.train_stats["reward"][ii] = current_episode_rewards[ii]

                                # if self.shell_args.tensorboard: #ZHR:debug1
                                if False:
                                    log_dict = {"single_episode/reward": self.train_stats["reward"][ii]}
                                    if self.shell_args.task == "pointnav":
                                        log_dict.update(
                                            {
                                                "single_episode/num_steps": self.train_stats["num_steps"][ii],
                                                "single_episode/spl": self.train_stats["spl"][ii],
                                                "single_episode/success": self.train_stats["success"][ii],
                                                "single_episode/start_geodesic_distance": self.train_stats[
                                                    "start_geodesic_distance"
                                                ][ii],
                                                "single_episode/end_geodesic_distance": self.train_stats[
                                                    "end_geodesic_distance"
                                                ][ii],
                                                "single_episode/delta_geodesic_distance": self.train_stats[
                                                    "delta_geodesic_distance"
                                                ][ii],
                                            }
                                        )
                                    elif self.shell_args.task == "exploration":
                                        log_dict["single_episode/visited_states"] = self.train_stats["visited_states"][
                                            ii
                                        ]
                                    elif self.shell_args.task == "flee":
                                        log_dict["single_episode/distance_from_start"] = self.train_stats[
                                            "distance_from_start"
                                        ][ii]
                                    # self.logger.dict_log(log_dict, step=(total_num_steps + self.shell_args.num_processes * step + ii)) #ZHR:debug1

                                episode_rewards.append(current_episode_rewards[ii])
                                current_episode_rewards[ii] = 0
                                episode_lengths.append(current_episode_lengths[ii])
                                current_episode_lengths[ii] = 0
                                self.train_stats["start_geodesic_distance"][ii] = obs["goal_geodesic_distance"][ii]

                        # If done then clean the history of observations.
                        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
                        bad_masks = torch.FloatTensor(
                            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
                        )

                        #ZHR:debug3
                        self.rollouts.insert(
                            obs, recurrent_hidden_states, action, action_log_prob, value, rewards, masks, bad_masks
                        )# the rewards will then be used to update weights      
                        #def insert(self, obs, ...):  self.obs[self.step + 1].copy_(obs) ... self.rewards[self.step].copy_(rewards) ...
                    
                    
                    ### zhr: boundary condition -> otherwise, the agent will not stop until the last step of rollout is successful
                    if dones[0]:
                        zhr_flag_next_zero = True 
                        # zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                        # rgb_img = Image.fromarray(zhr_rgb, mode="RGB")
                        # rgb_img.save(f"/home/u/Desktop/splitnet/zhr/training_success/{self.start_iter}_{iter_count}_{conf}.png")
                        break # wait to update weights
                
                if infos[0]["zhr_episode_over"]: #to restart the rollout step to 0
                    print("This step is now skipped")
                    zhr_iter_count -= 1
                    continue # continue iter, and not update this step
                                
                """
                self.rollouts.obs.shape                     torch.Size([9, 1, 3, 256, 256])
                self.rollouts.rewards.shape                 torch.Size([8, 1, 1])
                self.rollouts.value_preds.shape             torch.Size([9, 1, 1])
                self.rollouts.recurrent_hidden_states.shape torch.Size([9, 1, 256])
                """
                with torch.no_grad():
                    start_t = time.time()
                    next_value = self.agent.get_value(
                        {
                            "images": self.rollouts.obs[-1],
                            "target_vector": self.rollouts.additional_observations_dict["pointgoal"][-1],
                            "prev_action_one_hot": self.rollouts.additional_observations_dict["prev_action_one_hot"][
                                -1
                            ],
                            "zhr_new_input": self.rollouts.additional_observations_dict["zhr_new_input"][-1], #ZHR:debug3
                        },
                        self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1],
                    ).detach()
                    timers[1] += time.time() - start_t

                self.rollouts.compute_returns(
                    next_value, self.shell_args.use_gae, self.shell_args.gamma, self.shell_args.tau
                )# tensor([[-0.17840]], device='cuda:0'), True, 0.99, 0.95

                if not self.shell_args.no_weight_update:
                    start_t = time.time()
                    if self.shell_args.algo == "supervised":
                        (
                            total_loss,
                            action_loss,
                            visual_loss_total,
                            visual_loss_dict,
                            egomotion_loss,
                            forward_model_loss,
                        ) = self.optimizer.update(self.rollouts, self.shell_args)
                    else:
                        (
                            total_loss,
                            value_loss,
                            action_loss,
                            dist_entropy,
                            visual_loss_total,
                            visual_loss_dict,
                            egomotion_loss,
                            forward_model_loss,
                        ) = self.optimizer.update(self.rollouts, self.shell_args) # zhr: key method!!! execute PPO algorithm
                        print("Now update weights, executing 'self.optimizer.update(self.rollouts, self.shell_args)'")
                    timers[2] += time.time() - start_t
                zhr_log_dict = {}
                zhr_log_dict.update({
                    "zhr/rewards": rewards,
                    "zhr/step": step,
                    "zhr/validity": zhr_validity_index,
                    "zhr/dist_entropy": dist_entropy,
                    "zhr/success_validity": zhr_validity_index if dones[0] else 0.0,
                    })
                self.logger.dict_log(zhr_log_dict, step=zhr_iter_count+self.start_iter)#ZHR:debug2


                self.rollouts.after_update() # zhr:?? 129 dimensions. Replace var[0] with var[-1]

                # save for every interval-th episode or for the last epoch
                if zhr_iter_count % self.shell_args.save_interval == 0 or iter_count == num_updates - 1:
                    if zhr_iter_count != 0: # zhr: do not save for the first episode
                        # self.save_checkpoint(5, total_num_steps) #ZHR:original
                        self.save_checkpoint(5, zhr_iter_count) #ZHR:debug2

                total_num_steps += self.shell_args.num_processes * self.shell_args.num_forward_rollout_steps

                if not self.shell_args.no_weight_update and iter_count % self.shell_args.log_interval == 0:
                    # "--log-interval" default 10 
                    log_dict = {}
                    if len(episode_rewards) > 1:
                        end = time.time()
                        nsteps = total_num_steps - fps_timer[1]
                        fps = int((total_num_steps - fps_timer[1]) / (end - fps_timer[0]))
                        timers /= nsteps
                        env_spf = timers[0]
                        forward_spf = timers[1]
                        backward_spf = timers[2]
                        print(
                            (
                                "{} Updates {}, num timesteps {}, FPS {}, Env FPS "
                                "{}, \n Last {} training episodes: mean/median reward "
                                "{:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n"
                            ).format(
                                datetime.datetime.now(),
                                iter_count,
                                total_num_steps,
                                fps,
                                int(1.0 / env_spf),
                                len(episode_rewards),
                                np.mean(episode_rewards),
                                np.median(episode_rewards),
                                np.min(episode_rewards),
                                np.max(episode_rewards),
                            )
                        )

                        # if self.shell_args.tensorboard:#ZHR:debug1
                        if False:
                            log_dict.update(
                                {
                                    "stats/full_spf": 1.0 / (fps + 1e-10),
                                    "stats/env_spf": env_spf,
                                    "stats/forward_spf": forward_spf,
                                    "stats/backward_spf": backward_spf,
                                    "stats/full_fps": fps,
                                    "stats/env_fps": 1.0 / (env_spf + 1e-10),
                                    "stats/forward_fps": 1.0 / (forward_spf + 1e-10),
                                    "stats/ ": 1.0 / (backward_spf + 1e-10),
                                    "episode/mean_rewards": np.mean(episode_rewards),
                                    "episode/median_rewards": np.median(episode_rewards),
                                    "episode/min_rewards": np.min(episode_rewards),
                                    "episode/max_rewards": np.max(episode_rewards),
                                    "episode/mean_lengths": np.mean(episode_lengths),
                                    "episode/median_lengths": np.median(episode_lengths),
                                    "episode/min_lengths": np.min(episode_lengths),
                                    "episode/max_lengths": np.max(episode_lengths),
                                }
                            )
                        fps_timer[0] = time.time()
                        fps_timer[1] = total_num_steps
                        timers[:] = 0
                    # if self.shell_args.tensorboard:#ZHR:debug1
                    if False:
                        log_dict.update(
                            {
                                "loss/action": action_loss,
                                "loss/0_total": total_loss,
                                "loss/visual/0_total": visual_loss_total,
                                "loss/exploration/egomotion": egomotion_loss,
                                "loss/exploration/forward_model": forward_model_loss,
                            }
                        )
                        if self.shell_args.algo != "supervised":
                            log_dict.update({"loss/entropy": dist_entropy, "loss/value": value_loss})
                        for key, val in visual_loss_dict.items():
                            log_dict["loss/visual/" + key] = val
                        # self.logger.dict_log(log_dict, step=total_num_steps)#ZHR:debug1

                # save checkpoint when eval
                if self.shell_args.eval_interval is not None and total_num_steps % self.shell_args.eval_interval < (
                    self.shell_args.num_processes * self.shell_args.num_forward_rollout_steps 
                ):
                    # self.shell_args.eval_interval == 2500
                    # total_num_steps == 32
                    # self.shell_args.eval_interval == 2500
                    # self.shell_args.num_processes == 1
                    # self.shell_args.num_forward_rollout_steps == 8
                    self.save_checkpoint(-1, total_num_steps)
                    self.set_log_iter(total_num_steps)
                    
                    self.evaluate_model() # zhr:
                        # zhr:??Traceback (most recent call last):
                        #   File "/home/u/Desktop/splitnet/train_splitnet.py", line 618, in train_model
                        #     self.evaluate_model()
                        #   File "/home/u/Desktop/splitnet/eval_splitnet.py", line 97, in evaluate_model
                        #     if not os.path.exists(self.eval_dir):
                        #   File "/home/u/anaconda3/envs/habitat-yolo/lib/python3.6/genericpath.py", line 19, in exists
                        #     os.stat(path)
                        # TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
                        # Saved /home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/checkpoints/2021_01_21_15_31_49/000002528.pt
                    # reset the env datasets
                    self.envs.unwrapped.call(
                        ["switch_dataset"] * self.shell_args.num_processes, [("train",)] * self.shell_args.num_processes
                    )
                    obs = self.envs.reset()
                    current_episode_rewards = np.zeros(1) #ZHR
                    current_episode_lengths = np.zeros(1) #ZHR
                    
                    if self.compute_surface_normals:
                        obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
                    obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                    if self.shell_args.algo == "supervised":
                        obs["best_next_action"] = pt_util.from_numpy(obs["best_next_action"][:, ACTION_SPACE])
                    self.rollouts.copy_obs(obs, 0)
                    distances = pt_util.to_numpy(obs["goal_geodesic_distance"])
                    self.train_stats["start_geodesic_distance"][:] = distances
                    previous_visual_features = None
                    egomotion_pred = None
                    prev_action = None
                    prev_action_probs = None
                if dones[0]: #ZHR boundary condition
                    self.envs.reset()
                    current_episode_rewards = np.zeros(1)
                    current_episode_lengths = np.zeros(1)
        except:
            # Catch all exceptions so a final save can be performed
            import traceback

            traceback.print_exc()
        finally:
            self.save_checkpoint(-1, total_num_steps)


def main():
    runner = HabitatRLTrainAndEvalRunner()
    runner.train_model()


if __name__ == "__main__":
    main()
