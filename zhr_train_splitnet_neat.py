import sys
sys.path.insert(0, './yolov5')
from yolov5.zhr_detect_3 import hayo
object_detector = hayo()
import os
os.system('play -nq -t alsa synth 0.05 sine 300')
import json
import csv

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
        self.zhr_flag_prev_collision = None
        self.zhr_get_distance = None
        self.zhr_prev_action = None
        self.zhr_csvfolder = "/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/zhr_plot"
        with open(os.path.join(self.zhr_csvfolder,"ObejectSearch.csv"),"a+") as zhr_csvfile:
            zhr_writer = csv.writer(zhr_csvfile)
            zhr_writer.writerow(["epi_iter","zhr_difficulty","episode_id","num_steps","reward","success","validity","success_validity","entropy"])
        self.zhr_epi_step = None

        self.rollouts = None
        self.logger = None
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
        datasets = dataset.get_splits(self.shell_args.num_processes, remove_unused_episodes=True, collate_scene_ids=True)
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
        self.optimizer.update(self.rollouts, self.shell_args) # zhr: key method !!! excute PPO algorithm
        print("Done feeding dummy batch %.3f" % (time.time() - dummy_start))
        self.datasets = {"train": datasets, "val": self.eval_datasets}
        # create folder of tensorboard
        self.logger = tensorboard_logger.Logger( os.path.join(self.shell_args.log_prefix, self.shell_args.tensorboard_dirname, self.time_str + "_train")) 
        

    def train_model(self):
        zhr_flag_next_zero = False # To set zero some variables at next turn
        from habitat.utils.visualizations import maps
        from matplotlib import pyplot as plt
        from matplotlib import use as matplotlib_use
        from PIL import Image
        from matplotlib import patches

        episode_rewards = deque(maxlen=10)
        current_episode_rewards = np.zeros(1)
        episode_lengths = deque(maxlen=10)
        current_episode_lengths = np.zeros(1)
        current_rewards = np.zeros(1)
        num_episodes = 0

        obs = self.envs.reset()
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        self.rollouts.copy_obs(obs, 0)
        distances = pt_util.to_numpy(obs["goal_geodesic_distance"])

        num_updates = (int(self.shell_args.num_env_steps) // self.shell_args.num_forward_rollout_steps) // self.shell_args.num_processes
        try:
            zhr_iter_count = 0
            zhr_episode_count = 0
            self.zhr_flag_prev_collision = False
            self.save_checkpoint(-1, zhr_iter_count) #save the initial weights
            zhr_initial_lr = self.shell_args.lr
            self.zhr_epi_step = 0

            for X_iter in range(num_updates):
                zhr_iter_count += 1
                self.zhr_get_distance = 0.0
                self.zhr_prev_action = -1

                # decrease learning rate linearly
                if self.shell_args.use_linear_lr_decay:
                    self.shell_args.lr = zhr_initial_lr * (1.0 - zhr_iter_count/num_updates) 
                    # update_linear_schedule(self.optimizer.optimizer, zhr_iter_count, num_updates, self.shell_args.lr)
                if self.shell_args.algo == "ppo" and self.shell_args.use_linear_clip_decay:
                    self.optimizer.clip_param = self.shell_args.clip_param * (1 - zhr_iter_count / float(num_updates))
                if hasattr(self.agent.base, "enable_decoder"):
                    if self.shell_args.record_video:
                        self.agent.base.enable_decoder()
                    else:
                        self.agent.base.disable_decoder()

                for step in range(self.shell_args.num_forward_rollout_steps):
                    with torch.no_grad():
                        self.zhr_epi_step += 1

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

                        action_cpu = pt_util.to_numpy(action.squeeze(1))
                        translated_action_space = ACTION_SPACE[action_cpu]# zhr: ACTION_SPACE == [SimulatorActions.MOVE_FORWARD,SimulatorActions.TURN_LEFT,SimulatorActions.TURN_RIGHT]
                        if not self.shell_args.end_to_end:
                            self.rollouts.additional_observations_dict["visual_encoder_features"][
                                self.rollouts.step
                            ].copy_(self.agent.base.visual_encoder_features)

                        # save dists from previous step or else on reset they will be overwritten
                        distances = pt_util.to_numpy(obs["goal_geodesic_distance"])

                        obs, _, dones, infos = self.envs.step(translated_action_space)# zhr: when getting to the max-episode-length
                        obs["zhr_new_input"] = torch.from_numpy(np.array([1.0 if infos[0]["zhr_collision_flag"] else 0.0])) #ZHR:debug3
                        
                        self.zhr_get_distance = infos[0]["zhr_get_distance"] 
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
                            self.zhr_flag_prev_collision = False
                            zhr_flag_next_zero = False # To set zero some variables at next turn
                            self.zhr_get_distance = 0
                            infos[0]["zhr_get_distance"] = 0
                            infos[0]["zhr_prev_distance"] = 0
                            current_episode_rewards = np.zeros(1) 
                            current_episode_lengths = np.zeros(1)
                            self.zhr_epi_step = 0
                        if infos[0]["zhr_episode_over"]: 
                            with open(os.path.join(self.zhr_csvfolder,"ObejectSearch.csv"),"a+") as zhr_csvfile:
                                zhr_writer = csv.writer(zhr_csvfile)
                                zhr_writer.writerow([
                                    zhr_episode_count,
                                    infos[0]["zhr_difficulty"],
                                    infos[0]['episode_id'],
                                    self.zhr_epi_step,
                                    current_episode_rewards[0],
                                    1 if dones[0] else 0,
                                    zhr_validity_index,
                                    zhr_validity_index if dones[0] else 0.0,
                                    # 0.0,
                                    dist_entropy/self.shell_args.entropy_coef,
                                ])
                            zhr_flag_next_zero = True # To set zero some variables at next turn
                            print("This step will be skipped")
                            break # break the rollouts

                        success_rate = num_episodes/(1e-5+zhr_episode_count)
                        zhr_validity_index = infos[0]['zhr_get_distance']/(1e-5+infos[0]['zhr_accumulate_path'])
                        
                        ### zhr: design new rewards
                        penalty_time = -0.01
                        penalty_collision = -0.01 if infos[0]["zhr_collision_flag"] else 0.0

                        reward_success = 0.01*zhr_validity_index*self.shell_args.max_episode_length if dones[0] else 0.0
                        reward_flee = 0.01/0.25*(infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"])
                        reward_flee_clip = 0.01/0.25*10*np.clip(infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"], -0.025, 0.025)
                        reward_forward = 0.03 if action_cpu[0] == 0 else 0.0

                        if False: #a1 penalty:time(small); reward:success(huge)
                            rewards = penalty_time + 10*reward_success
                        if False:#a2 penalty:time(small); reward:success(small)
                            rewards = penalty_time + 1*reward_success

                        if False:#b1 penalty:time(small); reward:success(small),flee(small)
                            rewards = penalty_time + 1*reward_success + 1*reward_flee
                        if False:#b2 penalty:time(small); reward:success(huge),flee(huge)
                            rewards = penalty_time + 10*reward_success + 10*reward_flee
                        if False:#b3 penalty:time(small); reward:success(huge),flee_clip(moderate)
                            rewards = penalty_time + 10*reward_success + 5*reward_flee_clip

                        if True:#c1 penalty:time(small),collision(huge); reward:success(huge),flee_clip(huge)
                            rewards = penalty_time + 10*penalty_collision + 10*reward_success + 10*reward_flee_clip
                        
                        if False: # only consider collision
                            if self.zhr_flag_prev_collision:
                                rewards = 0.1 if action_cpu[0] == 2 else -0.1
                            else:
                                rewards = 0.1 if action_cpu[0] == 0 else -0.1

                        rewards = torch.from_numpy(np.array(rewards, dtype='float'))
                        print(f"Act:{action_cpu[0]}.", f"{success_rate*100:.1f}%Done:{success}.", f"rolls:{zhr_iter_count+self.start_iter}.",\
                            f"Epi_id:{infos[0]['episode_id']}. Epi_Step:{self.zhr_epi_step}. Epi_R:{current_episode_rewards[0]:.5f}. Cur_R:{rewards:.5f}.",\
                            f"{zhr_validity_index*100:.1f}%Acc_path{infos[0]['zhr_accumulate_path']:.5f}",\
                            f'delta_distance:{infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"]:.5f}',\
                            f"PrevColli:{self.zhr_flag_prev_collision}")

                        self.zhr_flag_prev_collision = infos[0]["zhr_collision_flag"]

                        # ### zhr: Show the ego information
                        with open("/home/u/Desktop/splitnet/zhr_flag_show.json","r") as f:
                            zhr_flag_show=json.load(f)
                        # if False:
                        # if True:           
                        if zhr_flag_show["show"][0] != 0:
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
                                episode_rewards.append(current_episode_rewards[ii])
                                current_episode_rewards[ii] = 0
                                episode_lengths.append(current_episode_lengths[ii])
                                current_episode_lengths[ii] = 0

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
                        zhr_episode_count += 1
                        break # wait to update weights            
                if infos[0]["zhr_episode_over"]: #to restart the rollout step to 0
                    print("This step is now skipped")
                    zhr_episode_count += 1
                    zhr_iter_count -= 1
                    continue # continue iter, and not update this step
                with torch.no_grad():
                    next_value = self.agent.get_value(
                        {"images": self.rollouts.obs[-1],
                            "target_vector": self.rollouts.additional_observations_dict["pointgoal"][-1],
                            "prev_action_one_hot": self.rollouts.additional_observations_dict["prev_action_one_hot"][-1],
                            "zhr_new_input": self.rollouts.additional_observations_dict["zhr_new_input"][-1], #ZHR:debug3
                        },
                        self.rollouts.recurrent_hidden_states[-1],
                        self.rollouts.masks[-1],
                    ).detach()
                self.rollouts.compute_returns(next_value, self.shell_args.use_gae, self.shell_args.gamma, self.shell_args.tau)# tensor([[-0.17840]], device='cuda:0'), True, 0.99, 0.95
                zhr_log_dict = {}
                if not self.shell_args.no_weight_update:
                    start_t = time.time()
                    (total_loss,
                        value_loss,
                        action_loss,
                        dist_entropy,
                        visual_loss_total,
                        visual_loss_dict,
                        egomotion_loss,
                        forward_model_loss,
                    ) = self.optimizer.update(self.rollouts, self.shell_args) # zhr: key method!!! execute PPO algorithm
                    # zhr_log_dict.update({
                    #     "zhr/dist_entropy": dist_entropy,
                    # })
                    print("Now update weights of the PhaseOne")
                zhr_log_dict.update({
                    "zhr/rewards": rewards,
                    "zhr/step": self.zhr_epi_step,
                    "zhr/validity": zhr_validity_index,
                    # 0.0,
                    "zhr/dist_entropy": dist_entropy,
                    "zhr/success_validity": zhr_validity_index if dones[0] else 0.0,
                    })
                self.logger.dict_log(zhr_log_dict, step=zhr_iter_count+self.start_iter)#ZHR:debug2
                self.rollouts.after_update() # zhr:?? 129 dimensions. Replace var[0] with var[-1]

                # save for every interval-th episode or for the last epoch
                if zhr_iter_count % self.shell_args.save_interval == 0 or zhr_iter_count == num_updates - 1:
                    # if zhr_iter_count != 0: # zhr: do not save for the first episode
                        # self.save_checkpoint(-1, zhr_iter_count) #ZHR:debug2
                    self.save_checkpoint(-1, zhr_iter_count) #ZHR:debug2
                if dones[0]: #ZHR boundary condition
                    with open(os.path.join(self.zhr_csvfolder,"ObejectSearch.csv"),"a+") as zhr_csvfile:
                        zhr_writer = csv.writer(zhr_csvfile)
                        # (["zhr_difficulty","episode_id","num_steps","reward","success","validity","entropy"])
                        zhr_writer.writerow([
                            zhr_episode_count-1, # must -1, due to the logic of counting
                            infos[0]["zhr_difficulty"],
                            infos[0]['episode_id'],
                            self.zhr_epi_step,
                            current_episode_rewards[0],
                            1 if dones[0] else 0,
                            zhr_validity_index,
                            zhr_validity_index if dones[0] else 0.0,
                            # 0.0,
                            dist_entropy/self.shell_args.entropy_coef,
                        ])
                    self.envs.reset()
                    current_episode_rewards = np.zeros(1)
                    current_episode_lengths = np.zeros(1)
                    self.zhr_epi_step = 0
                    
        except:
            # Catch all exceptions so a final save can be performed
            import traceback
            traceback.print_exc()
        finally:
            self.save_checkpoint(-1, zhr_iter_count)

def main():
    runner = HabitatRLTrainAndEvalRunner()
    runner.train_model()


if __name__ == "__main__":
    main()
