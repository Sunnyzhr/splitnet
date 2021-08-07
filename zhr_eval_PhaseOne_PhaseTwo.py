#!/usr/bin/env python3
import sys
sys.path.insert(0, './yolov5')
from yolov5.zhr_detect_3 import hayo
object_detector = hayo()
import os
os.system('play -nq -t alsa synth 0.05 sine 300')
import json
import csv

import datetime
import glob
import os
import random
import time
from collections import OrderedDict
from collections import deque
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from dg_util.python_utils import drawing
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils import tensorboard_logger
from habitat.datasets import make_dataset
from habitat import SimulatorActions
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from base_habitat_rl_runner import ACTION_SPACE, SIM_ACTION_TO_NAME
from base_habitat_rl_runner import BaseHabitatRLRunner
from reinforcement_learning.get_config import get_dataset_config
from utils import draw_outputs


def get_eval_dataset(shell_args, data_subset="val"):
    data_path = "data/datasets/pointnav/gibson/v1/{split}/zhr_{split}.json.gz"
    config = get_dataset_config(data_path, data_subset, shell_args.max_episode_length, 0, [], [])
    data_subset = shell_args.data_subset # zhr: Otherwise, the data_subset is the default "val"
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    assert len(dataset.episodes) > 0, "empty datasets"
    return dataset


class HabitatRLEvalRunner(BaseHabitatRLRunner):
    def __init__(self, create_decoder=True):
        self.eval_datasets = None
        self.eval_logger = None
        self.num_eval_episodes_total = None
        self.eval_dir = None
        self.log_iter = None
        self.zhr_len_datasets = None
        self.zhr_flag_near = None
        self.zhr_flag_hier_success = None
        self.zhr_flag_break = None

        super(HabitatRLEvalRunner, self).__init__(create_decoder)

    def setup(self, create_decoder):
        super(HabitatRLEvalRunner, self).setup(create_decoder) #create_decoder == True
        eval_dataset = get_eval_dataset(self.shell_args)
        self.num_eval_episodes_total = len(eval_dataset.episodes)
        self.eval_datasets = eval_dataset.get_splits(self.shell_args.num_processes, allow_uneven_splits=True)
        self.eval_logger = None
        self.datasets = {"val": self.eval_datasets}
        self.zhr_len_datasets = np.array([len(dataset.episodes) for dataset in self.eval_datasets])
        if self.shell_args.tensorboard and self.shell_args.eval_interval is not None:
            self.eval_logger = tensorboard_logger.Logger(os.path.join("/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/hier_tensorboard", self.time_str + "_zhr"))
        self.zhr_log_dict = {}
        self.zhrs_reward = []
        self.zhrs_step = []
        self.zhrs_validity = []
        self.zhrs_validity_success = []
        self.zhrs_difficulty = []
        self.zhrs_id = []
        
        
        
        self.eval_dir = os.path.join("/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/hier_results", self.shell_args.data_subset)
        """
        # log_prefix == '/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl'
        # results_dirname == 'results',     data_subset == 'val',    tensorboard_dirname == 'tensorboard'
        """
        self.set_log_iter(self.start_iter)

    def set_log_iter(self, iteration):
        self.log_iter = iteration
        if self.eval_logger is not None:
            self.eval_logger.count = iteration #zhr: This is the start number of iteration

    def evaluate_model(self):
        with open("/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/zhr_plot/my_eval.csv","a+") as zhr_csvfile:
            zhr_writer = csv.writer(zhr_csvfile)
            zhr_writer.writerow(["epi_iter","zhr_difficulty","episode_id","num_steps","reward","success","validity","success_validity"]) 

        self.envs.unwrapped.call(["switch_dataset"] * self.shell_args.num_processes, [("val",)] * self.shell_args.num_processes)      
        # zhr: in order to be compatible
        self.eval_dir = os.path.join(self.shell_args.log_prefix, self.shell_args.results_dirname, self.shell_args.data_subset)
        # '/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/   results/   val'
        self.set_log_iter(self.start_iter)
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        try:
            eval_net_file_name = sorted(
                glob.glob(os.path.join(self.shell_args.log_prefix, self.shell_args.checkpoint_dirname, "*") + "/*.pt"),
                key=os.path.getmtime,
            )[-1]
            eval_net_file_name = (
                self.shell_args.log_prefix.replace(os.sep, "_")
                + "_"
                + "_".join(eval_net_file_name.split(os.sep)[-2:])[:-3]
            )
        except IndexError:
            print("Warning, no weights found")
            eval_net_file_name = "random_weights"

        obs = self.envs.reset()
        if self.compute_surface_normals:
            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        recurrent_hidden_states = torch.zeros(self.shell_args.num_processes,self.agent.recurrent_hidden_state_size,dtype=torch.float32,device=self.device,)
        #ZHR:debug-1
        recurrent_hidden_states_PhaseTwo = torch.zeros(self.shell_args.num_processes,self.agent_PhaseTwo.recurrent_hidden_state_size,dtype=torch.float32,device=self.device,)
        masks = torch.ones(self.shell_args.num_processes, 1, dtype=torch.float32, device=self.device)
        current_episode_rewards = np.zeros(self.shell_args.num_processes)
        current_episode_lengths = np.zeros(self.shell_args.num_processes)
        # Initialize every time eval is run rather than just at the start
        dataset_sizes = np.array([len(dataset.episodes) for dataset in self.eval_datasets])

        self.agent.base.disable_decoder()
        self.agent_PhaseTwo.base.disable_decoder()

        length_ckpt = len(glob.glob('/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/hier_weights/*/*'))
        #zhr: it is /*/*
        progress_bar = tqdm.tqdm(total=length_ckpt)
        for cnt_ckpt in range(length_ckpt):
            # ### the coding unexpectedly stops
            if cnt_ckpt == 300:
                continue
            ### zhr3.0-hier
            myfocus = [0,50,100,150,200,250,300]
            if cnt_ckpt not in myfocus:
                continue


            phase = "Phase_1"       
            #zhr: it is /*
            self.start_iter = pt_util.restore_from_folder(
                self.agent,
                '/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/hier_weights/*',
                zhr_ckpt_num = cnt_ckpt
            )
            self.zhrs_reward = []
            self.zhrs_step = []
            self.zhrs_validity = []
            self.zhrs_validity_success = []
            self.zhrs_difficulty = []
            self.zhrs_id = []

            for cnt_episode in range(self.zhr_len_datasets[0]):
                zhr_step = 0
                
                ### zhr3.0-hierval
                current_episode_rewards = np.zeros(self.shell_args.num_processes)
                current_episode_lengths = np.zeros(self.shell_args.num_processes)
                
                while zhr_step < self.shell_args.max_episode_length:
                    with torch.no_grad():
                        zhr_step += 1
                        if phase == "Phase_1":
                            value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                                {
                                    "images": obs["rgb"].to(self.device),
                                    "prev_action_one_hot": obs["prev_action_one_hot"].to(self.device),
                                    "zhr_new_input": torch.from_numpy(np.array([[1.0 if obs["zhr_collision_flag"] else 0.0]],dtype="float32")).to(self.device), #ZHR:debug3
                                },
                                recurrent_hidden_states,
                                masks,
                            )
                            print(f"PhaseOne_{zhr_step}. Action{action.cpu().numpy()[0][0]}")

                        elif phase == "Phase_2":
                            value, action, action_log_prob, recurrent_hidden_states_PhaseTwo = self.agent_PhaseTwo.act(
                                {
                                    "images": obs["rgb"].to(self.device),
                                    "prev_action_one_hot": obs["prev_action_one_hot"].to(self.device),
                                    "target_vector": obs["pointgoal"].to(self.device),
                                },
                                recurrent_hidden_states_PhaseTwo,
                                masks,
                            )
                            print(f"PhaseTwo_{zhr_step}. Action{action.cpu().numpy()[0][0]}")
                        else:
                            raise NotImplemented("It's neither in Phase_1 nor in Phase_2!!!")
                        action_cpu = pt_util.to_numpy(action.squeeze(1))
                        translated_action_space = ACTION_SPACE[action_cpu]
                        if self.zhr_flag_near and phase == "Phase_2":
                            self.zhr_flag_break = True                            
                            print("In Phase_2, the agent successfully navigate itself to the target.\n"*3)
                            self.zhr_flag_hier_success = True
                            # self.zhrs_reward.append(reward)

                            ### zhr3.0-hierval
                            self.zhrs_reward.append(current_episode_rewards)
                            
                            self.zhrs_validity.append(zhr_validity_index)
                            self.zhrs_validity_success.append(zhr_validity_index if self.zhr_flag_hier_success else 0.0)
                            self.zhrs_step.append(zhr_step)
                            zhr_step = 0
                            phase = "Phase_1"
                            translated_action_space = translated_action_space*0
                        if zhr_step == self.shell_args.max_episode_length:
                            self.zhr_flag_break = True
                            # self.zhrs_reward.append(reward)

                            ### zhr3.0-hierval
                            self.zhrs_reward.append(current_episode_rewards)

                            self.zhrs_validity.append(zhr_validity_index)
                            self.zhrs_validity_success.append(zhr_validity_index if self.zhr_flag_hier_success else 0.0)
                            self.zhrs_step.append(zhr_step)
                            zhr_step = 0
                            translated_action_space = translated_action_space*0

                        obs, _, _, infos = self.envs.step(translated_action_space)
                        if len(self.zhrs_difficulty) == cnt_episode:
                            self.zhrs_difficulty.append(infos[0]["zhr_difficulty"])
                            self.zhrs_id.append(infos[0]['episode_id'])
                        zhr_validity_index = infos[0]['zhr_get_distance']/(1e-5+infos[0]['zhr_accumulate_path'])
                        if infos[0]["zhr_prev_distance"] is None:
                            infos[0]["zhr_prev_distance"] = 0.0
                        print(f"Epi_id:{infos[0]['episode_id']}. Step:{zhr_step}", 
                            f"{zhr_validity_index*100:.1f}%Acc_path{infos[0]['zhr_accumulate_path']:.5f}",
                            f'delta_distance:{infos[0]["zhr_get_distance"] - infos[0]["zhr_prev_distance"]:.5f}')
                        obs["zhr_new_input"] = torch.from_numpy(np.array([1.0 if infos[0]["zhr_collision_flag"] else 0.0]))
                        # if infos[0]["zhr_episode_over"]:
                        #     if zhr_step > 0: # to avoid the incorrect value of infos[0]["zhr_episode_over"]
                        #         self.zhrs_step.append(zhr_step)
                        #         zhr_step = 0
                        #         phase = "Phase_1"
                        #         break
                        if self.zhr_flag_break:
                            self.zhr_flag_break = False
                            break
                        self.zhr_flag_near = infos[0]["zhr_flag_near_target"] 
                        ### zhr: yolov5
                        zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                        self.detect_msg = object_detector.forward(obs=zhr_rgb)
                        success,box,zhr_rgb,conf = self.detect_msg["success"],self.detect_msg["box"],self.detect_msg["zhr_rgb"],self.detect_msg["conf"]
                        
                        zhr_validity_index = infos[0]['zhr_get_distance']/(1e-5+infos[0]['zhr_accumulate_path'])
                        reward_success = 3 * zhr_validity_index*0.01*self.shell_args.max_episode_length if success else 0.0
                        penalty_time = -0.01
                        reward_forward = 0.03 if action_cpu[0] == 0 else 0.0
                        penalty_collision = -0.04 if infos[0]["zhr_collision_flag"] else 0.0
                        if phase == "Phase_1":
                            reward = reward_success + penalty_time  + reward_forward + penalty_collision
                        elif phase == "Phase_2":
                            reward = penalty_time
                        else:
                            raise NotImplementedError("No such a phase!")
                        reward = torch.from_numpy(np.array(reward, dtype='float'))

                        if success:
                            phase = "Phase_2"
                            os.system('play -nq -t alsa synth 0.05 sine 400')

                        with open("/home/u/Desktop/splitnet/zhr_flag_show.json","r") as f:
                            zhr_flag_show=json.load(f)
                        # if False:
                        # if True:           
                        if zhr_flag_show["show"][0] != 0:
                            tmp=infos[0]["top_down_map"]["map"]
                            top_down_map = maps.colorize_topdown_map(infos[0]["top_down_map"]["map"])
                            from matplotlib import use as matplotlib_use
                            matplotlib_use('TkAgg')
                            from matplotlib import pyplot as plt
                            from matplotlib import patches
                            from PIL import Image
                            zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                            rgb_img = Image.fromarray(zhr_rgb, mode="RGB")
                            depth_img = Image.fromarray(((obs["depth"].squeeze().numpy()+1)*50).astype(np.uint8), mode="L")
                            plt.ion()
                            plt.clf()
                            ax = plt.subplot(1, 2, 1)
                            if box is not None:
                                rect=patches.Rectangle(xy=(int(box[0]),int(box[1])),width=int(box[2])-int(box[0]),height=int(box[3])-int(box[1]),linewidth=2,fill=False,edgecolor='r')
                                ax.add_patch(rect)
                            ax.set_title("rgb")
                            # temp=f"current_position:{infos[0]['zhr_ego_position']}"
                            # plt.text(-140,-70,temp,fontsize=10)
                            # temp=f"target_position:{infos[0]['zhr_target_position']}"
                            # plt.text(-140,-100,temp,fontsize=10)
                            # temp=f"delta_x:{infos[0]['zhr_target_position'][0]-infos[0]['zhr_ego_position'][0]}, delta_z:{infos[0]['zhr_target_position'][2]-infos[0]['zhr_ego_position'][2]}"
                            # plt.text(-140,-40,temp,fontsize=10)
                            plt.imshow(rgb_img)
                            # ax = plt.subplot(2, 2, 2)
                            # ax.set_title("depth")
                            # plt.imshow(depth_img)
                            ax = plt.subplot(1, 2, 2)
                            ax.set_title("maps")
                            plt.imshow(top_down_map)
                            plt.show()
                            plt.pause(0.001)
                            plt.ioff()
                        

                        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                        # reward *= 1.0
                        reward = np.clip(reward, -10, 10)
                        if self.compute_surface_normals:
                            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
                        current_episode_rewards += pt_util.to_numpy(reward).squeeze()
                        current_episode_lengths += 1
                        
                        # If done then clean the history of observations.
                        # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones]).to(self.device)
                        masks = torch.FloatTensor([[1.0]]).to(self.device)
                    

                # self.zhrs_reward.append(reward)
                # self.zhrs_validity.append(zhr_validity_index)
                # self.zhrs_validity_success.append(zhr_validity_index if self.zhr_flag_hier_success else 0.0)
                # if len(self.zhrs_step) < len(self.zhrs_reward):
                #     self.zhrs_step.append(zhr_step)

            progress_bar.update(1)
            self.zhr_log_dict.update(
                {
                    "zhr_mean/reward":np.mean(self.zhrs_reward),
                    "zhr_mean/step":np.mean(self.zhrs_step),
                    "zhr_mean/validity":np.mean(self.zhrs_validity),
                    "zhr_mean/validity_success":np.mean(self.zhrs_validity_success),

                    "zhr_med/reward":np.median(self.zhrs_reward),
                    "zhr_med/step":np.median(self.zhrs_step),
                    "zhr_med/validity":np.median(self.zhrs_validity),
                    "zhr_med/validity_success":np.median(self.zhrs_validity_success),

                    "zhr_min/reward":np.min(self.zhrs_reward),
                    "zhr_min/step":np.min(self.zhrs_step),
                    "zhr_min/validity":np.min(self.zhrs_validity),
                    "zhr_min/validity_success":np.min(self.zhrs_validity_success),

                    "zhr_max/reward":np.max(self.zhrs_reward),
                    "zhr_max/step":np.max(self.zhrs_step),
                    "zhr_max/validity":np.max(self.zhrs_validity),
                    "zhr_max/validity_success":np.max(self.zhrs_validity_success),
                }
            )
            self.eval_logger.dict_log(self.zhr_log_dict, step=self.start_iter)   
            with open("/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/zhr_plot/my_eval.csv","a+") as zhr_csvfile:
                zhr_writer = csv.writer(zhr_csvfile)
                for i in range(len(self.zhrs_reward)):
                    # zhr_writer.writerow([
                    #     cnt_ckpt,
                    #     self.zhrs_difficulty[i],
                    #     self.zhrs_id[i],
                    #     self.zhrs_step[i],
                    #     self.zhrs_reward[i].cpu().numpy(),
                    #     1.0 if self.zhrs_validity_success[i]>0.0 else 0.0,
                    #     self.zhrs_validity[i],
                    #     self.zhrs_validity_success[i],
                    #     ])        
                     
                    # ### zhr3.0-hierval    
                    zhr_writer.writerow([
                        cnt_ckpt,
                        self.zhrs_difficulty[i],
                        self.zhrs_id[i],
                        self.zhrs_step[i],
                        # self.zhrs_reward[i].cpu().numpy(),

                        ### zhr3.0-hierval
                        self.zhrs_reward[i][0],

                        1.0 if self.zhrs_step[i]<self.shell_args.max_episode_length else 0.0,
                        self.zhrs_validity[i],
                        self.zhrs_validity[i] if self.zhrs_step[i]<self.shell_args.max_episode_length else 0.0
                        ])                                    
        
        progress_bar.close()
        print("Finished testing")                               


def main():
    runner = HabitatRLEvalRunner()
    runner.evaluate_model()
if __name__ == "__main__":
    main()
