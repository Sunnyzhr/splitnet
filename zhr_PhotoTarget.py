import cv2
import sys
from eval_splitnet import HabitatRLEvalRunner
REWARD_SCALAR = 1.0

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


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

class TakePhoto(HabitatRLEvalRunner):
    def __init__(self, create_decoder=True):
        super(TakePhoto, self).__init__(create_decoder)
    def take_photo(self):
        self.envs.unwrapped.call(
            ["switch_dataset"] * self.shell_args.num_processes, [("val",)] * self.shell_args.num_processes
        )

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
        eval_output_file = open(os.path.join(self.eval_dir, eval_net_file_name + ".csv"), "w")
        print("Writing results to", eval_output_file.name)

        # Save the evaled net for posterity
        if self.shell_args.save_checkpoints:
            save_model = self.agent
            pt_util.save(
                save_model,
                os.path.join(self.shell_args.log_prefix, self.shell_args.checkpoint_dirname, "eval_weights"),
                num_to_keep=-1,
                iteration=self.log_iter,
            )
            print("Wrote model to file for safe keeping")

        obs = self.envs.reset()
        if self.compute_surface_normals:
            obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))
        obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
        recurrent_hidden_states = torch.zeros(
            self.shell_args.num_processes,
            self.agent.recurrent_hidden_state_size,
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.ones(self.shell_args.num_processes, 1, dtype=torch.float32, device=self.device)

        episode_rewards = deque(maxlen=10)
        current_episode_rewards = np.zeros(self.shell_args.num_processes)
        episode_lengths = deque(maxlen=10)
        current_episode_lengths = np.zeros(self.shell_args.num_processes)

        total_num_steps = self.log_iter
        fps_timer = [time.time(), total_num_steps]
        timers = np.zeros(3)

        num_episodes = 0

        print("Config\n", self.configs[0])

        # Initialize every time eval is run rather than just at the start
        dataset_sizes = np.array([len(dataset.episodes) for dataset in self.eval_datasets])

        eval_stats = dict(
            episode_ids=[None for _ in range(self.shell_args.num_processes)],
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
        eval_stats_means = dict(
            num_episodes=0,
            num_steps=0,
            reward=0,
            spl=0,
            visited_states=0,
            success=0,
            end_geodesic_distance=0,
            start_geodesic_distance=0,
            delta_geodesic_distance=0,
            distance_from_start=0,
        )
        eval_output_file.write("name,%s,iter,%d\n\n" % (eval_net_file_name, self.log_iter))
        if self.shell_args.task == "pointnav":
            eval_output_file.write(
                (
                    "episode_id,num_steps,reward,spl,success,start_geodesic_distance,"
                    "end_geodesic_distance,delta_geodesic_distance\n"
                )
            )
        elif self.shell_args.task == "exploration":
            eval_output_file.write("episode_id,reward,visited_states\n")
        elif self.shell_args.task == "flee":
            eval_output_file.write("episode_id,reward,distance_from_start\n")
        distances = pt_util.to_numpy(obs["goal_geodesic_distance"])
        eval_stats["start_geodesic_distance"][:] = distances
        progress_bar = tqdm.tqdm(total=self.num_eval_episodes_total)
        all_done = False
        iter_count = 0
        video_frames = []
        previous_visual_features = None
        egomotion_pred = None
        prev_action = None
        prev_action_probs = None
        if hasattr(self.agent.base, "enable_decoder"):
            if self.shell_args.record_video:
                self.agent.base.enable_decoder()
            else:
                self.agent.base.disable_decoder()
        action_zhr=-1
        while not all_done:
            with torch.no_grad():
                start_t = time.time()
                value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                    {
                        "images": obs["rgb"].to(self.device),
                        "target_vector": obs["pointgoal"].to(self.device),
                        "prev_action_one_hot": obs["prev_action_one_hot"].to(self.device),
                    },
                    recurrent_hidden_states,
                    masks,
                )
                action_cpu = pt_util.to_numpy(action.squeeze(1))

                if action_zhr > -1:
                    action_cpu = np.array([action_zhr])

                translated_action_space = ACTION_SPACE[action_cpu]

                timers[1] += time.time() - start_t

                # save dists from previous step or else on reset they will be overwritten
                distances = pt_util.to_numpy(obs["goal_geodesic_distance"])

                start_t = time.time()
                obs, rewards, dones, infos = self.envs.step(translated_action_space)
                zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                cv2.imshow("RGB", zhr_rgb)
                from matplotlib import pyplot as plt
                from PIL import Image
                zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)
                rgb_img = Image.fromarray(zhr_rgb, mode="RGB")
                depth_img = Image.fromarray(((obs["depth"].squeeze().numpy()+1)*50).astype(np.uint8), mode="L")
                plt.ion()
                plt.clf()
                ax = plt.subplot(1, 2, 1)
                ax.set_title("rgb")
                plt.imshow(rgb_img)
                ax = plt.subplot(1, 2, 2)
                ax.set_title("depth")
                plt.imshow(depth_img)
                plt.show()
                plt.pause(0.001)
                plt.ioff()
                keystroke = cv2.waitKey(0)
                if keystroke == ord(FORWARD_KEY):
                    action_zhr = 0
                    print("action: FORWARD")
                elif keystroke == ord(LEFT_KEY):
                    action_zhr = 1
                    print("action: LEFT")
                elif keystroke == ord(RIGHT_KEY):
                    action_zhr = 2
                    print("action: RIGHT")
                else:
                    print("INVALID KEY") 



                timers[0] += time.time() - start_t
                obs["prev_action_one_hot"] = obs["prev_action_one_hot"][:, ACTION_SPACE].to(torch.float32)
                rewards *= REWARD_SCALAR
                rewards = np.clip(rewards, -10, 10)

                if self.shell_args.record_video and not dones[0]:
                    obs["top_down_map"] = infos[0]["top_down_map"]

                if self.compute_surface_normals:
                    obs["surface_normals"] = pt_util.depth_to_surface_normals(obs["depth"].to(self.device))

                current_episode_rewards += pt_util.to_numpy(rewards).squeeze()
                current_episode_lengths += 1
                to_pause = []

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones]).to(self.device)

                # Reverse in order to maintain order in case of multiple.
                to_pause.reverse()
                for ii in to_pause:
                    # Pause the environments that are done from the vectorenv.
                    print("Pausing env", ii)
                    self.envs.unwrapped.pause_at(ii)
                    current_episode_rewards = np.concatenate(
                        (current_episode_rewards[:ii], current_episode_rewards[ii + 1 :])
                    )
                    current_episode_lengths = np.concatenate(
                        (current_episode_lengths[:ii], current_episode_lengths[ii + 1 :])
                    )
                    for key in eval_stats:
                        eval_stats[key] = np.concatenate((eval_stats[key][:ii], eval_stats[key][ii + 1 :]))
                    dataset_sizes = np.concatenate((dataset_sizes[:ii], dataset_sizes[ii + 1 :]))

                    for key in obs:
                        if type(obs[key]) == torch.Tensor:
                            obs[key] = torch.cat((obs[key][:ii], obs[key][ii + 1 :]), dim=0)
                        else:
                            obs[key] = np.concatenate((obs[key][:ii], obs[key][ii + 1 :]), axis=0)

                    recurrent_hidden_states = torch.cat(
                        (recurrent_hidden_states[:ii], recurrent_hidden_states[ii + 1 :]), dim=0
                    )
                    masks = torch.cat((masks[:ii], masks[ii + 1 :]), dim=0)

                if len(dataset_sizes) == 0:
                    progress_bar.close()
                    all_done = True

            total_num_steps += self.shell_args.num_processes

        eval_stats_means = {key: val / eval_stats_means["num_episodes"] for key, val in eval_stats_means.items()}
        if self.shell_args.tensorboard:
            log_dict = {"single_episode/reward": eval_stats_means["reward"]}
            if self.shell_args.task == "pointnav":
                log_dict.update(
                    {
                        "single_episode/num_steps": eval_stats_means["num_steps"],
                        "single_episode/spl": eval_stats_means["spl"],
                        "single_episode/success": eval_stats_means["success"],
                        "single_episode/start_geodesic_distance": eval_stats_means["start_geodesic_distance"],
                        "single_episode/end_geodesic_distance": eval_stats_means["end_geodesic_distance"],
                        "single_episode/delta_geodesic_distance": eval_stats_means["delta_geodesic_distance"],
                    }
                )
            elif self.shell_args.task == "exploration":
                log_dict["single_episode/visited_states"] = eval_stats_means["visited_states"]
            elif self.shell_args.task == "flee":
                log_dict["single_episode/distance_from_start"] = eval_stats_means["distance_from_start"]
            self.eval_logger.dict_log(log_dict, step=self.log_iter)
        
        self.envs.unwrapped.resume_all()


def example():
    runner = TakePhoto()
    runner.take_photo()



    
       


if __name__ == "__main__":
    example()