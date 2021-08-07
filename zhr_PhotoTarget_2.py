import sys
sys.path.insert(0, './yolov5')
from yolov5.zhr_detect_3 import hayo
object_detector = hayo()

import quaternion
import cv2

from zhr_geometry_utils import angle_between_quaternions
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from habitat.utils.visualizations import maps
import json
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
SAVE="s"
RESET="r"

from train_splitnet import HabitatRLTrainAndEvalRunner
class TakePhoto(HabitatRLTrainAndEvalRunner):
# from eval_splitnet import HabitatRLEvalRunner
# class TakePhoto(HabitatRLEvalRunner):
    def __init__(self, create_decoder=True):
        super(TakePhoto, self).__init__(create_decoder)
    def take_photo(self):
        from matplotlib import use as matplotlib_use
        matplotlib_use('TkAgg')
        
        obs = self.envs.reset()
        action_zhr=0
        cnt=0
        while(1):
            cnt+=1
            obs, rewards, dones, infos = self.envs.step(np.array([action_zhr]))
            zhr_rgb=np.array(obs["rgb"].squeeze()).transpose(1,2,0)

            self.detect_msg = object_detector.forward(obs=zhr_rgb)
            success,box,zhr_rgb,conf = self.detect_msg["success"],self.detect_msg["box"],self.detect_msg["zhr_rgb"],self.detect_msg["conf"]
            print(f"This is the {cnt} step. {'Succeed!' if success else 'Keep seeking...'}")
            # from yolov5.zhr_utils.plots import plot_one_box
            # if box is not None:
            #     zhr_rgb = zhr_rgb.transpose(2,0,1) 
            #     plot_one_box(box, zhr_rgb, label=f"refidgerator {conf}", color=[70, 25, 53], line_thickness=None)
            #     zhr_rgb = zhr_rgb.transpose(1,2,0)
            try:
                print(box[0].item(),box[1].item())
                print(box[2].item(),box[3].item())
            except:
                print("++++++++")
            rgb_img = Image.fromarray(zhr_rgb, mode="RGB")
            # depth_img = Image.fromarray(((obs["depth"].squeeze().numpy()+1)*50).astype(np.uint8), mode="L")
            with open("/home/u/Desktop/splitnet/zhr_global.json","r") as f:
                zhr_global=json.load(f)
                print("current_position",zhr_global["current_position"])
                # print("target_position",zhr_global["target_position"])
                print("agent_rotation",infos[0]["zhr_ego_rotation"])
                print(f"The confidence is {conf}")

                
                
                current_direction = angle_between_quaternions(infos[0]["zhr_ego_rotation"],quaternion.as_quat_array(np.array([1,0,0,0])))
                # if infos[0]["scene_id"].split("/")[-1].split(".")[0] == "Eudora":
                    # target_postion = np.array( [0.4759184718132019, 0.16554422676563263, -0.9458940029144287]) # refridgerator in Eudora
                # else:
                #     raise NotImplemented("What map is it?")
                target_postion = np.array( [0,0,0])
                current_position = zhr_global["current_position"]

                ground_truth_direction = target_postion - current_position
                # cosangle = a.dot(d)/(np.linalg.norm(a) * np.linalg.norm(d))
                print("current_direction is:(angle)",current_direction)
                print("ground_truth_direction is:(angle)",np.arctan2(ground_truth_direction[0],ground_truth_direction[2]))
                # print("distance is:",np.linalg.norm(ground_truth_direction))
                print("cosangle is:",current_direction-np.arctan2(ground_truth_direction[0],ground_truth_direction[2]))

            plt.ion()
            plt.clf()
            ax = plt.subplot(1, 2, 1)
            ax.set_title(f"Color {conf}")
            plt.imshow(rgb_img)
            # ax = plt.subplot(2, 2, 2)
            # ax.set_title("Depth")
            # plt.imshow(depth_img)
            ax = plt.subplot(1, 2, 2)
            ax.set_title(infos[0]["scene_id"].split('/')[-1])           
            tmp=infos[0]["top_down_map"]["map"]
            top_down_map = maps.colorize_topdown_map(infos[0]["top_down_map"]["map"])
            plt.imshow(top_down_map)
            plt.show()
            plt.pause(0.001)
            plt.ioff()

            b,g,r = np.array(obs["rgb"].squeeze())
            img_cv = cv2.merge([r,g,b])
            if box is not None:
                cv2.rectangle(img_cv, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), [72,36,125], thickness=5, lineType=cv2.LINE_AA)
            cv2.imshow("RGB", img_cv)
            
            keystroke = cv2.waitKey(0)
            for i in range(10):
                print(infos[0]["zhr_difficulty"])
            if keystroke == ord(FORWARD_KEY):
                action_zhr = 1
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action_zhr = 2
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action_zhr = 3
                print("action: RIGHT")
            elif keystroke == ord(SAVE):
                print("save the current observations")
                rgb_img.save(f"/home/u/Desktop/splitnet/zhr/color{cnt}.png")
                # depth_img.save(f"/home/u/Desktop/splitnet/zhr/depth{cnt}.png")
                plt.pause(1)
            elif keystroke == ord(RESET):
                obs = self.envs.reset()
                cnt = 0
            else:
                print("INVALID KEY") 

if __name__ == "__main__":
    runner = TakePhoto()
    runner.take_photo()

# "Denmark.glb", 1 big room; table->television
# "Elmira.glb", 1 big room; table->television
# "Eudora.glb", 1+1+1; a table->refridgerator
# Eastville-refridgerator
# Ribera-refridgerator

# "Cantwell.glb", multi-room
# "Denmark.glb", 1 big room; table->television
# "Eastville.glb", multi-room; one room with table, and goods
# "Edgemere.glb", 1 big 1 small; not table
# "Elmira.glb", 1 big room; table->television
# "Eudora.glb", 1+1+1; a table->refridgerator
# "Greigsville.glb", 1 big 2 small; no table
# "Mosquito.glb", multi-room
# "Pablo.glb", multi-room
# "Ribera.glb", 1 big N small; two tables
# "Sands.glb", multi-room
# "Sargents.glb", broken image
# "Scioto.glb",  multi-room
# "Scottsmoor.glb", broken image
# "Sisters.glb", very simple big room + small room
# "Swormville.glb", multi-room; one room with table, and goods

# {"episode_id": "1", "scene_id": "data/scene_datasets/gibson/Denmark.glb", "start_position": [-3.340601921081543, 0.16988539695739746, 2.016998052597046], "start_rotation": [0, 0.6000644294121716, 0, -0.7999516738867698], "info": {"geodesic_distance": 4.896366119384766, "difficulty": "easy"}, "goals": [{"position": [0.4480510354042053, 0.16988539695739746, -0.2640528380870819], "radius": null}], "shortest_paths": null, "start_room": null}, 
# {"episode_id": "1", "scene_id": "data/scene_datasets/gibson/Elmira.glb", "start_position": [1.912306547164917, 0.11030919849872589, -1.518726110458374], "start_rotation": [0, 0.5665859993106956, 0, -0.8240026124868175], "info": {"geodesic_distance": 4.720672130584717, "difficulty": "easy"}, "goals": [{"position": [-0.8647806644439697, 0.11030919849872589, -0.10190017521381378], "radius": null}], "shortest_paths": null, "start_room": null}, 



