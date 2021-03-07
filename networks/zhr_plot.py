print("Now, let's plot!")
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import csv
import os

myfolder = "/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/zhr_plot"

class plt_PointNav():
    def __init__(self,filename):
        self.data = pd.read_csv(os.path.join(myfolder,filename+".csv"))
        self.data_easy = self.data[self.data.zhr_difficulty=="easy"]
        self.data_medium = self.data[self.data.zhr_difficulty=="medium"]
        self.data_hard = self.data[self.data.zhr_difficulty=="hard"]
        self.data_easy.reset_index(drop=True, inplace=True)
        self.data_medium.reset_index(drop=True, inplace=True)
        self.data_hard.reset_index(drop=True, inplace=True)
        self.data_new_1 = pd.concat([self.data_easy, self.data_medium, self.data_hard]).dropna().sort_index() 
        self.data_new_2 = self.data_easy.append(self.data_medium, sort=False)
        self.data_new_2 = self.data_new_2.append(self.data_hard, sort=False)
        self.data_new_2.reset_index(drop=True, inplace=True)
        
        self.data_new_1.rename(columns={'zhr_difficulty':'Difficulty'},inplace=True)
        self.data_new_2.rename(columns={'zhr_difficulty':'Difficulty'},inplace=True)

        with sns.axes_style("darkgrid"):
            g = sns.relplot(x=self.data_new_1.index, y="validity",data=self.data_new_1,kind='scatter',hue='Difficulty',style="Difficulty")
            g.fig.suptitle("ScatterPlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Scatter_validity_1.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x=self.data_new_2.index, y="validity",data=self.data_new_2,kind='scatter',hue='Difficulty',style="Difficulty")
            g.fig.suptitle("ScatterPlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Scatter_validity_2.png"))
        with sns.axes_style("darkgrid"):
            g = sns.catplot(x="validity", y="Difficulty", hue="Difficulty",kind="violin", data=self.data_new_2)
            g.fig.suptitle("ViolinPlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Volin_validity.png"))
        with sns.axes_style("darkgrid"):
            g = sns.catplot(x="num_steps", y="Difficulty", hue="Difficulty",kind="violin", data=self.data_new_2)
            g.fig.suptitle("ViolinPlot of num_steps",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Volin_step.png"))
        with sns.axes_style("darkgrid"):
            g = sns.catplot(x="reward", y="Difficulty", hue="Difficulty",kind="violin", data=self.data_new_2)
            g.fig.suptitle("ViolinPlot of reward",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Volin_reward.png"))
        success_rate = sum(self.data_new_2.success)/len(self.data_new_2.success)
        print("The success rate of ",filename+".csv",f"policy is {success_rate}")
        plt.show()
plt_PointNav("PointNav_0_True")#0.68
plt_PointNav("PointNav_1_Flase")#0.736
plt_PointNav("PointNav_2_Flase")#0.738
plt_PointNav("PointNav_3_Flase")#0.728

