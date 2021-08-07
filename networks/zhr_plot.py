print("Now, let's plot!")
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import savgol_filter

myfolder = "/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/zhr_plot"

class plt_PointNav():
    def __init__(self,filename):
        self.data = pd.read_csv(os.path.join(myfolder,filename+".csv"))
        self.data["validity"][self.data["validity"]>1]=1.0

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

        # with sns.axes_style("darkgrid"):
        #     g = sns.relplot(x=self.data_new_1.index, y="validity",data=self.data_new_1,kind='scatter',hue='Difficulty',style="Difficulty")
        #     g.fig.suptitle("ScatterPlot of validity",fontsize=16, fontdict={"weight": "bold"})
        #     plt.savefig(os.path.join(myfolder,filename+"_Scatter_validity_1.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x=self.data_new_2.index, y="validity",data=self.data_new_2,kind='scatter',hue='Difficulty',style="Difficulty")
            g.fig.suptitle("ScatterPlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Scatter_validity_2.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x=self.data_new_2.index, y="num_steps",data=self.data_new_2,kind='scatter',hue='Difficulty',style="Difficulty")
            g.fig.suptitle("ScatterPlot of num_steps",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Scatter_num_steps_2.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x=self.data_new_2.index, y="reward",data=self.data_new_2,kind='scatter',hue='Difficulty',style="Difficulty")
            g.fig.suptitle("ScatterPlot of reward",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Scatter_reward_2.png"))

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

class plt_ObjectSearch():
    def __init__(self,filename):
        self.data = pd.read_csv(os.path.join(myfolder,filename+".csv"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="num_steps",data=self.data,kind='line')
            g.fig.suptitle("LinePlot of num_steps",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+"num_steps.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="reward",data=self.data,kind='line')
            g.fig.suptitle("LinePlot of reward",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+"reward.png"))        
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="success_validity",data=self.data,kind='line')
            # g = sns.relplot(x="epi_iter", y="validity",data=self.data,kind='line',hue='episode_id',style="episode_id")
            g.fig.suptitle("LinePlot of success_validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+"success_validity.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="validity",data=self.data,kind='line')
            g.fig.suptitle("LinePlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+"validity.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="entropy",data=self.data,kind='line')
            g.fig.suptitle("LinePlot of entropy",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+"entropy.png"))
        plt.show()

class plt_Hier():
    def __init__(self,filename):
        self.data = pd.read_csv(os.path.join(myfolder,filename+".csv"))
        weight = 0.99
        tmp_data = self.data.values
        for i in range(len(self.data)-1):
            tmp_data[i+1,3:] = weight*tmp_data[i,3:] + (1-weight)*tmp_data[i+1,3:]
        self.data = pd.DataFrame(tmp_data)
        self.data.columns = ["epi_iter","zhr_difficulty","episode_id","num_steps","reward","success","validity","success_validity","total"]
        self.data.epi_iter = pd.to_numeric(self.data.epi_iter, downcast="integer")
        # self.data.num_steps = pd.to_numeric(self.data.num_steps, downcast="integer")
        self.data.num_steps = 160 - pd.to_numeric(self.data.num_steps, downcast="integer")
        self.data.reward = pd.to_numeric(self.data.reward, downcast="float")
        self.data.success = pd.to_numeric(self.data.success, downcast="float")
        self.data.validity = pd.to_numeric(self.data.validity, downcast="float")
        self.data.success_validity  = pd.to_numeric(self.data.success_validity , downcast="float")
        # # data = data[data.epi_iter<101]
        # from scipy.signal import savgol_filter
        # a=199
        # b=3
        # num_epis = sum(data.epi_iter==0)
        # tmp = list(np.zeros(num_epis))
        # for i in range(num_epis):
        #     tmp[i] = data.iloc[i::num_epis]
        #     tmp[i].num_steps = savgol_filter(tmp[i].num_steps,a,b,mode="nearest")
        #     tmp[i].reward = savgol_filter(tmp[i].reward,a,b,mode="nearest")
        #     tmp[i].success = savgol_filter(tmp[i].success,a,b,mode="nearest")
        #     tmp[i].validity = savgol_filter(tmp[i].validity,a,b,mode="nearest")
        #     tmp[i].success_validity = savgol_filter(tmp[i].success_validity,a,b,mode="nearest")
        # data = pd.concat([tmp[0],tmp[1]])
        # data = pd.concat([data,tmp[2]])
        # data = pd.concat([data,tmp[3]])
        # data = pd.concat([data,tmp[4]])
        # data = pd.concat([data,tmp[5]])
        # tmp = "episode_id"
        tmp = None
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="reward",data=self.data,kind='line',hue=tmp)
            g.fig.suptitle("LinePlot of reward",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+f"{weight}_"+"reward.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="num_steps",data=self.data,kind='line',hue=tmp)
            g.fig.suptitle("LinePlot of num_steps",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+f"{weight}_"+"num_steps.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="success_validity",data=self.data,kind='line',hue=tmp)
            g.fig.suptitle("LinePlot of success_validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+f"{weight}_"+"success_validity.png"))
        with sns.axes_style("darkgrid"):
            g = sns.relplot(x="epi_iter", y="validity",data=self.data,kind='line',hue=tmp)
            g.fig.suptitle("LinePlot of validity",fontsize=16, fontdict={"weight": "bold"})
            plt.savefig(os.path.join(myfolder,filename+"_Line_"+f"{weight}_"+"validity.png"))
        plt.show()

if False:
    plt_PointNav("PointNav_0_True")#0.68
    plt_PointNav("PointNav_1_Flase")#0.736
    plt_PointNav("PointNav_2_Flase")#0.738
    plt_PointNav("PointNav_3_Flase")#0.728

if False:
    plt_ObjectSearch("ObejectSearch_1_a1")
    plt_ObjectSearch("ObejectSearch_2_a2")
    plt_ObjectSearch("ObejectSearch_3_b1")
    plt_ObjectSearch("ObejectSearch_4_b2")
    plt_ObjectSearch("ObejectSearch_5_b3")
    plt_ObjectSearch("ObejectSearch_6_c1")
if False:
    plt_Hier("Hier_3_c1_easy")
    plt_Hier("Hier_4_c1_hard")
    plt_Hier("Hier_5_c1_medium")

plt_Hier("Hier_3_c1_easy")