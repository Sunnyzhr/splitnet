import torch
pthfile=r"/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/checkpoints/2021_01_23_00_00_00/[2]128*388_0.68.pt"
net = torch.load(pthfile,map_location=torch.device('cpu'))
for i in net.keys():
    print(i,net[i].size())
print("\n"*5)

# save_pth="/home/u/Desktop/splitnet/output_files/pointnav/gibson/splitnet_pretrain_supervised_rl/checkpoints/2021_01_23_00_00_00/BEST_no_FC_A_C_GRU.pt"
# for i in list(net.keys()):
#     if "rl_layers" in i  or "critic" in i or "dist" in i or "gru" in i:
#         del net[i]
# for i in net.keys():
#     print(i+'   '+str(len(net[i])))
# torch.save(net,save_pth)
# print("\n"*5)

# tmp=torch.load(save_pth,map_location=torch.device('cpu'))
# for i in tmp.keys():
#     print(i+'   '+str(len(tmp[i])))

