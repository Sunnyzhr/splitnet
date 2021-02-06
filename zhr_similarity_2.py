from networks.networks import RLBaseWithVisualEncoder
from torchvision import transforms
from PIL import Image  
from networks.networks import ShallowVisualEncoder
import torch
import torchvision.models as models
import numpy as np

# alexnet = models.alexnet(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)


def get_similarity(i,see="no_see"):
    trans = transforms.Compose([transforms.ToTensor()])
    n_target="Denmark_target"
    n_obs=f"Denmark_obs_{i}" if see=="no_see" else f"Denmark_see_{i}"
    target = Image.open(f"/home/u/Desktop/splitnet/zhr/{n_target}.png").convert('RGB')
    # target.show()
    target_tensor = trans(target)
    targetFeature = mynet(torch.unsqueeze(target_tensor,0))
    targetFeature = targetFeature.detach()

    obs = Image.open(f"/home/u/Desktop/splitnet/zhr/{n_obs}.png").convert('RGB')
    # obs.show()
    obs_tensor = trans(obs)
    obsFeature = mynet(torch.unsqueeze(obs_tensor,0))
    obsFeature = obsFeature.detach()

    similarity = torch.cosine_similarity(obsFeature,targetFeature,dim=1)
    # tmp1 = np.reshape(obsFeature.detach().numpy(),[16,16])
    # tmp2 = np.reshape(targetFeature.detach().numpy(),[16,16])
    # (score, diff) = compare_ssim(tmp1,tmp2,full=True)
    # print(i,see,"ssim:",score,"cosine",np.array(self.similarity))
    print(i,see,"The cosine similarity is:",similarity)

if __name__=="__main__":
    mynet = models.resnet152(pretrained=True)
    # models.vgg16(pretrained=True)
    for i in range(6):
        get_similarity(i,see="no_see")
    for i in range(3):
        get_similarity(i,see="see")


