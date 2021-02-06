from networks.networks import RLBaseWithVisualEncoder
from torchvision import transforms
from PIL import Image  
from networks.networks import ShallowVisualEncoder
import torch
from skimage.measure import compare_ssim
import numpy as np
class Similar(RLBaseWithVisualEncoder):
    def __init__(self):
        super(Similar,self).__init__( 
        encoder_type=ShallowVisualEncoder,
        decoder_output_info=[("reconstruction", 3), ("depth", 1), ("surface_normals", 3)],
        recurrent=True,
        end_to_end=False,
        hidden_size=256,
        target_vector_size=2,
        action_size=3,
        gpu_ids="0",
        create_decoder=True,
        blind=False,
        )

    def get_similarity(self,i,see="no_see"):
        trans = transforms.Compose([transforms.ToTensor()])
        n_target="Denmark_target"
        n_obs=f"Denmark_obs_{i}" if see=="no_see" else f"Denmark_see_{i}"

        target = Image.open(f"/home/u/Desktop/splitnet/zhr/{n_target}.png").convert('RGB')
        # target.show()
        target_tensor = trans(target)
        self.targetFeature , _ , _ = self.visual_encoder(torch.unsqueeze(target_tensor,0),False)
        self.targetFeature = self.targetFeature.detach()
        self.targetFeature = self.visual_projection(self.targetFeature)

        obs = Image.open(f"/home/u/Desktop/splitnet/zhr/{n_obs}.png").convert('RGB')
        # obs.show()
        obs_tensor = trans(obs)
        self.obsFeature , _ , _ = self.visual_encoder(torch.unsqueeze(obs_tensor,0),False)
        self.obsFeature = self.obsFeature.detach()
        self.obsFeature = self.visual_projection(self.obsFeature)

        # self.obsFeature=torch.rand(1,256)
        self.similarity = torch.cosine_similarity(self.obsFeature,self.targetFeature,dim=1).detach()
        print("The cosine similarity is:",self.similarity)

        tmp1 = np.reshape(self.obsFeature.detach().numpy(),[16,16])
        tmp2 = np.reshape(self.targetFeature.detach().numpy(),[16,16])
        (score, diff) = compare_ssim(tmp1,tmp2,full=True)
        diff = (diff * 255).astype("float32")
        print(i,see,"ssim:",score,"cosine",np.array(self.similarity))

def main():
    runner = Similar()
    for i in range(6):
        runner.get_similarity(i,see="no_see")
    for i in range(3):
        runner.get_similarity(i,see="see")

if __name__ == "__main__":
    main()
