
import numpy as np
import torchvision.models as models
from torch import nn

a = list(models.resnet50(pretrained=True).children())
b = nn.Sequential(*a)
print(b)

