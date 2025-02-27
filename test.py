import torch

from models.synthesis.fwd import FWD
from models.layers.upsampler import JBUStack
from models.layers.fuse import SNetDS2BNBase8
from models.synthesis.encoder import MultiScaleEfficientNetV2


model = MultiScaleEfficientNetV2().cuda()
x = torch.rand(4, 3, 256, 256).cuda()

feats = model(x, 1, 4)
for f in feats:
    print(f.shape)