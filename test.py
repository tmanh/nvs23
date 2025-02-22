import torch

from models.synthesis.fwd import FWD
from models.layers.upsampler import JBUStack
from models.layers.fuse import SNetDS2BNBase8


model = SNetDS2BNBase8().cuda()

x = model(torch.rand(1, 3, 512, 384).cuda())
print(x.shape)

exit()

model = torch.hub.load("nx-ai/vision-lstm", "VisionLSTM2")

# load your own model
model = torch.hub.load(
    "nx-ai/vision-lstm", 
    "VisionLSTM2",  # VisionLSTM2 is an improved version over VisionLSTM
    dim=192,  # latent dimension (192 for ViL-T)
    depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
    patch_size=16,  # patch_size (results in 196 patches for 224x224 images)
    input_shape=(3, 224, 224),  # RGB images with resolution 224x224
    output_shape=(1000,),  # classifier with 1000 classes
    drop_path_rate=0.05,  # stochastic depth parameter
)

print(model)