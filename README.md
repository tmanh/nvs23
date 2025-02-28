# Novel View Synthesis

## TOC

- [Dependencies](#dependencies)
- [Directory](#directory)

## Getting Started

### Dependencies

```

# Init the environment
conda env create -f environment.yml

# Install pytorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install Adaptive Conv
cd models/layers/adaptive_conv_cuda
python setup.py install
```

### Directory
```
.
|____environment.yml                    # To install the dependencies
|____configs                            # All the config files using to init the dataloader and the model for training process
|____utils
|____models
| |____losses                           # All the loss functions are defined here
| |____projection                       # To handle the projection
| | |____z_buffer_layers.py             # The core implementation of the point rasterization
| | |____z_buffer_manipulator.py        # Wrapper of the point rasterization class (calculating the mapping)
| |____synthesis
| | |____base.py                        # Base rendering class 
| | |____encoder.py                     # Implement different multi-scale feature extraction approaches
| | |____deepblendplus.py               # Deep Blending (Unet)
| | |____fwd.py                         # FWD (the blending part)
| | |____global_syn.py                  # Not done yet
| | |____local_syn.py                   # Similar to Deep Blending but just consider the small window instead
| |____layers                           # Different architectures that are used (or used in the past) in synthesis 
|____data                               # Dataloader
|____training.sh                        # Start the training here
|____eval.py                            # Simple testing code, not evaluation yet
|____training.py                        # All training things happen here
```

## TODO

- Optimizing pose + depth
[ ] Check the posibility of Met3r
[ ] Check the posibility of SDSLoss
[ ] Check the posibility