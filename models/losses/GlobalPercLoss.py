import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# https://github.com/BurguerJohn/global_perceptual_similarity_loss/blob/main/GlobalPercLoss.py

class GlobalPercConfig():
  def __init__(
        self, start_weight=1.0, end_weight=2.0, curve_force=0,
        modules_to_hook=[], transform_normalization=None,  print_data=True
    ):
    
    self.start_weight = start_weight
    self.end_weight = end_weight
    self.curve_force = curve_force
    
    self.modules_to_hook = modules_to_hook
    self.print_data = print_data
    self.transform_normalization = transform_normalization


class GlobalPercLoss(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        
        self.config = config
        self.activations = []

        def getActivation():
            def hook(model, input, output):
                self.activations.append(output)
            return hook

        self.model = model.eval()

        count = 0

        def traverse_modules(module):
            nonlocal count
            for name, sub_module in module.named_children():                  
                traverse_modules(sub_module)

                if (len(config.modules_to_hook) == 0 and len(list(sub_module.named_children())) == 0) or isinstance(sub_module, tuple(config.modules_to_hook)):
                  if config.print_data:          
                    print("~ Hook in module:", sub_module)
                  count += 1
                  sub_module.register_forward_hook(getActivation())

        traverse_modules(self.model)

        if config.curve_force == 0:
            self.weights = np.linspace(config.start_weight, config.end_weight, count)
        elif config.start_weight <= config.end_weight:
            self.weights = self.ease_in_curve(config.start_weight, config.end_weight, config.curve_force, count)
        else:
            self.weights = self.ease_out_curve(config.start_weight, config.end_weight, config.curve_force, count)
        if config.print_data:
          print(f"~ Total Layers Hook: {count}")
          print(f"~ Weight for each Hook: ", self.weights)


        self.normalize = config.transform_normalization
        
    def ease_in_curve(self, start_value, end_value, curve_strength, qtd_points):
        # Generate a tensor of points from 0 to 1
        points = torch.linspace(0, 1, qtd_points)
        # Apply the ease-in curve (acceleration)
        eased_points = points ** curve_strength
        # Scale and offset the points to the desired range
        return start_value + (end_value - start_value) * eased_points

    def ease_out_curve(self, start_value, end_value, curve_strength, qtd_points):
        # Generate a tensor of points from 0 to 1
        points = torch.linspace(0, 1, qtd_points)
        # Apply the ease-out curve (deceleration)
        eased_points = 1 - (1 - points) ** curve_strength
        # Scale and offset the points to the desired range
        return start_value + (end_value - start_value) * eased_points

    def cosine_loss(self, A, B):
        if A.dim() > 3:
            A = A.view(A.size(0), A.size(1), -1)
            B = B.view(B.size(0), B.size(1), -1)
            
        return (1 - nn.functional.cosine_similarity(A, B.detach(), dim=-1)).pow(2).mean()

    def forward(self, X, Y):
        layers_loss  = self._forward_features(X, Y)
        loss = 0

        for i in range(len(layers_loss)):
            loss_l = layers_loss[i] * self.weights[i]
            loss += loss_l

        return loss

    def _forward_features(self, X, Y):
        if self.normalize:
            X = self.normalize(X)
            Y = self.normalize(Y.detach())

        self.activations = []

        if hasattr(self.model, 'get_nearest_supported_resolution'):
            nearest_res = self.model.get_nearest_supported_resolution(*X.shape[-2:])
            X = F.interpolate(X, nearest_res, mode='bilinear', align_corners=False)
            Y = F.interpolate(Y, nearest_res, mode='bilinear', align_corners=False)
        self.model(X)
        X_VAL = self.activations

        self.activations = []
        with torch.no_grad():
            self.model(Y) 
        Y_VAL = self.activations

        layers_loss = []
        for i in range(len(X_VAL)):
            A = X_VAL[i]
            B = Y_VAL[i]
            
            loss = self.cosine_loss(A, B) 
            layers_loss.append(loss)
              
        return layers_loss
    

def radiov2_5_loss():
    model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True)
    model.cuda().eval()

    config = GlobalPercConfig(
        start_weight=1.,
        end_weight=1.,
		curve_force = 1.,
        modules_to_hook=[nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU],
        transform_normalization=None,
        print_data=True
    )

    return GlobalPercLoss(model, config)