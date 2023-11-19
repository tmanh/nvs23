import torch

import torch.nn as nn
import torch.autograd.profiler as profiler

from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model
        self.opt = opt

        depth_model_name = ["mvs_depth_estimator", "pts_regressor"]
        depth_params = list(filter(lambda kv: kv[0].split(".")[1] in depth_model_name, self.model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0].split(".")[1] not in depth_model_name, self.model.named_parameters()))
        base_params = [i[1] for i in base_params]
        depth_params =  [i[1] for i in depth_params]
        
        self.optimizer_G = torch.optim.Adam(
            [
                {"params": base_params},
                {"params": depth_params, "lr": opt.lr_g * opt.depth_lr_scaling}
            ],
            lr=opt.lr_g,
            betas=(opt.beta1, opt.beta2)
        )

        if opt.isTrain:
            self.old_lr = opt.lr

        if opt.init:
            self.init_weights()

    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if self.opt.init == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif self.opt.init == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif self.opt.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0) 
                elif self.opt.init == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.opt.init == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif self.opt.init == "":
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f"initialization method [{self.opt.init}] is not implemented")

                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def enable_training(self, iter_count):
        self.model.module.enable_training(iter_count)

    def __call__(self, iter_dataloader, isval=False, num_steps=1):
        """
        Main function call
        - iter_dataloader: The sampler that choose data samples.
        - isval: Whether to train the discriminator etc.
        - num steps: not fully implemented but is number of steps in the discriminator for
        each in the generator
        - return_batch: Whether to return the input values
        """
        weight = 1.0 / float(num_steps)
        if isval:
            with profiler.record_function("load data"):
                batch = next(iter_dataloader)
            with profiler.record_function("run model"):
                t_losses, output_images = self.model(batch)
            
            if self.opt.normalize_image:
                for k in output_images.keys():
                    if "Img" in k:
                        output_images[k] = 0.5 * output_images[k] + 0.5 if torch.is_tensor(output_images[k]) else [0.5 * output_image + 0.5 for output_image in output_images[k]]

            return t_losses, output_images
        
        self.optimizer_G.zero_grad()
        
        for _ in range(num_steps):
            with profiler.record_function("load data"):
                batch = next(iter_dataloader)
            with profiler.record_function("run model"):
                t_losses, output_images = self.model(batch)
                del batch
            (t_losses["Total Loss"] / weight).mean().backward()
        self.optimizer_G.step()

        if self.opt.normalize_image:
            for k in output_images.keys():
                if "Img" in k:
                    output_images[k] = 0.5 * output_images[k] + 0.5 if torch.is_tensor(output_images[k]) else [0.5 * output_image + 0.5 for output_image in output_images[k]]

        return t_losses, output_images

    def lr_annealing(self, scale):
        for g in self.optimizer_G.param_groups:
            g['lr'] = g['lr'] * scale
