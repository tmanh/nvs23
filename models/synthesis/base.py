import math
import random
import numpy as np

from models.layers.module import differentiable_warping, single_warping

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from models.losses.synthesis import SynthesisLoss
from models.losses.sigloss import SigLoss
from models.losses.multi_view import *

from models.networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.projection.z_buffer_manipulator import Screen_PtsManipulator


class BaseModule(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.init_hyper_params(opt)
        self.init_models()
        self.init_loss()
        self.freeze()

    def enable_training(self, iter_count):
        self.train()
        self.freeze()

        self.sigloss.warm_up_counter = iter_count

    @staticmethod
    def allocated():
        current = torch.cuda.memory_allocated(0)/1024/1024/1024
        print(f'allocated: {current} GB')
        return current

    @staticmethod
    def gain(previous):
        current = torch.cuda.memory_allocated(0)/1024/1024/1024
        print(f'allocated: {current} GB')
        print(f'gain: {current - previous} GB')

    def init_hyper_params(self, opt):
        ##### LOAD PARAMETERS
        opt.decode_in_dim = 1
        self.opt = opt

        # Use H if specifid in opt or H = W
        self.H = opt.H if hasattr(self.opt, "H") else opt.W
        self.W = opt.W
        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer("max_z", torch.Tensor([self.opt.max_z]))
        
        self.scale_factor = opt.scale_factor
        self.inverse_depth = self.opt.inverse_depth

        # Whether using depth completions. It is expected when taking sensor depths or MVS estimated depths as inputs.
        self.depth_com = opt.depth_com

    def init_models(self, encoder_type='Resnet'):
        self.init_depth_module()
        out_dim = self.init_color_encoder()
        out_dim = self.init_latent_encoder(out_dim)
        self.init_renderer()
        self.init_fusion_module()

    def init_depth_module(self):
        pass

    def init_color_encoder(self):
        pass

    def init_latent_encoder(self, out_dim=64, n_view_feats=32, n_color_feats=64):
        pass
    
    def init_renderer(self):
        width = self.opt.DW if self.opt.down_sample else self.opt.W
        height = self.opt.DH if self.opt.down_sample else self.opt.H

        self.pts_transformer = Screen_PtsManipulator(W=width, H=height, opt=self.opt)

    def init_fusion_module(self):
        pass

    def init_loss(self):
        self.sigloss = SigLoss()
        self.mvloss = ReprojectionLoss()
        self.loss_function = SynthesisLoss(opt=self.opt)

    ##### GETTING DATA ###################################

    def process_input_data(self, batch, num_inputs):
        input_imgs = []
        original_imgs = []
        for i in range(num_inputs):
            input_img = deepcopy(batch["images"][i])
            H, W = input_img.shape[-2:]

            down_sample_flag = self.opt.down_sample if self.opt.dataset != 'folder' else random.choice([True, False])

            if down_sample_flag:
                if self.opt.DH > 0 and self.opt.DW > 0:
                    input_img = F.interpolate(input_img, size=(self.opt.DH, self.opt.DW), mode="area")
                else:
                    input_img = F.interpolate(input_img, size=(H // 2, W // 2), mode="area")

            original_img = deepcopy(input_img)

            original_imgs.append(original_img)
            input_imgs.append(input_img)

        original_imgs = torch.stack(original_imgs, 1) # B x num_inputs x C x H x W
        input_imgs = torch.stack(input_imgs, 1) # B x num_inputs x C x H x W

        return original_imgs, input_imgs

    def process_output_data(self, batch, num_inputs, num_outputs):
        if self.opt.dataset == 'folder':
            output_imgs = [batch["images"][i] for i in range(num_outputs)]
        else:
            output_imgs = [batch["images"][i+num_inputs] for i in range(num_outputs)]
        output_imgs = torch.stack(output_imgs, 1)  # B x num_outputs x C x H x W
        return output_imgs

    def data_process(self, batch):
        num_inputs = self.opt.input_view_num if self.opt.dataset != 'folder' else 1
        num_outputs = len(batch['images']) - num_inputs if self.opt.dataset != 'folder' else 1

        original_imgs, input_imgs = self.process_input_data(batch, num_inputs)         
        output_imgs = self.process_output_data(batch, num_inputs, num_outputs)

        if self.opt.dataset != 'folder':
            K, K_inv = self.extract_intrinsic_parameters(batch)
            input_RTs, input_RTinvs, output_RTs, output_RTinvs = self.extract_extrinsic_parameters(batch, num_inputs, num_outputs)

            return self.to_cuda(
                original_imgs, input_imgs, output_imgs, K, K_inv, input_RTs, input_RTinvs, output_RTs, output_RTinvs)
        else:
            original_imgs, input_imgs, output_imgs = self.to_cuda(original_imgs, input_imgs, output_imgs)
            return original_imgs, input_imgs, output_imgs, None, None, None, None, None, None

    def to_cuda(self, *args):
        if torch.cuda.is_available():
            new_args = [args[i].cuda() for i in range(len(args))]
        return new_args

    def extract_extrinsic_parameters(self, batch, num_inputs, num_outputs):
        input_RTs = []
        input_RTinvs = []
        for i in range(num_inputs):
            input_RTs.append(batch["cameras"][i]["P"])
            input_RTinvs.append(batch["cameras"][i]["Pinv"])
        input_RTs = torch.stack(input_RTs, 1)
        input_RTinvs = torch.stack(input_RTinvs, 1)
        
        output_RTs =  []
        output_RTinvs = []
        for i in range(num_outputs):
            output_RTs.append(batch["cameras"][i+num_inputs]["P"])
            output_RTinvs.append(batch["cameras"][i+num_inputs]["Pinv"])
        output_RTs = torch.stack(output_RTs, 1)
        output_RTinvs = torch.stack(output_RTinvs, 1)
        
        return input_RTs, input_RTinvs, output_RTs, output_RTinvs

    def extract_intrinsic_parameters(self, batch):
        K = deepcopy(batch["cameras"][0]["K"]).clone()
        K_inv = deepcopy(batch["cameras"][0]["Kinv"]).clone()
        if self.opt.down_sample:
            scale = self.opt.H / self.opt.DH
            K[:, 0:2, 0:3] = K[:, 0:2, 0:3] / scale
            K_inv = torch.inverse(K)
        return K, K_inv
    
    def depth_tensor_from(self, batch, height, width):
        src_depth_imgs = []
        dst_depth_imgs = []
        for i, b in enumerate(batch):
            if i < self.opt.input_view_num:
                src_depth_imgs.append(
                    b.cuda() if torch.cuda.is_available() else b
                )
            else:
                dst_depth_imgs.append(
                    b.cuda() if torch.cuda.is_available() else b
                )

        src_depth_imgs = torch.cat(src_depth_imgs, 1)
        dst_depth_imgs = torch.cat(dst_depth_imgs, 1)
        if self.opt.down_sample:
            src_depth_imgs = F.interpolate(
                src_depth_imgs, size=(height, width), mode="nearest"
            )
            dst_depth_imgs = F.interpolate(
                dst_depth_imgs, size=(height, width), mode="nearest"
            )
        src_depth_imgs = src_depth_imgs.unsqueeze(2)
        dst_depth_imgs = dst_depth_imgs.unsqueeze(2)
        return src_depth_imgs, dst_depth_imgs

    def get_raw_depth(self, batch, height=None, width=None, isval=False):
        num_inputs = self.opt.input_view_num

        if self.opt.down_sample:
            height = self.opt.DH if height is None else height
            width = self.opt.DW if width is None else width

        # Use provided incomplete sensor depths.
        if "depths" in batch.keys():
            src_depths, dst_depths = self.depth_tensor_from(batch["depths"], height, width)

        augmented_masks = None
        if "augmented_mask" in batch.keys():
            augmented_masks = batch["augmented_mask"][:num_inputs]
            if torch.cuda.is_available():
                augmented_masks = [a.cuda() for a in augmented_masks]

            if self.opt.down_sample:
                augmented_masks = torch.cat(augmented_masks, 1)
                augmented_masks = F.interpolate(augmented_masks, size=(height, width), mode="nearest").unsqueeze(2)
            else:
                augmented_masks = torch.stack(augmented_masks, 1)  # B x num_outputs x 1 x H x W

        return src_depths, dst_depths, augmented_masks

    ##### EVAL ###################################

    def eval_batch(self, batch, chunk=8, **kwargs):
        total_time = 0
        num_inputs = kwargs.get('num_view', self.opt.input_view_num)
        num_outputs = len(batch["images"]) - num_inputs
        num_chunks = math.ceil(num_outputs / chunk)

        results = {}
        for chunk_idx in range(num_chunks):
            endoff = min(num_inputs + chunk_idx * chunk + chunk, num_outputs + num_inputs)
            start = num_inputs + chunk_idx * chunk
            instance_num = int(endoff - start)
            new_batch = {'path': batch['path'], 'img_id': batch['img_id'], 'images': batch['images'][:num_inputs]}

            new_batch['images'] = [new_batch['images'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]

            new_batch['images'].append(torch.cat(batch['images'][start:endoff], 0))

            new_camera = {item: [] for item in batch['cameras'][0]}
            for instance in batch['cameras']:
                for item in instance:
                    new_camera[item].append(instance[item])
            camera_list = []
            for i in range(num_inputs):
                camera_tmp = {}
                for item in new_camera:
                    camera_tmp[item] = new_camera[item][i]
                    the_shape = camera_tmp[item].shape
                    new_shhape = (instance_num, ) + the_shape[1:]
                    camera_tmp[item] = camera_tmp[item].expand(new_shhape).clone()
                camera_list.append(camera_tmp)
            camera_tmp = {item: torch.cat(new_camera[item][start:endoff]) for item in new_camera}

            camera_list.append(camera_tmp)
            new_batch['cameras'] = camera_list

            if "depths" in batch.keys():
                new_batch['depths'] = batch['depths'][:num_inputs]
                new_batch['depths'] = [new_batch['depths'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
                new_batch['depths'].append(torch.cat(batch['depths'][start:endoff], 0))

                if 'mdepths' in batch.keys():
                    new_batch['mdepths'] = batch['mdepths'][:num_inputs]
                    new_batch['mdepths'] = [new_batch['mdepths'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
                    new_batch['mdepths'].append(torch.cat(batch['mdepths'][start:endoff], 0))

                # TODO: comment
                new_batch['augmented_mask'] = batch['augmented_mask'][:num_inputs]
                new_batch['augmented_mask'] = [new_batch['augmented_mask'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
                new_batch['augmented_mask'].append(torch.cat(batch['augmented_mask'][start:endoff], 0))

            elapsed, result = self.eval_one_step(new_batch)
            total_time += elapsed
            results[chunk_idx] = result

        new_results = {}
        for term in results[0].keys():
            buffer = [results[i][term] for i in range(len(results))]
            new_results[term] = torch.cat(buffer, 0)
        return [total_time, new_results]
    
    def eval_speed(self, batch, chunk=8, **kwargs):
        total_time = 0
        n_samples = 0

        num_inputs = kwargs.get('num_view', self.opt.input_view_num)
        num_outputs = len(batch["images"]) - num_inputs
        num_chunks = math.ceil(num_outputs / chunk)

        for chunk_idx in range(num_chunks):
            endoff = min(num_inputs + chunk_idx * chunk + chunk, num_outputs + num_inputs)
            start = num_inputs + chunk_idx * chunk
            instance_num = int(endoff - start)
            
            new_batch = {'path': batch['path'], 'img_id': batch['img_id'], 'images': batch['images'][:num_inputs]}
            new_batch['images'] = [new_batch['images'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
            new_batch['images'].append(torch.cat(batch['images'][start:endoff], 0))

            new_camera = {item: [] for item in batch['cameras'][0]}
            for instance in batch['cameras']:
                for item in instance:
                    new_camera[item].append(instance[item])
            camera_list = []
            for i in range(num_inputs):
                camera_tmp = {}
                for item in new_camera:
                    camera_tmp[item] = new_camera[item][i]
                    the_shape = camera_tmp[item].shape
                    new_shhape = (instance_num, ) + the_shape[1:]
                    camera_tmp[item] = camera_tmp[item].expand(new_shhape).clone()
                camera_list.append(camera_tmp)
            camera_tmp = {item: torch.cat(new_camera[item][start:endoff]) for item in new_camera}

            camera_list.append(camera_tmp)
            new_batch['cameras'] = camera_list

            if "depths" in batch.keys():
                new_batch['depths'] = batch['depths'][:num_inputs]
                new_batch['depths'] = [new_batch['depths'][i].expand(instance_num, -1, -1, -1) for i in range(num_inputs)]
                new_batch['depths'].append(torch.cat(batch['depths'][start:endoff], 0))

            elapsed, samples, _ = self.eval_one(new_batch)

            total_time += elapsed
            n_samples += samples

        return total_time, n_samples

    def eval_one_step(self, batch):
        pass

    def eval_one(self, batch):
        pass

    ##### FEATURES ##################################

    def compute_ray_diff(self, ray2train_pose, ray2tar_pose):
        ray2train_pose = ray2train_pose / (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-06)

        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-06)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        return ray_diff

    def com_view_diff(self, xyz, source_camera, target_camera):
        """ Compute the view direction change from source camera to target camera. 
        Inputs:
            -- xyz: [BS, N, 3], points positions. BS: batch size. N: point numbers. 
            -- source_camera: [BS, 4, 4]
            -- target_camera: [BS, nviews, 4, 4]
        Outputs:
            -- [BS, num_views, N, 4]; The first 3 channels are unit-length vector of the difference between
            query and target ray directions, the last channel is the inner product of the two directions.
        """
        ray2tar_pose = (source_camera[:, :3, 3].unsqueeze(1) - xyz).unsqueeze(1) # Bs x 1 x N x 3
        ray2tar_pose = ray2tar_pose / (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (target_camera[:, :, :3, 3].unsqueeze(2) - xyz.unsqueeze(1)) # Bs x nviews x N x 3
        return self.compute_ray_diff(ray2train_pose, ray2tar_pose)
    
    def com_view_diff_single(self, xyz, source_camera, target_camera):
        """ Compute the view direction change from source camera to target camera. 
        Inputs:
            -- xyz: [BS, N, 3], points positions. BS: batch size. N: point numbers. 
            -- source_camera: [BS, 4, 4]
            -- target_camera: [BS, 4, 4]
        Outputs:
            -- [BS, N, 4]; The first 3 channels are unit-length vector of the difference between
            query and target ray directions, the last channel is the inner product of the two directions.
        """
        ray2tar_pose = (source_camera[:, :3, 3].unsqueeze(1) - xyz)
        ray2tar_pose = ray2tar_pose / (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (target_camera[:, :3, 3].unsqueeze(1) - xyz)
        return self.compute_ray_diff(ray2train_pose, ray2tar_pose)
    
    ##### DATA AUGMENTATION ###################################

    def augment(self, output_imgs, warped, rotation=False):
        if rotation:
            rotate_index = np.random.randint(0, 4)
            warped = torch.rot90(warped, k=rotate_index, dims=[-2, -1])
            output_imgs = torch.rot90(output_imgs, k=rotate_index, dims=[-2, -1])

        flip_index = np.random.randint(0, 1)
        warped = self.augment_flip(warped, flip_index)
        output_imgs = self.augment_flip(output_imgs, flip_index)

        return output_imgs, warped

    def augment_with_feats(self, output_imgs, gen_fs, warped, rotation=False):
        if rotation:
            rotate_index = np.random.randint(0, 4)
            gen_fs = torch.rot90(gen_fs, k=rotate_index, dims=[-2, -1])
            warped = torch.rot90(warped, k=rotate_index, dims=[-2, -1])
            output_imgs = torch.rot90(output_imgs, k=rotate_index, dims=[-2, -1])

        flip_index = np.random.randint(0, 1)
        warped = self.augment_flip(warped, flip_index)
        output_imgs = self.augment_flip(output_imgs, flip_index)
        gen_fs = self.augment_flip(gen_fs, flip_index)

        return output_imgs, gen_fs, warped
    
    def augment_flip(self, tensor, flip_index):
        if flip_index == 1:
            tensor = torch.flip(tensor, dims=[-1])
        elif flip_index == 2:
            tensor = torch.flip(tensor, dims=[-2])
        elif flip_index == 3:
            tensor = torch.flip(tensor, dims=[-2, -1])
        
        return tensor
    
    ##### DATA AUGMENTATION ###################################
    
    def freeze(self):
        self.freeze_shallow_color_encoder()

    def freeze_shallow_color_encoder(self):
        self.encoder.backbone.eval()
        for param in self.encoder.backbone.parameters():
            param.requires_grad = False

    ##### DEPTH LOSS ###################################

    def add_depth_loss(
            self, src_depths, dst_depths, src_imgs, dst_imgs,
            K, input_RTs, output_RTs, src_coms, src_ests):
        scaled = F.interpolate(src_depths, size=src_ests.shape[-2:], mode='nearest')
        depth_loss = self.apply_sigloss(scaled, src_coms, src_ests)
        # depth_loss = self.apply_basic_loss(scaled, src_ests) * 0.5
        # depth_loss += 0.02 * self.apply_consis_loss(
        #     src_imgs, dst_imgs, src_ests, dst_depths, K, input_RTs, output_RTs)
        return depth_loss
    
    def add_depth_loss_fwd(self, src_depths, src_coms):
        depth_loss = self.apply_basic_loss(src_depths, src_coms)
        return depth_loss

    def apply_sigloss(self, gt_depth, completed, estimated):
        depth_loss = 0
        if self.opt.train_depth:
            depth_loss += self.sigloss(estimated, gt_depth)
            # depth_loss += self.sigloss(completed, gt_depth)
        return depth_loss

    def apply_consis_loss(
            self, src_imgs, dst_imgs, src_ests, dst_depths, K, input_RTs, output_RTs):
        b, v = input_RTs.shape[:2]
        c, h, w = src_imgs.shape[-3:]
        src_imgs = src_imgs.view(b, v, c, h, w)

        dst_imgs = dst_imgs.view(b, c, *dst_imgs.shape[-2:])
        dst_imgs = F.interpolate(dst_imgs, size=src_imgs.shape[-2:], mode='nearest')
        
        scaled = F.interpolate(src_ests, size=src_imgs.shape[-2:], mode='nearest')
        scaled = scaled.view(b, v, 1, h, w)
        dst_depths = dst_depths.view(b, 1, h, w)
        
        k = 0
        total = 0
        mask = dst_depths > 0
        for i in range(v):
            wc, pd = single_warping(
                src_imgs[:, i], scaled[:, i],
                K, K, input_RTs[:, i], output_RTs[:, 0], scaled[:, i])

            total += self.apply_basic_loss(pd[mask], dst_depths[mask])
            total += torch.mean(self.mvloss(wc, dst_imgs)[mask])
            k += 1

            # import cv2
            # vis = dst_depths[0, 0]
            # vis = vis / vis.max() * 255
            # vis = vis.detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite('vis1.png', vis)

            # vis = pd[0, 0]
            # vis = vis / vis.max() * 255
            # vis = vis.detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite('vis2.png', vis)

            # vis = (dst_imgs[0].permute(1, 2, 0) + 1) / 2 * 255
            # vis = vis.detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite('vis3.png', vis)

            # vis = (wc[0].permute(1, 2, 0) + 1) / 2 * 255
            # vis = vis.detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite('vis4.png', vis)
            # exit()
        
        return total / k
    
    def compute_h_consis_loss(self, input_imgs, scores, depths, K, K_inv, input_RTs, n_samples, n_views, height, width):
        input_imgs = input_imgs.view(n_samples, n_views, 3, height, width)

        vis_loss = 0
        for i in range(n_views):
            for j in range(n_views):
                if j == i:
                    continue

                warped = differentiable_warping(
                    input_imgs[:, j], K, K_inv, input_RTs[:, j], input_RTs[:, i],
                    depths[:, i, :, :height, :width].contiguous())
                merged = torch.sum(warped * scores[:, i].unsqueeze(1), dim=2)

                vis_loss += self.l1(merged, input_imgs[:, i])
        
        vis_loss /= n_views * (n_views - 1)
        
        return vis_loss
    
    def scale_intrinsic(self, K, oh, ow, sh, sw):
        with torch.no_grad():
            lk = K.clone()
            sx, sy = sw / ow, sh / oh

            lk[:, 0, :] = sx * lk[:, 0, :]
            lk[:, 1, :] = sy * lk[:, 1, :]

            lkinv = torch.inverse(lk)
        
        return lk, lkinv

    def apply_basic_loss(self, gt_depth, regressed_pts):
        depth_loss = 0

        with torch.no_grad():
            valid = gt_depth > 0.0

        depth_loss += F.l1_loss(regressed_pts[valid], gt_depth[valid])
        depth_loss += F.mse_loss(regressed_pts[valid], gt_depth[valid])

        return depth_loss