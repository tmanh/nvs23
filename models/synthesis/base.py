import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as functional

from copy import deepcopy

from models.losses.multi_view_depth_loss import *


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        v_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()

        return (h_tv + v_tv) / (batch_size * channels * height * width) * 0.1


class BaseModule(nn.Module):
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
                    input_img = functional.interpolate(input_img, size=(self.opt.DH, self.opt.DW), mode="area")
                else:
                    input_img = functional.interpolate(input_img, size=(H // 2, W // 2), mode="area")

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
        else:
            new_args = args
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
        depth_imgs = []
        num_inputs = self.opt.input_view_num
        for i in range(num_inputs):
            if torch.cuda.is_available():
                depth_imgs.append(batch[i].cuda())
            else:
                depth_imgs.append(batch[i])

        if self.opt.down_sample:
            depth_imgs = torch.cat(depth_imgs, 1)
            return functional.interpolate(
                depth_imgs, size=(height, width), mode="nearest"
            ).unsqueeze(2)
        else:
            return torch.stack(depth_imgs, 1)

    def get_raw_depth(self, batch, height=None, width=None):
        raw_depth = None
        predicted_depth = None
        num_inputs = self.opt.input_view_num

        if self.opt.down_sample:
            height = self.opt.DH if height is None else height
            width = self.opt.DW if width is None else width

        # Use provided incomplete sensor depths.
        if "depths" in batch.keys():
            raw_depth = self.depth_tensor_from(batch["depths"], height, width)
            merge_depth = self.depth_tensor_from(batch["mdepths"], height, width)

        if self.opt.mvs_depth and self.opt.depth_com:
            if self.opt.learnable_mvs:
                results, _, _ = self.depth_estimator(batch)
            else:
                with torch.no_grad():
                    results, _, _ = self.depth_estimator(batch)

            predicted_depth = torch.stack(results, 1)

        augmented_masks = None
        if "augmented_mask" in batch.keys():
            augmented_masks = batch["augmented_mask"][:num_inputs]
            if torch.cuda.is_available():
                augmented_masks = [a.cuda() for a in augmented_masks]

            if self.opt.down_sample:
                augmented_masks = torch.cat(augmented_masks, 1)
                augmented_masks = functional.interpolate(augmented_masks, size=(height, width), mode="nearest").unsqueeze(2)
            else:
                augmented_masks = torch.stack(augmented_masks, 1)  # B x num_outputs x 1 x H x W

        return raw_depth, merge_depth, predicted_depth, augmented_masks

    ##### DEPTH COMPLETION ###################################

    def com_depth_fwd(self, depth, color, input_RTs, K):
        """ Simple depth completion process in fwd paper
        Ref: https://github.com/Caoang327/fwd_code

        If the depth from get_init_depth is None, we need to estimate the depths.
        If the depth is not None, we complete the depths.

        Args:
            depth (torch.Tensor or None): Depth map of shape [B, 1, H, W], where B is the
                batch size, and H, W are the height and width of the depth map. If None,
                the depths need to be estimated.
            color (torch.Tensor): Color image of shape [B, C, H, W], where B is the batch
                size, C is the number of channels, and H, W are the height and width of
                the color image.
            input_RTs (torch.Tensor): Input camera transformation matrices of shape [B, M, 4, 4],
                where B is the batch size, M is the number of input views, and 4 represents
                the 4x4 transformation matrix.
            K (torch.Tensor): Intrinsic camera matrix of shape [B, M, 4, 4], where B
                is the batch size, M is the number of input views, and 4 represents the
                4x4 transformation matrix.
            batch (int or None, optional): The batch index to process. If None, all batches
                are processed. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                four tensors:
                - regressed_pts (torch.Tensor): Regressed 3D points of shape [B, M, N, 3], where
                    B is the batch size, M is the number of input views, N is the number of points
                    in the point cloud, and 3 represents the 3D coordinates (x, y, z).
                - ref_depth (torch.Tensor or None): Reference depth map of shape [B, 1, H, W], where
                    B is the batch size, and H, W are the height and width of the depth map. If None,
                    a reference depth map is not available.
                - refine_depth (torch.Tensor or None): Refined depth map of shape [B, 1, H, W], where
                    B is the batch size, and H, W are the height and width of the depth map. If None,
                    the depth map is not refined.
        """
        if self.opt.depth_com:
            inverse_depth_img = 1. / torch.clamp(depth, min=0.001)
            inverse_depth_img[depth < 0.001] = 0
            inverse_depth_img = inverse_depth_img / (1.0 / self.opt.min_z)
            feats = torch.cat((color, inverse_depth_img), dim=1)
        else:
            feats = color

        regressed_pts, _ = self.pts_regressor(feats, input_RTs, K)
        # regressed_pts, _ = self.pts_unet_regressor(feats, input_RTs, K)
        # macs, params = profile(self.pts_regressor, inputs=(feats, input_RTs, K))
        # print('Completion Module FLOPs: ', macs, ' - ', params)
        # exit()
        return regressed_pts
    
    def com_depth_light(self, depth, color):
        if self.opt.depth_com:
            inverse_depth_img = 1. / torch.clamp(depth, min=0.001)
            inverse_depth_img[depth < 0.001] = 0
            inverse_depth_img = inverse_depth_img / (1.0 / self.opt.min_z)
        else:
            inverse_depth_img = None

        completed, estimated, c_feats = self.pts_regressor(color, inverse_depth_img)

        estimated = self.convert_depth(estimated)
        completed = self.convert_depth(completed) if self.opt.depth_com else estimated

        return completed, estimated, c_feats
    
    def com_depth_light2(self, depth, color):
        with torch.no_grad():
            norm_depth = depth / self.opt.max_z
            norm_depth[depth < 0.001] = -1

        completed, estimated, c_feats = self.pts_regressor(color, norm_depth)

        return completed, estimated, c_feats
    
    def com_depth_syn(self, depth, color):
        with torch.no_grad():
            inverse_depth = 1. / torch.clamp(depth, min=0.001)
            inverse_depth[depth < 0.001] = 0
            inverse_depth = inverse_depth / (1.0 / self.opt.min_z)

        h_fine, completed, estimated, df, df2, cf, cf2 = self.pts_regressor(
            color, inverse_depth)

        estimated = self.convert_depth(estimated)
        completed = self.convert_depth(completed)

        return h_fine, completed, estimated, df, df2, cf, cf2

    def com_depth_syn_test(self, depth, color):
        with torch.no_grad():
            inverse_depth = self.norm_depth(depth)

        completed, _, _, df, df2, cf, cf2 = self.pts_regressor(
            color, inverse_depth)
        
        completed = self.convert_depth(completed)
        
        return completed, df, df2, cf, cf2

    def norm_depth(self, depth):
        inverse_depth = 1. / torch.clamp(depth, min=0.001)
        inverse_depth[depth < 0.001] = 0
        inverse_depth = inverse_depth / (1.0 / self.opt.min_z)
        return inverse_depth

    def convert_depth(self, depth):
        if self.opt.inverse_depth:
            out = depth * (1.0 / self.opt.min_z)
            out = (1.0 / torch.clamp(out, min=0.001)) * (out >= 0.001).float()

        elif self.opt.normalize_depth:
            out = depth * self.opt.max_z
        else:
            out = (depth * (self.opt.max_z - self.opt.min_z)) + self.opt.min_z
        return out

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
            new_batch = {'images': batch['images'][:num_inputs]}

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

    def scale_intrinsic(self, K, oh, ow, sh, sw):
        with torch.no_grad():
            lk = K.clone()
            sx, sy = sw / ow, sh / oh

            lk[:, 0, :] = sx * lk[:, 0, :]
            lk[:, 1, :] = sy * lk[:, 1, :]

            lkinv = torch.inverse(lk)
        
        return lk, lkinv
