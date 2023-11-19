import torch
import torch.nn.functional as functional
import torchvision.transforms.functional_tensor as F_t

import os
import cv2
import natsort

import os.path as osp
import numpy as np

from .util import get_image_to_tensor_balanced, ColorJitter
from .mask_generator import MaskGenerator


class ScannetDataset(torch.utils.data.Dataset):
    def __init__(self, stage="train", opts=None):
        super().__init__()
        path = opts.path
        
        self.input_view_num = opts.input_view_num
        self.num_views = opts.num_views
        self.base_path = path
        self.stage = stage
        self.image_size = opts.image_size
        assert osp.exists(self.base_path)
        
        self.train_path = osp.join(self.base_path, 'scans (train)')  # 'scans (train)'
        self.val_path = osp.join(self.base_path, 'scans (test)')

        if stage == "train":
            self.train = True
            dir_data = self.train_path
        elif stage in ["val", "test"]:
            self.train = False
            dir_data = self.val_path
        self.scan_all_data(dir_data)

        self.image_to_tensor = get_image_to_tensor_balanced()
        print("Loading Scannet dataset", self.base_path)

        self.scale_factor = opts.scale_factor
        self.max_imgs = opts.max_imgs
        self.z_near = opts.z_near
        self.z_far = opts.z_far

        self.colorjitter = ColorJitter()
        if self.stage == "test":
            self.test_view = list(map(int, opts.test_view.split()))

        self.prepare_for_training_depth = opts.train_depth_only

        self.mask_generator = MaskGenerator(196, 196, 1)

    def init_data_list(self):
        self.colors = []
        self.depths = []
        self.poses = []
        self.intrinsics = []
        self.counting = []
        self.n_samples = 0

    def scan_all_data(self, directory):
        self.init_data_list()

        folders = [osp.join(directory, f) for f in os.listdir(directory)]
        folders.sort()
        for f in folders:
            pose_path = osp.join(f, 'pose')
            all_poses = [osp.join(pose_path, d) for d in os.listdir(pose_path)]
            all_poses = natsort.natsorted(all_poses)

            depth_path = osp.join(f, 'depth')
            all_depths = [osp.join(depth_path, d) for d in os.listdir(depth_path)]
            all_depths = natsort.natsorted(all_depths)

            color_path = osp.join(f, 'color')
            all_colors = [osp.join(color_path, d) for d in os.listdir(color_path)]
            all_colors = natsort.natsorted(all_colors)

            intrinsic_path = osp.join(f, 'intrinsic/intrinsic_color.txt')

            self.colors.append(all_colors)
            self.depths.append(all_depths)
            self.poses.append(all_poses)
            self.intrinsics.append(intrinsic_path)

            self.n_samples += len(all_colors)
            self.counting.append(self.n_samples)

    def compute_indices(self, index):
        current_count = 0
        for i, c in enumerate(self.counting):
            scene_idx = i
            if index < c:
                break
            current_count = c

        sample_idx = index - current_count

        return scene_idx, sample_idx

    def compute_neighbor_indices(self, index, n_total_samples):
        if index == 0:
            output_list = [1, 2, 3, 4]
        elif index == 1:
            output_list = [0, 2, 3, 4]
        elif index == n_total_samples - 1:
            output_list = [index - 1, index - 2, index - 3, index - 4]
        elif index == n_total_samples - 2:
            output_list = [index - 1, index - 2, index - 3, index + 1]
        else:
            output_list = [index - 1, index - 2, index + 1, index + 2]

        if self.train:
            np.random.shuffle(output_list)
        return output_list

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        scene_idx, sample_idx = self.compute_indices(index)
        neighbor_indices = self.compute_neighbor_indices(sample_idx, len(self.colors[scene_idx]))

        rgb_paths, pose_paths, depth_paths, intrinsic_path = self.get_path_from_idx(scene_idx, sample_idx, neighbor_indices)
        colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic = self.load_data(rgb_paths, pose_paths, depth_paths, intrinsic_path)
        colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic = self.to_tensor(
            colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic)

        if self.image_size is not None and colors.shape[-2:] != self.image_size:
            scale = self.image_size[0] / colors.shape[-2]
            intrinsic[0:2, 0:3] *= scale
            inv_intrinsic = torch.inverse(intrinsic)
            colors = functional.interpolate(colors, size=self.image_size, mode="area")

        if self.train and np.random.randint(0, 25) <= 1:
            colors = self.colorjitter.apply_color_jitter(colors)

        depths = functional.interpolate(depths, size=colors.shape[-2:], mode="nearest")
        depths = [depths[i] for i in range(depths.shape[0])]

        colors = [colors[i] for i in range(colors.shape[0])]
        cameras = [{'P': extrinsics[i], 'Pinv': inv_extrinsics[i], 'K': intrinsic, 'Kinv': inv_intrinsic} for i in range(len(colors))]

        if self.train and self.prepare_for_training_depth:
            patch_size = 196
            height, width = colors[0].shape[-2:]

            for k in range(len(colors)):
                flip_index = np.random.randint(0, 2)
                y = np.random.randint(0, height - 1 - patch_size)
                x = np.random.randint(0, width - 1 - patch_size)

                colors[k] = self.augment(colors[k], flip_index, x, y, patch_size)
                depths[k] = self.augment(depths[k], flip_index, x, y, patch_size)

        augmented_masks = []
        height, width = depths[0].shape[-2:]
        for _ in depths:
            augmented_mask = self.mask_generator.sample(height, width, 1)
            augmented_masks.append(torch.tensor(augmented_mask.reshape((1, height, width))))
        return {
            "idx": scene_idx, "path": self.colors[scene_idx][sample_idx], "img_id": sample_idx, "images": colors,
            "cameras": cameras, "depths": depths, "augmented_mask": augmented_masks}

    def get_path_from_idx(self, scene_idx, sample_idx, neighbor_indices):
        rgb_paths = [self.colors[scene_idx][sample_idx], self.colors[scene_idx][neighbor_indices[0]], self.colors[scene_idx][neighbor_indices[1]],
                     self.colors[scene_idx][neighbor_indices[2]], self.colors[scene_idx][neighbor_indices[3]]]

        depth_paths = [self.depths[scene_idx][sample_idx], self.depths[scene_idx][neighbor_indices[0]], self.depths[scene_idx][neighbor_indices[1]],
                       self.depths[scene_idx][neighbor_indices[2]], self.depths[scene_idx][neighbor_indices[3]]]

        pose_paths = [self.poses[scene_idx][sample_idx], self.poses[scene_idx][neighbor_indices[0]], self.poses[scene_idx][neighbor_indices[1]],
                      self.poses[scene_idx][neighbor_indices[2]], self.poses[scene_idx][neighbor_indices[3]]]

        intrinsic_path = self.intrinsics[scene_idx]
                           
        return rgb_paths, pose_paths, depth_paths, intrinsic_path

    @staticmethod
    def load_data(rgb_paths, pose_paths, depth_paths, intrinsic_path):
        colors = [cv2.imread(d) for d in rgb_paths]
        depths = [cv2.imread(d, cv2.IMREAD_UNCHANGED) / 1000.0 for d in depth_paths]

        inv_extrinsics = [np.loadtxt(d) for d in pose_paths]
        extrinsics = [np.linalg.inv(ie) for ie in inv_extrinsics]

        intrinsic = np.loadtxt(intrinsic_path)
        inv_intrinsic = np.linalg.inv(intrinsic)

        return colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic

    def to_tensor(self, colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic):
        colors = [self.image_to_tensor(c).float() for c in colors]
        depths = [torch.tensor(d).float() for d in depths]

        inv_extrinsics = [torch.tensor(ie).float() for ie in inv_extrinsics]
        extrinsics = [torch.tensor(e).float() for e in extrinsics]

        intrinsic = torch.tensor(intrinsic).float()
        inv_intrinsic = torch.tensor(inv_intrinsic).float()

        colors = torch.stack(colors, dim=0)
        depths = torch.stack(depths, dim=0).unsqueeze(1)

        return colors, depths, extrinsics, intrinsic, inv_extrinsics, inv_intrinsic

    def augment(self, inputs, flip_index, x, y, patch_size):
        inputs = inputs[:, y:y + patch_size, x:x + patch_size]

        if flip_index > 0:
            inputs = torch.flip(inputs, dims=[-2])
        return inputs

    def totrain(self, epoch=0):
        self.train = True
        self.stage = "train"
        self.scan_all_data(self.train_path)

    def toval(self, epoch=0):
        self.train = False
        self.stage = "val"
        self.scan_all_data(self.val_path)

    def totest(self, epoch=0):
        self.train = False
        self.stage = "test"
        self.scan_all_data(self.val_path)
        
