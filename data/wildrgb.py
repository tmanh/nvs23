import os
import torch
import torch.nn.functional as F

import cv2
import glob
import random
import imageio
import numpy as np
from collections import defaultdict

from .util import get_image_to_tensor_balanced, load_pfm


class WildRGBDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
        super().__init__()
        self.file_list = file_list
        self.n_samples = 3
        self.image_to_tensor = get_image_to_tensor_balanced()

        self.scale_factor = 100
        self.all_objs = [os.path.join(self.file_list, d, 'scenes') for d in os.listdir(self.file_list) if os.path.isdir(os.path.join(self.file_list, d))]
        self.all_scenes = []
        for o in self.all_objs:
            self.all_scenes.extend([os.path.join(o, d) for d in os.listdir(o) if os.path.isdir(os.path.join(o, d))])

        self.H = 256
        self.W = 192

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, index):
        flag = True
        while flag:
            flag = False

            rgb_paths, depth_paths, cam_paths = self.get_path_from(index)

            tgt_idx = np.random.randint(0, len(rgb_paths))
            src_idx = random.sample(list(np.arange(-10, 11, 1)), k=self.n_samples - 1)
            sel_indices = [tgt_idx, *src_idx]

            rgb_paths = [rgb_paths[i] for i in sel_indices]
            depth_paths = [depth_paths[i] for i in sel_indices]
            cam_paths = [cam_paths[i] for i in sel_indices]

            all_imgs = []
            all_poses = []
            all_raw_depths = []
            fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
            for (rgb_path, dep_path, cam_path) in zip(rgb_paths, depth_paths, cam_paths):
                img = cv2.imread(rgb_path)
                raw_depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED) / 1000.0

                K, pose = self.read_cam(cam_path)

                H, W = img.shape[:2]

                img = cv2.resize(img, dsize=(self.W, self.H), interpolation=cv2.INTER_LANCZOS4)
                raw_depth = cv2.resize(raw_depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LANCZOS4)

                raw_depth_tensor = torch.tensor(raw_depth)
                all_raw_depths.append(raw_depth_tensor)

                pose =  torch.tensor(pose, dtype=torch.float32)
                # pose = torch.inverse(pose)

                fx += K[0, 0] * (self.W / W)
                fy += K[1, 1] * (self.H / H)
                cx += K[0, 2] * (self.W / W)
                cy += K[1, 2] * (self.H / H)
                img_tensor = self.image_to_tensor(img)
                all_imgs += [img_tensor]
                all_poses += [pose]

            if flag:
                continue

            fx /= len(rgb_paths)
            fy /= len(rgb_paths)
            cx /= len(rgb_paths)
            cy /= len(rgb_paths)

            all_imgs = torch.stack(all_imgs)

            K = np.zeros((4, 4))
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1.0
            K[3, 3] = 1.0
            K = torch.tensor(K, dtype=torch.float32)

            all_raw_depths = self.stack_depth_tensors(all_imgs, all_raw_depths)
            all_poses = torch.stack(all_poses, dim=0)

            return all_imgs[:1], all_imgs[1:], all_raw_depths[:1], all_raw_depths[1:], K, all_poses[:1], all_poses[1:]

    def stack_depth_tensors(self, all_imgs, all_raw_depths):
        all_raw_depths = torch.stack(all_raw_depths, dim=0)
        return F.interpolate(all_raw_depths.unsqueeze(0), size=all_imgs.shape[-2:], mode="nearest")[0].unsqueeze(1)

    def read_cam(self, cam_path):
        X = np.load(cam_path)

        Ks = X['camera_intrinsics']
        Rt = X['camera_pose']
        
        return Ks, Rt
        

    def get_path_from(self, scan_index):
        root_dir = os.path.join(self.all_scenes[scan_index])

        rgb_paths = [
            os.path.join(root_dir, 'rgb', d) for d in os.listdir(os.path.join(root_dir, 'rgb'))
        ]
        
        depth_paths = [
            os.path.join(root_dir, 'depth', d) for d in os.listdir(os.path.join(root_dir, 'depth'))
        ]

        cam_paths = [
            os.path.join(root_dir, 'metadata', d) for d in os.listdir(os.path.join(root_dir, 'metadata'))
        ]

        rgb_paths.sort()
        depth_paths.sort()
        cam_paths.sort()

        return rgb_paths, depth_paths, cam_paths
 