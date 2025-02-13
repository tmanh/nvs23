import os
import torch
import torch.nn.functional as F

import glob
import imageio
import numpy as np

from .util import get_image_to_tensor_balanced, load_pfm


class DTU_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val):
        super().__init__()
        path = file_list
        self.file_list = file_list
        self.n_samples = 3
        self.base_path = os.path.join(path, 'DTU/Rectified')
        self.image_to_tensor = get_image_to_tensor_balanced()

        self.scale_factor = 100
        self.camera_path = os.path.join(path, 'camera.npy')
        self.depth_path = os.path.join(path, 'Depths_2')
        self.camera = np.load(self.camera_path, allow_pickle=True)

        self.all_objs = os.listdir(self.base_path)
        
    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        scan_index = self.all_objs[index]

        ## read RGB image
        root_dir, rgb_paths, depth_paths = self.get_path_from(scan_index)

        sample_num = self.n_samples
        sel_indices = torch.randperm(len(rgb_paths))[:sample_num]

        rgb_paths = [rgb_paths[i] for i in sel_indices]

        all_imgs = []
        all_poses = []
        all_raw_depths = []
        raw_depth_max = 0
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx, rgb_path in enumerate(rgb_paths):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            raw_depth, _ = load_pfm(depth_paths[i])
            raw_depth_tensor = torch.tensor(raw_depth.copy())

            all_raw_depths.append(raw_depth_tensor)
            if raw_depth_tensor.max() > raw_depth_max:
                raw_depth_max = raw_depth_tensor.max()

            pose = self.camera[i][0]
            pose[:3, 3] = pose[:3, 3] / self.scale_factor
            K = self.camera[i][1]

            pose =  torch.tensor(pose, dtype=torch.float32)

            fx += K[0, 0]
            fy += K[1, 1]
            cx += K[0, 2]
            cy += K[1, 2]
            img_tensor = self.image_to_tensor(img)
            all_imgs += [img_tensor]
            all_poses += [pose]

        all_raw_depths = [depth / self.scale_factor for depth in all_raw_depths]
        
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
        all_poses = torch.inverse(all_poses)

        return all_imgs[:1], all_imgs[1:], all_raw_depths[:1], all_raw_depths[1:], K, all_poses[:1], all_poses[1:]

    def stack_depth_tensors(self, all_imgs, all_raw_depths):
        all_raw_depths = torch.stack(all_raw_depths, dim=0)
        return F.interpolate(all_raw_depths.unsqueeze(0), size=all_imgs.shape[-2:], mode="nearest")[0].unsqueeze(1)

    def get_path_from(self, scan_index):
        root_dir = os.path.join(self.base_path, scan_index)

        rgb_paths = os.path.join(root_dir, 'image', "*")
        rgb_paths = [
            x
            for x in glob.glob(rgb_paths)
            if x.endswith(".jpg") or x.endswith(".png")
        ]
        
        depth_sen_path = os.path.join(self.depth_path, scan_index, 'depth_map_*.pfm')
        depth_sens = glob.glob(depth_sen_path)
        depth_sens = sorted(depth_sens)

        rgb_paths = sorted(rgb_paths)
        return root_dir, rgb_paths, depth_sens
 