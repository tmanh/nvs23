import os
import torch
import torch.nn.functional as F

import glob
import imageio
import numpy as np

from .util import get_image_to_tensor_balanced, load_pfm


class DTU_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
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

        all_imgs = []
        all_poses = []
        all_raw_depths = []
        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0

        for idx in sel_indices:
            img = imageio.imread(rgb_paths[idx])
            raw_depth, _ = load_pfm(depth_paths[idx])
            raw_depth_tensor = torch.tensor(raw_depth.copy())

            pose = np.eye(4, 4)
            pose[:3, :3] = self.camera[idx][0][:3, :3]
            pose[:3, 3] = self.camera[idx][0][:3, 3] / self.scale_factor
            pose = torch.tensor(pose, dtype=torch.float32)
            
            K = self.camera[idx][1]
            fx += K[0, 0] / 2
            fy += K[1, 1] / 2
            cx += K[0, 2] / 2
            cy += K[1, 2] / 2
            
            all_imgs.append(self.image_to_tensor(img))
            all_poses.append(pose)
            all_raw_depths.append(raw_depth_tensor)

        all_raw_depths = [depth / self.scale_factor for depth in all_raw_depths]
        
        fx /= len(sel_indices)
        fy /= len(sel_indices)
        cx /= len(sel_indices)
        cy /= len(sel_indices)

        all_imgs = torch.stack(all_imgs)
        all_raw_depths = torch.stack(all_raw_depths, dim=0).unsqueeze(1)

        K = np.eye(4, 4)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K = torch.tensor(K, dtype=torch.float32)

        all_poses = torch.stack(all_poses, dim=0)
        all_poses = torch.inverse(all_poses)

        H, W = all_imgs.shape[-2:]
        all_imgs = F.interpolate(all_imgs, size=(H // 2, W // 2), align_corners=False, mode='bilinear', antialias=True)
        all_raw_depths = F.interpolate(all_raw_depths, size=(H // 2, W // 2), mode='nearest')

        return all_imgs[:1], all_imgs[1:], all_raw_depths[:1], all_raw_depths[1:], K, all_poses[:1], all_poses[1:]

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
 