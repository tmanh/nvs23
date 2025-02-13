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


class ArkitDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, val=None):
        super().__init__()
        self.file_list = file_list
        self.n_samples = 3
        self.image_to_tensor = get_image_to_tensor_balanced()

        self.scale_factor = 100
        self.all_objs = [d for d in os.listdir(self.file_list) if os.path.isdir(os.path.join(self.file_list, d))]
        
    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        flag = True
        while flag:
            flag = False

            scan_index = self.all_objs[index]
            Ks, Rt, pairs, inames = self.read_cam(scan_index)

            ## read RGB image
            rgb_paths, depth_paths = self.get_path_from(scan_index, inames)

            sample_num = self.n_samples
            tgt_idx = np.random.randint(0, len(rgb_paths))
            if tgt_idx not in pairs or len(pairs[tgt_idx]) <= sample_num - 1:
                index = np.random.randint(0, len(self))
                flag = True
                continue
            src_idx = random.sample(pairs[tgt_idx], k=sample_num - 1)
            sel_indices = [tgt_idx, *src_idx]

            rgb_paths = [rgb_paths[i] for i in sel_indices]
            depth_paths = [depth_paths[i] for i in sel_indices]

            all_imgs = []
            all_poses = []
            all_raw_depths = []
            fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
            for idx, (rgb_path, dep_path) in enumerate(zip(rgb_paths, depth_paths)):
                i = sel_indices[idx]
                img = cv2.imread(rgb_path)
                raw_depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED) / 1000.0

                H, W = img.shape[:2]
                if H != 480 or W != 640:
                    index = np.random.randint(0, len(self))
                    flag = True
                    break
                
                img = cv2.resize(img, dsize=(W // 2, H // 2), interpolation=cv2.INTER_LANCZOS4)
                raw_depth = cv2.resize(raw_depth, dsize=(W // 2, H // 2), interpolation=cv2.INTER_LANCZOS4)

                raw_depth_tensor = torch.tensor(raw_depth)
                all_raw_depths.append(raw_depth_tensor)

                pose = Rt[i]
                pose[:3, 3] = pose[:3, 3]
                K = Ks[i]

                pose =  torch.tensor(pose, dtype=torch.float32)

                fx += K[0, 0] / 2
                fy += K[1, 1] / 2
                cx += K[0, 2] / 2
                cy += K[1, 2] / 2
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

    def read_cam(self, scan_index):
        root_dir = os.path.join(self.file_list, scan_index)
        Ks = np.load(os.path.join(root_dir, 'intrinsics.npy'))
        Ks = self.convert_to_intrinsic_matrices(Ks)

        Rt = np.load(os.path.join(root_dir, 'trajectories.npy'))
        
        pairs = np.load(os.path.join(root_dir, 'pairs.npy'))
        pairs = self.build_camera_pair_dict(pairs)

        inames = np.load(os.path.join(root_dir, 'images.npy'))
        
        return Ks, Rt, pairs, inames

    def build_camera_pair_dict(self, B):
        """
        Builds a dictionary where each camera index is mapped to a list of paired camera indices.

        Parameters:
        - B (numpy array): Shape (N, 3), where each row represents (camera1, camera2, similarity_score).

        Returns:
        - camera_dict (dict): Dictionary where keys are camera indices (int), 
                            and values are lists of paired camera indices.
        """
        camera_dict = defaultdict(list)

        for row in B:
            cam1, cam2, _ = row  # Extract camera indices (ignore similarity score)
            cam1, cam2 = int(cam1), int(cam2)  # Convert indices to integers
            
            # Add pair to dictionary
            camera_dict[cam1].append(cam2)
            camera_dict[cam2].append(cam1)  # Ensure bidirectional linking

        return dict(camera_dict)

    def convert_to_intrinsic_matrices(self, A):
        """
        Converts an array of [width, height, fx, fy, cx, cy] into intrinsic matrices.

        Parameters:
        - A (numpy array): Shape (N, 6), where each row represents (width, height, fx, fy, cx, cy).

        Returns:
        - intrinsic_matrices (numpy array): Shape (N, 3, 3), where each 3x3 matrix is an intrinsic matrix.
        """
        num_rows = A.shape[0]
        intrinsic_matrices = np.zeros((num_rows, 4, 4))

        for i in range(num_rows):
            _, _, fx, fy, cx, cy = A[i]  # Extract values from each row
            intrinsic_matrices[i] = np.array(
                [
                    [fx, 0, cx, 0], 
                    [0, fy, cy, 0], 
                    [0,  0,  1, 0],
                    [0,  0,  0, 1]
                ]
            )

        return intrinsic_matrices
        

    def get_path_from(self, scan_index, inames):
        root_dir = os.path.join(self.file_list, scan_index)

        rgb_paths = [
            os.path.join(root_dir, 'vga_wide', i.replace('png', 'jpg')) for i in inames
        ]
        
        depth_sens = [
            os.path.join(root_dir, 'lowres_depth', i) for i in inames
        ]

        return rgb_paths, depth_sens
 