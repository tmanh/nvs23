import os
import torch
import torch.nn.functional as F

import glob
import imageio
import numpy as np

from .util import get_image_to_tensor_balanced, get_mask_to_tensor


class ShapeNet(torch.utils.data.Dataset):
    def __init__(
        self,
        stage="train",
        opts=None
    ):
        super().__init__()

        path = opts.path
        self.base_path = path
        list_prefix = opts.list_prefix
        self.input_view_num = opts.input_view_num
        self.num_views = opts.num_views
        sub_format = 'shapenet'

        cats = [x for x in glob.glob(os.path.join(path, "*")) if os.path.isdir(x)]

        self.train_path = [os.path.join(x, f"{list_prefix}train.lst") for x in cats]
        self.val_path = [os.path.join(x, f"{list_prefix}val.lst") for x in cats]
        self.test_path = [os.path.join(x, f"{list_prefix}test.lst") for x in cats]
        if stage == "train":
            file_lists = [os.path.join(x, f"{list_prefix}train.lst") for x in cats]
        elif stage == "val":
            file_lists = [os.path.join(x, f"{list_prefix}val.lst") for x in cats]
        elif stage == "test":
            file_lists = [os.path.join(x, f"{list_prefix}test.lst") for x in cats]

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = stage

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading DVR dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )
        self.image_size = opts.image_size
        self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
        )

        self.max_imgs = opts.max_imgs

        self.z_near = opts.z_near
        self.z_far = opts.z_far
        self.lindisp = False

        if self.stage == "test":
            self.source_lut = None
            if opts.test_view.endswith(".txt"):
                print("Using views from list", opts.test_view)
                with open(opts.test_view, "r") as f:
                    tmp = [x.strip().split() for x in f.readlines()]
                source_lut = {f"{x[0]}/{x[1]}": torch.tensor(list(map(int, x[2:])), dtype=torch.long) for x in tmp}

                self.source_lut = source_lut
            else:
                self.test_view = list(map(int, opts.test_view.split()))

    def __len__(self):
        return len(self.all_objs)
    
    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]
        rgb_paths, mask_paths, cam_path = self.get_path_from(root_dir)
        all_cam = np.load(cam_path)

        if self.stage == "test":
            ## For evaluation, the input view is predefined.
            query = f"{cat}/" + root_dir.split("/")[-1]
            if self.source_lut is not None:
                test_view = [self.source_lut[query].item()]
            else:
                test_view = self.test_view

            all_indices = list(np.arange(len(rgb_paths)))
            sel_indices = list(test_view)
            all_indices = [x for x in all_indices if x not in sel_indices]
            sel_indices += all_indices
        else:
            ## We randomly select num_views images, whhile the first input_view_num image for input.
            sample_num = min(self.num_views, self.max_imgs)
            sel_indices = torch.randperm(len(rgb_paths))[:sample_num]
        
        rgb_paths = [rgb_paths[i] for i in sel_indices]
        mask_paths = [mask_paths[i] for i in sel_indices]
        all_masks = []
        all_bboxes = []
        all_imgs = []
        all_poses = []
        all_poses_inverse = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

            if isinstance(i, torch.Tensor):
                i = i.item()

            K = all_cam[f"camera_mat_{str(i)}"]
            K[0, 0] = K[0, 0] * 32
            K[1, 1] = K[1, 1] * 32
            K[0, 2] = 32
            K[1, 2] = 32

            fx = K[0, 0]
            fy = K[1, 1]
            cx = 32
            cy = 32

            pose = all_cam[f'world_mat_{str(i)}']  # np.linalg.inv(all_cam[f'world_mat_{str(i)}'])
            pose = self._coord_trans_world @ torch.tensor(pose, dtype=torch.float32)

            img_tensor = self.image_to_tensor(img)
            pose_inverse = torch.inverse(pose)
            all_imgs += [img_tensor]
            all_poses += [pose]
            all_poses_inverse += [pose_inverse]

            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError("ERROR: Bad image at", rgb_path, "please investigate!")
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

        if mask_paths is not None:
            all_bboxes = torch.stack(all_bboxes)
        
        all_masks = torch.stack(all_masks, 0) if all_masks else None
        focal = torch.tensor((fx, fy), dtype=torch.float32)
        c = torch.tensor((cx, cy), dtype=torch.float32)
        all_imgs = torch.stack(all_imgs)

        K = torch.tensor(K, dtype=torch.float32)

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            K[0:2, 0:3] *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if mask_paths is not None:
                all_bboxes *= scale
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")
        
        inverse_K = torch.inverse(K)

        all_imgs = [all_imgs[i] for i in range(all_imgs.shape[0])]
        all_bboxes = [all_bboxes[i] for i in range(all_bboxes.shape[0])]
        all_masks = [all_masks[i] for i in range(all_masks.shape[0])]

        cameras = [{'focal': focal, 'P': all_poses[i], 'Pinv': all_poses_inverse[i], 'K': K, 'Kinv': inverse_K, 'c': c, 'i': i} for i in range(len(all_imgs))]

        return {"indices": sel_indices, "path": root_dir, "img_id": index, "images": all_imgs, "cameras": cameras, "boxes": all_bboxes}

    def get_path_from(self, root_dir):
        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "image", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))

        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        cam_path = os.path.join(root_dir, "cameras.npz")
        return rgb_paths,mask_paths,cam_path

    def totrain(self, epoch=0):
        self.train = True 
        all_objs = []
        for file_list in self.train_path:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = "train"
    
    def toval(self, epoch=0):
        self.train = False 
        all_objs = []
        for file_list in self.val_path:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = "val"
