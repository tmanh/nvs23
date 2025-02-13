import os

def get_dataset(opt):
    print(f"Loading dataset {opt.dataset} ...")
    if opt.dataset == 'dtu':
        opt.path = os.path.join(opt.dataset_path, "DTU/")
        opt.camera_path = os.path.join(opt.dataset_path, "camera.npy")
        opt.depth_path = os.path.join(opt.dataset_path, "Depths_2/")
        opt.list_prefix = "new_"
        opt.image_size = opt.H, opt.W
        opt.scale_focal = False
        opt.max_imgs = 100000
        opt.z_near = 0.1
        opt.z_far = 20.0
        opt.skip_step = None
        from data.dtu import DTU_Dataset
        return DTU_Dataset
    elif opt.dataset == 'shapenet':
        opt.path = os.path.join(opt.dataset_path, "NMR_Dataset")
        opt.list_prefix = "softras_"
        opt.scale_focal = True
        opt.max_imgs = 100000
        opt.z_near = 1.2
        opt.z_far = 4.0
        opt.min_z = 1.2
        opt.max_z = 4.0
        opt.image_size = opt.H, opt.W
        from data.shapenet import ShapeNet
        return ShapeNet
    elif opt.dataset == 'scannet':
        opt.path = opt.dataset_path
        opt.scale_focal = True
        opt.max_imgs = 100000
        opt.z_near = 0.1
        opt.z_far = 10.0
        opt.min_z = 0.1
        opt.max_z = 10.0
        opt.image_size = opt.H, opt.W
        from data.scannet import ScannetDataset
        return ScannetDataset
    elif opt.dataset == 'folder':
        opt.path = opt.dataset_path
        opt.scale_focal = True
        opt.max_imgs = 100000
        opt.z_near = 0.1
        opt.z_far = 10.0
        opt.min_z = 0.1
        opt.max_z = 10.0
        opt.image_size = opt.H, opt.W
        from data.folder_dataset import ImageNetDataset
        return ImageNetDataset
    else:
        raise NotImplementedError("no such dataset")
