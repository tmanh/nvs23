import argparse
import datetime
import os
import time

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="generalizable multi view synthesis")
        self.add_data_params()
        self.add_train_params()
        self.add_model_params()
        self.add_fwd_model_params()
        self.add_pytorch3d_splatter_params()

    def add_pytorch3d_splatter_params(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument("--accumulation",type=str, default="alphacomposite", choices=("wsum", "wsumnorm", "alphacomposite"), help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)")
        model_params.add_argument("--rad_pow", type=int, default=2, help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
        model_params.add_argument("--pp_pixel", type=int, default=24, help='the number of points to conisder in the z-buffer.')
        model_params.add_argument("--radius", type=float, default=1.5, help="Radius of points to project.")

    def add_fwd_model_params(self):
        model_params = self.parser.add_argument_group("model")

        # NORM AND REFINE
        model_params.add_argument("--norm_G", type=str, default="sync:spectral_batch",
                                  choices=['batch', 'spectral_batch', 'spectral_batchstanding', 'spectral_instance',
                                           'sync:batch', 'sync:spectral_batch', 'sync:spectral_batchstanding', 'sync:spectral_instance'],
                                  help="type of normalization layers")
        model_params.add_argument("--refine_model_type", type=str, default="resnet_256W8customup",
                                  choices=['resnet_256W8', 'resnet_256W8UpDown', 'resnet_256W8UpDown3', 'resnet_256W8UpDownDV',
                                           'resnet_256W8UpDown64', 'resnet_256W8UpDownRGB', 'resnet_256W8custom', 'resnet_256W8customup',
                                           'resnet_64W8custom', 'unet', ],
                                  help="Model to be used for the refinement network and the feature encoder.")
        model_params.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer.")
        model_params.add_argument("--decoder_norm", type=str,default="instance", help="normalization type of decoder.")

        # FUSION
        model_params.add_argument("--atten_k_dim", type=int, default=16)
        model_params.add_argument("--atten_v_dim", type=int, default=64)
        model_params.add_argument("--atten_n_head", type=int, default=4)

        # DEPTH REGRESSION
        model_params.add_argument("--depth_regressor",type=str,default="unet",help="depth regression network.")
        model_params.add_argument("--regressor_model", type=str, default="Unet", help="feature regression network.")

    def add_model_params(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument("--model_type", type=str, default="multi_z_transformer", help='model to be used.')
        model_params.add_argument("--down_sample", default=False, action="store_true", help="if downsample the input  image")

        model_params.add_argument("--use_inverse_depth", action="store_true", default=False, help='if true the depth is sampled as a long tail distribution, else the depth is sampled uniformly. Set to true if the dataset has points that are very far away (e.g. a dataset with landscape images, such as KITTI).')
        model_params.add_argument("--depth_com", action="store_true", default=False, help="whether use depth completion module.")
        model_params.add_argument("--inverse_depth", action="store_true", default=False)
        model_params.add_argument("--mvs_depth", default=False, action="store_true", help="use mvs to predict depths.")
        model_params.add_argument("--learnable_mvs", default=False, action="store_true", help="whether mvs is learnable or not.")
        model_params.add_argument("--pretrained_MVS", action="store_true", default=False)
        model_params.add_argument("--device", type=str, default='cuda')

    def add_data_params(self):
        dataset_params = self.parser.add_argument_group("data")
        dataset_params.add_argument("--dataset", type=str, default="dtu")
        dataset_params.add_argument("--dataset_path", type=str, default="./data")
        dataset_params.add_argument("--scale_factor", type=float, default=100.0, help="the factor to scale the xyz coordinate")
        dataset_params.add_argument("--num_views", type=int, default=4, help='Number of views considered per batch (input images + target images).')
        dataset_params.add_argument("--input_view_num", type=int, default=3, help="Number of views of input images per batch.")
        dataset_params.add_argument("--normalize_image", action="store_true", default=True)
        dataset_params.add_argument("--test_view", type=str, default="22 25 28")
        dataset_params.add_argument("--min_z", type=float, default=4.25)
        dataset_params.add_argument("--max_z", type=float, default=10.0)
        dataset_params.add_argument("--normalize_depth", action="store_true", default=True)
        dataset_params.add_argument("--W", type=int, default=200)
        dataset_params.add_argument("--H", type=int, default=150)
        dataset_params.add_argument("--DW", type=int, default=-1)
        dataset_params.add_argument("--DH", type=int, default=-1)


    def add_train_params(self):
        training = self.parser.add_argument_group("training")
        training.add_argument("--debug_path", type=str, default="./debug")
        training.add_argument("--num_workers", type=int, default=12)
        training.add_argument("--start-epoch", type=int, default=0)
        training.add_argument("--num_accumulations", type=int, default=1)
        training.add_argument("--lr", type=float, default=0.0001 * 0.5)
        training.add_argument("--lr_g", type=float, default=0.0001 * 0.5)
        training.add_argument("--beta1", type=float, default=0.9)
        training.add_argument("--beta2", type=float, default=0.999)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--init", type=str, default="")

        training.add_argument("--consis_loss", action="store_true", default=False)
        training.add_argument("--depth_lr_scaling", type=float,default=1.0)
        
        training.add_argument("--anneal_start", type=int, default=10000)
        training.add_argument("--anneal_t", type=int, default=100)
        training.add_argument("--anneal_factor",type=float,default=0.8)

        training.add_argument("--val_period", type=int, default=5)

        training.add_argument("--train_depth", action="store_true", default=False)
        training.add_argument("--train_depth_only", action="store_true", default=False)
        training.add_argument("--train_sr", action="store_true", default=False)

        training.add_argument("--losses", type=str, nargs="+", default=['5.0_l2','1.0_content'])

        training.add_argument("--gt_depth_loss_cal", type=str, default='normal')
        training.add_argument("--gt_depth_loss_weight", type=float, default=3.0)

        training.add_argument("--resume", action="store_true", default=False)
        training.add_argument("--log_dir", type=str, default="./checkpoint/ow045820/logging/viewsynthesis3d/%s/")

        training.add_argument("--batch_size", type=int, default=16)
        training.add_argument("--continue_epoch", type=int, default=0)
        training.add_argument("--max_epoch", type=int, default=40000)
        training.add_argument("--model_epoch_path", type=str, default="models/lr%0.5f_bs%d_model%s")
        training.add_argument("--run_dir", type=str, default="/runs/lr%0.5f_bs%d_model%s/")
        training.add_argument("--gpu_ids", type=str, default="0")

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)
