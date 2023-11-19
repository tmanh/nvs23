import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class multi_view_depth_loss(nn.Module):

    def __init__(self, opt):

        super().__init__()

        self.opt = opt
        self.loss = nn.L1Loss()
        
    def single_loss(self, src_depth, tar_depth, K, src_RTs, tar_RTs, height, width):
        """
        src_depth: bs x H x W
        tar_depth: bs x H x W
        K: bs x 4 x 4
        src_RTs: bs x 4 x 4
        tar_RTs: bs x 4 x 4
        """

        batch = src_depth.shape[0]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_depth.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_depth.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x), torch.ones_like(x)))  # [4, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 4, H*W]
        xyz = xyz * src_depth.view(src_depth.shape[0], 1, -1)
        xyz[:, 3] = 1.0

        word_xyz = torch.bmm(torch.inverse(K), xyz)
        word_xyz = torch.bmm(torch.inverse(src_RTs), word_xyz)
        project_xyz = torch.bmm(tar_RTs, word_xyz)
        project_xyz = torch.bmm(K, project_xyz)
        project_xyz = project_xyz / project_xyz[:, 3:4]
        projected_depth = project_xyz[:, 2:3]
        proj_x_normalized = project_xyz[:, 0, ] / ((width - 1) / 2) - 1
        proj_y_normalized = project_xyz[:, 1, ]/ ((height - 1) / 2) - 1
        with torch.no_grad():
            valid_x = torch.bitwise_and(proj_x_normalized >= -1.0, proj_x_normalized <= 1.0)
            valid_y = torch.bitwise_and(proj_y_normalized >= -1.0, proj_y_normalized <= 1.0)
            valid_mask = torch.bitwise_and(valid_x, valid_y)
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]

        warped_depths = F.grid_sample(tar_depth, proj_xy.view(-1, height, width, 2), mode='bilinear', align_corners=False, padding_mode='zeros')
        warped_depths = warped_depths.view(warped_depths.shape[0], 1, -1)
        return self.loss(warped_depths[valid_mask], projected_depth[valid_mask])

    def forward(self, pred_pts, K, input_RTs):
        """
        pred_pts: predicted depth of points with shape bs x num_inputs x H x W
        K: bs x  4 x 4
        input_RTs: bs x num_inputs x 4 x 4
        """
        height, width = pred_pts.shape[-2], pred_pts.shape[-1]
        num_inputs = pred_pts.shape[1]
        return sum(self.single_loss(pred_pts[:, i], pred_pts[:, j], K, input_RTs[:, i], input_RTs[:, j], height, width) for i, j in itertools.product(range(num_inputs), range(num_inputs)))


class TextureConsistencyLoss(nn.Module):
    def __init__(self, opt):

        super().__init__()

        self.opt = opt
        self.loss = nn.L1Loss()
        
    def single_loss(self, src_depths, src_colors, dst_colors, K, src_RTs, dst_RTs, height, width):
        batch = src_depths.shape[0]
        with torch.no_grad():
            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_depths.device),
                                torch.arange(0, width, dtype=torch.float32, device=src_depths.device)], indexing='ij')
            y, x = y.contiguous().view(-1), x.contiguous().view(-1)
            xyz = torch.stack((x, y, torch.ones_like(x), torch.ones_like(x)))  # [4, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 4, H*W]
        xyz = xyz * src_depths.view(src_depths.shape[0], 1, -1)
        xyz[:, 3, :] = 1

        world_xyz = torch.bmm(torch.inverse(torch.bmm(K, src_RTs)), xyz)
        project_xyz = torch.bmm(torch.bmm(K, dst_RTs), world_xyz)

        proj_x_normalized = project_xyz[:, 0] / (project_xyz[:, 2] + 1e-7) / ((width - 1) / 2) - 1
        proj_y_normalized = project_xyz[:, 1] / (project_xyz[:, 2] + 1e-7) / ((height - 1) / 2) - 1

        with torch.no_grad():
            valid_x = torch.bitwise_and(proj_x_normalized >= -1.0, proj_x_normalized <= 1.0)
            valid_y = torch.bitwise_and(proj_y_normalized >= -1.0, proj_y_normalized <= 1.0)
            valid_mask = torch.bitwise_and(valid_x, valid_y)
        
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]

        warped_colors = F.grid_sample(dst_colors, proj_xy.view(-1, height, width, 2), mode='bilinear', align_corners=False, padding_mode='zeros')
        valid_mask = valid_mask.view(-1, 1, height, width).repeat(1, 3, 1, 1)

        return self.loss(warped_colors[valid_mask], src_colors[valid_mask])

    def forward(self, pred_pts, src_colors, dst_colors, K, input_RTs, output_RTs):
        height, width = pred_pts.shape[-2], pred_pts.shape[-1]
        out_height, out_width = dst_colors.shape[-2], dst_colors.shape[-1]
        n_samples, n_views = input_RTs.shape[:2]

        pred_pts = pred_pts.view(n_samples * n_views, height, width)
        src_colors = src_colors.view(n_samples * n_views, 3, height, width)

        dst_colors = dst_colors.view(n_samples, 3, out_height, out_width)
        scaled_dst_colors = F.interpolate(dst_colors, size=(height, width), mode='bilinear', align_corners=False).view(n_samples, 1, 3, height, width)
        scaled_dst_colors = scaled_dst_colors.repeat(1, n_views, 1, 1, 1).view(n_samples * n_views, 3, height, width)

        input_RTs = input_RTs.view(n_samples * n_views, 4, 4)
        output_RTs = output_RTs.repeat(1, n_views, 1, 1).view(n_samples * n_views, 4, 4)
        K = K.view(n_samples, 1, 4, 4).repeat(1, n_views, 1, 1).view(n_samples * n_views, 4, 4)

        return self.single_loss(pred_pts, src_colors, scaled_dst_colors, K, input_RTs, output_RTs, height, width)
