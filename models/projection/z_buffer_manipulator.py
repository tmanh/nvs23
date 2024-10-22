import torch
import torch.nn as nn

from pytorch3d import __version__ as p3dv

from models.projection.z_buffer_layers import RasterizePointsXYsBlending


EPS = 1e-2


def get_pixel_grids(height, width, device):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0, width - 1, width, device=device).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height, device=device).view(height, 1).expand(height, width)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width, device=device)
        indices_grid = torch.stack(
            [
                x_coordinates,
                y_coordinates,
                ones,
                torch.ones(height * width, device=device)
            ], dim=0
        )
    
    return indices_grid


class Screen_PtsManipulator(nn.Module):
    def __init__(self, W, H, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = RasterizePointsXYsBlending(
            radius=opt.model.radius,
            points_per_pixel=opt.model.pp_pixel,
            opts=opt,
        )
        self.H = H
        self.W = W

    def view_to_world_coord(self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, H, W):
        xyzs = get_pixel_grids(height=H, width=W, device=pts3D.device).view(1,  4, -1)

        # PERFORM PROJECTION
        # Project the world points into the new view
        if len(pts3D.shape) > 3:
            pts3D = pts3D.contiguous().view(pts3D.shape[0], 1, -1)

        projected_coors = xyzs * pts3D
        projected_coors[:, -1, :] = 1

        cam1_X = K_inv.bmm(projected_coors)
        return RTinv_cam1.bmm(cam1_X)

    def world_to_view_screen(self, pts3D, K, K_inv, RT_cam2, RTinv_cam2):
        wrld_X = RT_cam2.bmm(pts3D)
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        zs = xy_proj[:, 2:3, :]
        with torch.no_grad():
            mask = (zs.abs() < EPS).float()
        zs = (1 - mask) * zs + mask * EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)
        sampler = (1 - mask) * sampler - 10 * mask
        
        return sampler
        
    def world_to_view(self, pts3D, K, K_inv, RT_cam2, RTinv_cam2):
        sampler = self.world_to_view_screen(pts3D, K, K_inv, RT_cam2, RTinv_cam2)

        # NOTE: after 0.4.0, pytorch3d changed the NDC coordinate system.
        # They added the aspect ratio to the NDC coordinate system.
        # So, the [-1, 1] x [-1, 1] is changed to [-1, 1] x [-s, s]
        if p3dv <= '0.4.0':
            sampler[:, 0] = (sampler[:, 0] / (self.W - 1) * 2.0 - 1)
            sampler[:, 1] = (sampler[:, 1] / (self.H - 1) * 2.0 - 1)
        else:
            min_size = min(self.W, self.H)
            sampler[:, 0] = (sampler[:, 0] / (min_size - 1) * 2.0 - (self.W / min_size))
            sampler[:, 1] = (sampler[:, 1] / (min_size - 1) * 2.0 - (self.H / min_size))

        return sampler
