from torch import nn

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points


# torch.manual_seed(42)


class RasterizePointsXYsBlending(nn.Module):
    """
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(self, radius=1.5, size=256, points_per_pixel=8, opts=None):
        super().__init__()

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.opts = opts

    def forward(self, pts3D, src, depth=False):
        if isinstance(pts3D, list):
            # if the pts3d has different point number for each point cloud
            bs = len(src)
            image_size = self.size 
            for i in range(len(pts3D)):
                pts3D[i][:, 1] = - pts3D[i][:, 1]
                pts3D[i][:, 0] = - pts3D[i][:, 0] 

            if len(image_size) > 1:
                radius = float(self.radius) / float(image_size[0]) * 2.0
            else:
                radius = float(self.radius) / float(image_size) * 2.0

            src = [src[i].permute(1,0) for i in range(len(src))]
            pts3D = Pointclouds(points=pts3D, features=src)
        else:
            bs = src.shape[0]
            if len(src.shape) > 3:
                bs, c, w, _ = src.shape
                image_size = w

                pts3D = pts3D.permute(0, 2, 1)
                src = src.unsqueeze(2).repeat(1, 1, w, 1, 1).view(bs, c, -1)
            else:
                bs = src.shape[0]
                image_size = self.size

            # Make sure these have been arranged in the same way
            assert pts3D.shape[2] == 3
            assert pts3D.shape[1] == src.shape[2]  

            pts3D[:, :, :2] = -pts3D[:, :, :2]

            # Add on the default feature to the end of the src
            if len(image_size) > 1:
                radius = float(self.radius) / float(image_size[0]) * 2.0
            else:
                radius = float(self.radius) / float(image_size) * 2.0

            pts3D = Pointclouds(points=pts3D, features=src.permute(0, 2, 1))

        # NOTE:
        # self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)
        # self._num_points_per_cloud = self._P * torch.ones((self._N,), dtype=torch.int64, device=self.device)
        # Modified this in the pytorch code
        points_idx, z_buf, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        )

        dist = dist / pow(radius, self.opts.rad_pow)

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .permute(0, 3, 1, 2)
        )

        if self.opts.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.opts.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

        if depth is False:
            return transformed_src_alphas

        w_normed = alphas * (points_idx.permute(0,3,1,2) >= 0).float()
        w_normed = w_normed / w_normed.sum(dim=1, keepdim=True).clamp(min=1e-9)
        z_weighted = z_buf.permute(0,3,1,2).contiguous() * w_normed.contiguous()
        z_weighted = z_weighted.sum(dim=1, keepdim=True)
        
        return transformed_src_alphas, z_weighted
