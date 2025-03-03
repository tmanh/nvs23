import torch
import numpy as np

from liegroups.torch import SE3


def getProjectionMatrix(znear, zfar, fx, fy, cx, cy):
    # TODO: remove hard-coded image size
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2 * fx / 256
    P[1, 1] = 2 * fy / 256
    P[0, 2] = 2 * (cx / 256) - 1
    P[1, 2] = 2 * (cy / 256) - 1
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[3, 2] = z_sign
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class CustomCamera:
    def __init__(self, cam_params=None, index=None, c2w=None, opt_pose=False):
        # TODO: remove hard-coded image size
        # c2w (pose) should be in NeRF convention.
        # This this the camera class that supports pose optimization.

        self.image_width, self.image_height = 256, 256
        self.fx, self.fy = cam_params["focal_length"]
        self.cx, self.cy = cam_params["principal_point"]
        self.FoVy = 2 * np.arctan(self.image_height / 2 / self.fy)
        self.FoVx = 2 * np.arctan(self.image_width / 2 / self.fx)
        self.R = torch.tensor(cam_params["R"])
        self.T = torch.tensor(cam_params["T"]) 
        self.znear = 0.01
        self.zfar = 100
        self.opt_pose = opt_pose
        self.index = index

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, 
                zfar=self.zfar, 
                fx=self.fx, 
                fy=self.fy,
                cx=self.cx, 
                cy=self.cy,
            )
            .transpose(0, 1)
            .cuda()
        )

        if not opt_pose:
            if c2w:
                w2c = torch.from_numpy(c2w)
                w2c[1:3] *= -1  # OpenGL to OpenCV
            else:
                R = self.R.T  # note the transpose here
                T = self.T
                upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
                lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
                w2c = torch.cat([upper, lower], dim=0)

                w2c[:2] *= -1  # PyTorch3D to OpenCV

            self.w2c = w2c
            self.cam_params = torch.zeros(6)
            self.world_view_transform = w2c.transpose(0, 1).cuda()
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        else:
            R = self.R.T  # note the transpose here
            T = self.T
            upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
            lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
            w2c = torch.cat([upper, lower], dim=0)

            w2c[:2] *= -1  # PyTorch3D to OpenCV

            self.w2c = w2c
            self.cam_params = torch.randn(6) * 1e-6
            self.cam_params.requires_grad_()

            self.world_view_transform = w2c.transpose(0, 1).cuda()
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix
            self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def perspective(self):
        P = torch.zeros(4, 4)

        z_sign = -1.0

        P[0, 0] = 2 * self.fx / 256
        P[1, 1] = -2 * self.fy / 256
        P[0, 2] = -(2 * (self.cx / 256) - 1)
        P[1, 2] = -(2 * (self.cy / 256) - 1)
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[3, 2] = z_sign
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P.numpy()

    @property
    def c2w(self):
        if self.opt_pose:
            w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
            w2c[1:3] *= -1  # OpenCV to OpenGL
        else:
            R = self.R.T  # note the transpose here
            T = self.T
            upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
            lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
            w2c = torch.cat([upper, lower], dim=0)
            w2c[:2, :] *= -1  # PyTorch3D to OpenCV
            w2c[1:3, :] *= -1  # OpenCV to OpenGL

        return torch.inverse(w2c).numpy()

    @property
    def focal_length(self):
        return np.array([self.fx, self.fy])

    @property
    def rotation(self):
        w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
        w2c[:2] *= -1
        return w2c[:3, :3].T

    @property
    def translation(self):
        w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
        w2c[:2] *= -1
        return w2c[:3, 3]
    

class MyCustomCamera:
    def __init__(self, K, index=None, c2w=None, opt_pose=False):
        # TODO: remove hard-coded image size
        # c2w (pose) should be in NeRF convention.
        # This the camera class that supports pose optimization.

        self.image_width, self.image_height = 256, 256
        self.fx, self.fy = K[0, 0, 0], K[0, 1, 1]
        self.cx, self.cy = K[0, 0, 2], K[0, 1, 2]
        self.FoVy = 2 * np.arctan(self.image_height / 2 / self.fy)
        self.FoVx = 2 * np.arctan(self.image_width / 2 / self.fx)
        self.R = torch.tensor(cam_params["R"])
        self.T = torch.tensor(cam_params["T"]) 
        self.znear = 0.01
        self.zfar = 100
        self.opt_pose = opt_pose
        self.index = index

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, 
                zfar=self.zfar, 
                fx=self.fx, 
                fy=self.fy,
                cx=self.cx, 
                cy=self.cy,
            )
            .transpose(0, 1)
            .cuda()
        )

        if not opt_pose:
            if c2w:
                w2c = torch.from_numpy(c2w)
                w2c[1:3] *= -1  # OpenGL to OpenCV
            else:
                R = self.R.T  # note the transpose here
                T = self.T
                upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
                lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
                w2c = torch.cat([upper, lower], dim=0)

                w2c[:2] *= -1  # PyTorch3D to OpenCV

            self.w2c = w2c
            self.cam_params = torch.zeros(6)
            self.world_view_transform = w2c.transpose(0, 1).cuda()
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix
            self.camera_center = self.world_view_transform.inverse()[3, :3]
        else:
            R = self.R.T  # note the transpose here
            T = self.T
            upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
            lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
            w2c = torch.cat([upper, lower], dim=0)

            w2c[:2] *= -1  # PyTorch3D to OpenCV

            self.w2c = w2c
            self.cam_params = torch.randn(6) * 1e-6
            self.cam_params.requires_grad_()

            self.world_view_transform = w2c.transpose(0, 1).cuda()
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix
            self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def perspective(self):
        P = torch.zeros(4, 4)

        z_sign = -1.0

        P[0, 0] = 2 * self.fx / 256
        P[1, 1] = -2 * self.fy / 256
        P[0, 2] = -(2 * (self.cx / 256) - 1)
        P[1, 2] = -(2 * (self.cy / 256) - 1)
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[3, 2] = z_sign
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        return P.numpy()

    @property
    def c2w(self):
        if self.opt_pose:
            w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
            w2c[1:3] *= -1  # OpenCV to OpenGL
        else:
            R = self.R.T  # note the transpose here
            T = self.T
            upper = torch.cat([R, T[:, None]], dim=1)  # Upper 3x4 part of the matrix
            lower = torch.tensor([[0, 0, 0, 1]], device=R.device, dtype=R.dtype)  # Last row
            w2c = torch.cat([upper, lower], dim=0)
            w2c[:2, :] *= -1  # PyTorch3D to OpenCV
            w2c[1:3, :] *= -1  # OpenCV to OpenGL

        return torch.inverse(w2c).numpy()

    @property
    def focal_length(self):
        return np.array([self.fx, self.fy])

    @property
    def rotation(self):
        w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
        w2c[:2] *= -1
        return w2c[:3, :3].T

    @property
    def translation(self):
        w2c = self.w2c @ SE3.exp(self.cam_params.detach()).as_matrix()
        w2c[:2] *= -1
        return w2c[:3, 3] 