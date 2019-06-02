# coding=utf-8

import numpy as np
import tensorflow as tf
import torchver.spatial_transformer as spatial_transformer
import torch.nn.functional as F

import torch


class TVNet(object):
    GRAD_IS_ZERO = 1e-12

    def __init__(self):
        pass

    def grey_scale_image(self, x):
        assert len(x.shape) == 4
        assert x.shape[1] == 3, 'number of channels must be 3 (i.e. RGB)'

        kernel = torch.FloatTensor(np.reshape([0.114, 0.587, 0.299],
                                              (1, 3, 1, 1)))
        grey_x = F.conv2d(x, kernel, stride=[1, 1], bias=None, padding=0)
        return torch.floor(grey_x)

    def normalize_images(self, x1, x2):
        min_x1 = max_x1 = x1
        min_x2 = max_x2 = x2
        while len(min_x1.shape) >= 2:
            min_x1, _ = torch.min(min_x1, dim=1)
            max_x1, _ = torch.max(max_x1, dim=1)

            min_x2, _ = torch.min(min_x2, dim=1)
            max_x2, _ = torch.max(max_x2, dim=1)

        min_val = torch.min(min_x1, min_x2)
        max_val = torch.max(max_x1, max_x2)

        den = max_val - min_val

        expand_dims = [-1 if i == 0 else 1 for i in range(len(x1.shape))]
        min_val_ex = torch.reshape(min_val, expand_dims)
        den_ex = torch.reshape(den, expand_dims)

        x1_norm = x1
        x2_norm = x2
        x1_norm[den > 0] = 255. * (x1 - min_val_ex) / den_ex
        x2_norm[den > 0] = 255. * (x2 - min_val_ex) / den_ex

        return x1_norm, x2_norm

    def gaussian_smooth(self, x):
        assert len(x.shape) == 4

        kernel = torch.FloatTensor(np.reshape([
            [0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
            [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
            [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
            [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]
        ], (1, 1, 5, 5)))
        smooth_x = F.conv2d(x, kernel, bias=None, padding=2)

        return smooth_x

    def warp_image(self, x, u, v):
        assert len(x.shape) == 4
        assert len(u.shape) == 3
        assert len(v.shape) == 3
        u = u / x.shape[2] * 2
        v = v / x.shape[1] * 2

        delta = torch.cat((u, v), 1)
        return spatial_transformer.transformer(x, x.shape[-2], x.shape[-1])

    def centered_gradient(self, x, name):
        assert len(x.shape) == 4  # N, C, H, W

        x_kernel = torch.FloatTensor(np.reshape(
            [[-0.5, 0, 0.5]], (1, 1, 1, 3)))
        diff_x = F.conv2d(x, x_kernel, bias=None, padding=(0, 1))

        y_kernel = torch.FloatTensor(np.reshape(
            [[-0.5], [0], [0.5]], (1, 1, 3, 1)))
        diff_y = F.conv2d(x, y_kernel, bias=None, padding=(1, 0))

        # refine the boundary
        first_col = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])[:, :, :, None]
        last_col = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])[:, :, :, None]
        diff_x_valid = diff_x[:, :, :, 1:-1]
        diff_x = torch.cat((first_col, diff_x_valid, last_col), dim=3)

        first_row = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])[:, :, None, :]
        last_row = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])[:, :, None, :]
        diff_y_valid = diff_y[:, :, 1:-1, :]
        diff_y = torch.cat((first_row, diff_y_valid, last_row), dim=2)

        return diff_x, diff_y

    def forward_gradient(self, x, name):
        assert len(x.shape) == 4

        x_kernel = torch.FloatTensor(np.reshape([[-1, 1]], (1, 1, 1, 2)))
        diff_x = F.conv2d(x, x_kernel, bias=None, padding=(0, 1))[:, :, :, :-1]

        y_kernel = torch.FloatTensor(np.reshape([[-1], [1]], (1, 1, 2, 1)))
        diff_y = F.conv2d(x, y_kernel, bias=None, padding=(1, 0))[:, :, :-1, :]

        # refine the boundary
        first_col = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])[:, :, :, None]
        last_col = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])[:, :, :, None]
        diff_x_valid = diff_x[:, :, :, 1:-1]
        diff_x = torch.cat((first_col, diff_x_valid, last_col), dim=3)

        first_row = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])[:, :, None, :]
        last_row = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])[:, :, None, :]
        diff_y_valid = diff_y[:, :, 1:-1, :]
        diff_y = torch.cat((first_row, diff_y_valid, last_row), dim=2)

        return diff_x, diff_y

    def divergence(self, x, y, name):
        assert len(x.shape) == 4

        x_valid = x[:, :, :, :-1]
        first_col = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1,
                                dtype=torch.float32)
        x_pad = torch.cat((first_col, x_valid), dim=3)

        y_valid = x[:, :, :-1, :]
        first_row = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3],
                                dtype=torch.float32)
        y_pad = torch.cat((first_row, y_valid), dim=2)

        x_kernel = torch.FloatTensor(np.reshape([[-1, 1]], (1, 1, 1, 2)))
        diff_x = F.conv2d(x_pad, x_kernel,
                          bias=None, padding=(0, 1))[:, :, :, :-1]

        y_kernel = torch.FloatTensor(np.reshape([[-1], [1]], (1, 1, 2, 1)))
        diff_y = F.conv2d(y_pad, y_kernel,
                          bias=None, padding=(1, 0))[:, :, :-1, :]

        div = diff_x + diff_y
        return div


    def zoom_size(self, height, width, factor):
        new_height = int(float(height) * factor + 0.5)
        new_width = int(float(width) * factor + 0.5)

        return new_height, new_width

    def zoom_image(self, x, new_height, new_width):
        assert len(x.shape) == 4

        # delta = torch.zeros((x.shape[0], 2, new_height * new_width))
        zoomed_x = spatial_transformer.transformer(x, new_height, new_width)
        return zoomed_x.view(x.shape[0], x.shape[1], new_height, new_width)

    def dual_tvl1_optic_flow(self, x1, x2, u1, u2,
                             tau=0.25,  # time step
                             lbda=0.15,  # weight parameter for the data term
                             theta=0.3,  # weight parameter for (u - v)^2
                             warps=5,  # number of warpings per scale
                             max_iterations=5
                             # maximum number of iterations for optimization
                             ):

        l_t = lbda * theta
        taut = tau / theta

        diff2_x, diff2_y = self.centered_gradient(x2, 'x2')

        p11 = p12 = p21 = p22 = torch.zeros_like(x1)

        for warpings in range(warps):
            u1_flat = torch.reshape(u1, (
                x2.shape[0], 1, x2.shape[2] * x2.shape[3]))
            u2_flat = torch.reshape(u2, (
                x2.shape[0], 1, x2.shape[2] * x2.shape[3]))

            x2_warp = self.warp_image(x2, u1_flat, u2_flat)
            x2_warp = torch.reshape(x2_warp, x2.shape)

            diff2_x_warp = self.warp_image(diff2_x, u1_flat, u2_flat)
            diff2_x_warp = torch.reshape(diff2_x_warp, diff2_x.shape)

            diff2_y_warp = self.warp_image(diff2_y, u1_flat, u2_flat)
            diff2_y_warp = torch.reshape(diff2_y_warp, diff2_y.shape)

            diff2_x_sq = torch.pow(diff2_x_warp, 2)
            diff2_y_sq = torch.pow(diff2_y_warp, 2)

            grad = diff2_x_sq + diff2_y_sq + self.GRAD_IS_ZERO

            rho_c = x2_warp - diff2_x_warp * u1 - diff2_y_warp * u2 - x1

            for ii in range(max_iterations):
                rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + self.GRAD_IS_ZERO;

                masks1 = rho < -l_t * grad
                d1_1 = torch.where(masks1, l_t * diff2_x_warp,
                                   torch.zeros_like(diff2_x_warp))
                d2_1 = torch.where(masks1, l_t * diff2_y_warp,
                                   torch.zeros_like(diff2_y_warp))

                masks2 = rho > l_t * grad
                d1_2 = torch.where(masks2, -l_t * diff2_x_warp,
                                   torch.zeros_like(diff2_x_warp))
                d2_2 = torch.where(masks2, -l_t * diff2_y_warp,
                                   torch.zeros_like(diff2_y_warp))

                masks3 = (~masks1) & (~masks2) & (
                            grad > self.GRAD_IS_ZERO)
                d1_3 = torch.where(masks3, -rho / grad * diff2_x_warp,
                                   torch.zeros_like(diff2_x_warp))
                d2_3 = torch.where(masks3, -rho / grad * diff2_y_warp,
                                   torch.zeros_like(diff2_y_warp))

                v1 = d1_1 + d1_2 + d1_3 + u1
                v2 = d2_1 + d2_2 + d2_3 + u2

                u1 = v1 + theta * self.divergence(p11, p12, 'div_p1')
                u2 = v2 + theta * self.divergence(p21, p22, 'div_p2')

                u1x, u1y = self.forward_gradient(u1, 'u1')
                u2x, u2y = self.forward_gradient(u2, 'u2')

                p11 = (p11 + taut * u1x) / (
                        1.0 + taut * torch.sqrt(
                            torch.pow(u1x, 2) \
                            + torch.pow(u1y, 2) \
                            + self.GRAD_IS_ZERO))
                p12 = (p12 + taut * u1y) / (
                        1.0 + taut * torch.sqrt(
                            torch.pow(u1x, 2) \
                            + torch.pow(u1y, 2) \
                            + self.GRAD_IS_ZERO))
                p21 = (p21 + taut * u2x) / (
                        1.0 + taut * torch.sqrt(
                            torch.pow(u2x, 2) \
                            + torch.pow(u2y, 2) \
                            + self.GRAD_IS_ZERO))
                p22 = (p22 + taut * u2y) / (
                        1.0 + taut * torch.sqrt(
                            torch.pow(u2x, 2) \
                            + torch.pow(u2y, 2) \
                            + self.GRAD_IS_ZERO))

        return u1, u2, rho

    def tvnet_flow(self, x1, x2,
                   tau=0.25,  # time step
                   lbda=0.15,  # weight parameter for the data term
                   theta=0.3,  # weight parameter for (u - v)^2
                   warps=5,  # number of warpings per scale
                   zfactor=0.5,  # factor for building the image pyramid
                   max_scales=5,  # maximum number of scales for image piramid
                   max_iterations=5
                   # maximum number of iterations for optimization
                   ):

        for i in range(len(x1.shape)):
            assert x1.shape[i] == x2.shape[i]

        zfactor = np.float32(zfactor)

        height = x1.shape[-2]
        width = x1.shape[-1]

        n_scales = 1 + np.log(np.sqrt(height ** 2 + width ** 2) / 4.0) / np.log(
            1 / zfactor)
        # 7.6
        n_scales = min(n_scales, max_scales)
        # n_scales = 1

        # (N_frames, H, W, C)

        grey_x1 = self.grey_scale_image(x1)
        grey_x2 = self.grey_scale_image(x2)
        norm_imgs = self.normalize_images(grey_x1, grey_x2)

        smooth_x1 = self.gaussian_smooth(norm_imgs[0])
        smooth_x2 = self.gaussian_smooth(norm_imgs[1])
        for ss in range(n_scales - 1, -1, -1):
            down_sample_factor = zfactor ** ss
            down_height, down_width = self.zoom_size(height, width,
                                                     down_sample_factor)

            if ss == n_scales - 1:
                u1 = torch.zeros(smooth_x1.shape[0], 1, down_height, down_width,
                                 dtype=torch.float32)
                u2 = torch.zeros(smooth_x1.shape[0], 1, down_height, down_width,
                                 dtype=torch.float32)

            down_x1 = self.zoom_image(smooth_x1, down_height,
                                      down_width)
            down_x2 = self.zoom_image(smooth_x2, down_height,
                                      down_width)

            u1, u2, rho = self.dual_tvl1_optic_flow(down_x1, down_x2,
                                                    u1, u2,
                                                    tau=tau, lbda=lbda,
                                                    theta=theta,
                                                    warps=warps,
                                                    max_iterations=max_iterations)

            if ss == 0:
                return u1, u2, rho

            up_sample_factor = zfactor ** (ss - 1)
            up_height, up_width = self.zoom_size(height, width,
                                                 up_sample_factor)
            u1 = self.zoom_image(u1, up_height, up_width) / zfactor
            u2 = self.zoom_image(u2, up_height, up_width) / zfactor

    def get_loss(self, x1, x2,
                 tau=0.25,  # time step
                 lbda=0.15,  # weight parameter for the data term
                 theta=0.3,  # weight parameter for (u - v)^2
                 warps=5,  # number of warpings per scale
                 zfactor=0.5,  # factor for building the image pyramid
                 max_scales=5,  # maximum number of scales for image pyramid
                 max_iterations=5
                 # maximum number of iterations for optimization
                 ):

        u1, u2, rho = self.tvnet_flow(x1, x2,
                                      tau=tau, lbda=lbda, theta=theta,
                                      warps=warps,
                                      zfactor=zfactor, max_scales=max_scales,
                                      max_iterations=max_iterations)

        # computing loss
        u1x, u1y = self.forward_gradient(u1, 'u1')
        u2x, u2y = self.forward_gradient(u2, 'u2')

        u1_flat = u1.view(x2.shape[0], 1, x2.shape[1] * x2.shape[2])
        u2_flat = u2.view(x2.shape[0], 1, x2.shape[1] * x2.shape[2])

        x2_warp = self.warp_image(x2, u1_flat, u2_flat)
        x2_warp = x2_warp.view(x2.shape)
        loss = lbda * (x2_warp - x1).abs().mean() + (
                u1x.abs() + u1y.abs() + u2x.abs() + u2y.abs()).mean()
        return loss, u1, u2
