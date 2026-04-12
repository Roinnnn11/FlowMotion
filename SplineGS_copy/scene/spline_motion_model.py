import torch
from torch import nn

from scene.motion_model_base import MotionModelBaseCZ


def inverse_cubic_hermite(curves, times, n_pts=5, return_error=False):
    transform_matrix = torch.zeros((times.shape[0], times.shape[1], n_pts), device=curves.device)

    times_scaled = times * (n_pts - 1)
    indices = torch.floor(times_scaled).long()
    indices = torch.clamp(indices, 0, n_pts - 2)
    left_indices = torch.clamp(indices - 1, 0, n_pts - 1)
    right_indices = torch.clamp(indices + 1, 0, n_pts - 1)
    right_right_indices = torch.clamp(indices + 2, 0, n_pts - 1)

    t = times_scaled - indices.float()
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    p1_coef = h00
    p0_coef = torch.zeros_like(h00)
    p2_coef = h01
    p3_coef = torch.zeros_like(h00)

    h10_add_p0 = torch.where(left_indices == indices, 0, -h10 / 2)
    h10_add_p1 = torch.where(left_indices == indices, -h10, 0)
    h10_add_p2 = torch.where(left_indices == indices, h10, h10 / 2)

    h11_add_p1 = torch.where(right_right_indices == right_indices, -h11, -h11 / 2)
    h11_add_p2 = torch.where(right_right_indices == right_indices, h11, 0)
    h11_add_p3 = torch.where(right_right_indices == right_indices, 0, h11 / 2)

    p0_coef = p0_coef + h10_add_p0
    p1_coef = p1_coef + h10_add_p1 + h11_add_p1
    p2_coef = p2_coef + h10_add_p2 + h11_add_p2
    p3_coef = p3_coef + h11_add_p3

    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=left_indices, src=p0_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=indices, src=p1_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(input=transform_matrix, dim=-1, index=right_indices, src=p2_coef, reduce="sum")
    transform_matrix = torch.scatter_reduce(
        input=transform_matrix, dim=-1, index=right_right_indices, src=p3_coef, reduce="sum"
    )

    control_pts = torch.linalg.lstsq(transform_matrix, curves).solution
    if return_error:
        error = torch.dist(control_pts, torch.linalg.pinv(transform_matrix) @ curves)
        return control_pts, error
    return control_pts


def interpolate_cubic_hermite(signal, times, n_points):
    times_scaled = times * (n_points - 1)[:, None]
    indices = torch.floor(times_scaled).long()
    indices = torch.clamp(
        indices, torch.zeros_like(n_points)[:, None].expand(-1, 3, -1), (n_points - 2)[:, None].expand(-1, 3, -1)
    ).long()
    left_indices = torch.clamp(
        indices - 1, torch.zeros_like(n_points)[:, None].expand(-1, 3, -1), (n_points - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_indices = torch.clamp(
        indices + 1, torch.zeros_like(n_points)[:, None].expand(-1, 3, -1), (n_points - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_right_indices = torch.clamp(
        indices + 2, torch.zeros_like(n_points)[:, None].expand(-1, 3, -1), (n_points - 1)[:, None].expand(-1, 3, -1)
    ).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    return interpolation.squeeze(-1)


def interpolate_cubic_hermite_infer(signal, times, n_points, index_offset):
    times_scaled = times * (n_points - 1)
    indices = torch.floor(times_scaled).long()
    indices = torch.clamp(indices, torch.zeros_like(n_points).expand(-1, 3), (n_points - 2).expand(-1, 3)).long()
    left_indices = torch.clamp(indices - 1, torch.zeros_like(n_points).expand(-1, 3), (n_points - 1).expand(-1, 3)).long()
    right_indices = torch.clamp(indices + 1, torch.zeros_like(n_points).expand(-1, 3), (n_points - 1).expand(-1, 3)).long()
    right_right_indices = torch.clamp(
        indices + 2, torch.zeros_like(n_points).expand(-1, 3), (n_points - 1).expand(-1, 3)
    ).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, 0, left_indices + index_offset)
    p1 = torch.gather(signal, 0, indices + index_offset)
    p2 = torch.gather(signal, 0, right_indices + index_offset)
    p3 = torch.gather(signal, 0, right_right_indices + index_offset)

    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    return h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1


class SplineMotionModelCZ(MotionModelBaseCZ):
    motion_model_type = "spline"

    def initialize_random(self, fused_point_cloud):
        mean_xyz = fused_point_cloud.mean(dim=0)
        std_xyz = fused_point_cloud.std(dim=0)
        control_xyz = (
            torch.randn(fused_point_cloud.shape[0], self.control_num, 3, device=fused_point_cloud.device) * std_xyz[None, None]
            + mean_xyz[None, None]
        )
        current_control_num = torch.full(
            (fused_point_cloud.shape[0], 1), self.control_num, dtype=torch.int64, device=fused_point_cloud.device
        )
        self.control_xyz = nn.Parameter(control_xyz.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        del canonical_xyz
        time_step = 1 / (dyn_trajectory.shape[1] - 1.0)
        t_step = torch.arange(0, 1 + time_step, time_step, device=dyn_trajectory.device).float()
        t_step = t_step[None, :, None].expand(dyn_trajectory.shape[0], -1, -1)
        init_control_pts = inverse_cubic_hermite(
            dyn_trajectory / deform_spatial_scale, t_step, n_pts=self.control_num
        )
        current_control_num = torch.full(
            (dyn_trajectory.shape[0], 1), self.control_num, dtype=torch.int64, device=dyn_trajectory.device
        )
        self.control_xyz = nn.Parameter(init_control_pts.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def forward(self, x0, tau, deform_spatial_scale=1.0):
        del x0
        curr_time = torch.as_tensor(tau, device=self.control_xyz.device, dtype=self.control_xyz.dtype)
        if curr_time.ndim > 0:
            curr_time = curr_time.reshape(-1)[0]
        deform_means3d = interpolate_cubic_hermite(
            self.control_xyz.permute(0, 2, 1),
            curr_time[None, None].expand(self.control_xyz.shape[0], 3, 1),
            n_points=self.current_control_num,
        )
        return deform_means3d * deform_spatial_scale

    def forward_infer(self, x0, tau, deform_spatial_scale=1.0):
        if self.flat_control_xyz.numel() == 0 or self.index_offset.numel() == 0:
            return self.forward(x0, tau, deform_spatial_scale)

        curr_time = torch.as_tensor(tau, device=self.flat_control_xyz.device, dtype=self.flat_control_xyz.dtype)
        if curr_time.ndim > 0:
            curr_time = curr_time.reshape(-1)[0]
        deform_means3d = interpolate_cubic_hermite_infer(
            self.flat_control_xyz,
            curr_time[None].expand(x0.shape[0], 3),
            n_points=self.current_control_num,
            index_offset=self.index_offset,
        )
        return deform_means3d * deform_spatial_scale
