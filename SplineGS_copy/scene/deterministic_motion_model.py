import torch
from torch import nn

from scene.motion_model_base import MotionModelBaseCZ, expand_tau


class DeterministicMotionModelCZ(MotionModelBaseCZ):
    motion_model_type = "deterministic"

    def initialize_random(self, fused_point_cloud):
        control_xyz = torch.zeros((fused_point_cloud.shape[0], 2, 3), dtype=fused_point_cloud.dtype, device=fused_point_cloud.device)
        current_control_num = torch.full((fused_point_cloud.shape[0], 1), 2, dtype=torch.int64, device=fused_point_cloud.device)
        self.control_xyz = nn.Parameter(control_xyz.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        del deform_spatial_scale
        delta_xyz = dyn_trajectory[:, -1, :] - canonical_xyz
        control_xyz = torch.zeros((canonical_xyz.shape[0], 2, 3), dtype=canonical_xyz.dtype, device=canonical_xyz.device)
        control_xyz[:, 1, :] = delta_xyz
        current_control_num = torch.full((canonical_xyz.shape[0], 1), 2, dtype=torch.int64, device=canonical_xyz.device)
        self.control_xyz = nn.Parameter(control_xyz.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def _delta(self, device, dtype):
        if self.control_xyz.numel() != 0:
            return self.control_xyz[:, 1, :].to(device=device, dtype=dtype)
        if self.flat_control_xyz.numel() != 0 and self.index_offset.numel() != 0:
            return torch.gather(self.flat_control_xyz, 0, self.index_offset.expand(-1, 3) + 1).to(device=device, dtype=dtype)
        raise RuntimeError("DeterministicMotionModelCZ is not initialized.")

    def forward(self, x0, tau, deform_spatial_scale=1.0):
        del deform_spatial_scale
        tau = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)
        return x0 + tau * self._delta(x0.device, x0.dtype)
