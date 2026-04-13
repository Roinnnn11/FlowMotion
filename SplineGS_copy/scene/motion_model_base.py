import os

import torch
from torch import nn


class MotionModelBaseCZ(nn.Module):
    motion_model_type = "base"

    def __init__(self, control_num):
        super().__init__()
        self.control_num = control_num
        self.control_xyz = nn.Parameter(torch.empty(0), requires_grad=True)
        self.current_control_num = nn.Parameter(torch.empty(0, dtype=torch.int64), requires_grad=False)
        self.flat_control_xyz = torch.empty(0)
        self.index_offset = torch.empty(0, dtype=torch.long)

    def forward(self, x0, tau, deform_spatial_scale=1.0):
        raise NotImplementedError

    def forward_infer(self, x0, tau, deform_spatial_scale=1.0):
        return self.forward(x0, tau, deform_spatial_scale)

    def velocity(self, x_t, t, tau):
        raise NotImplementedError(f"{self.motion_model_type} does not expose a velocity model.")

    def initialize_random(self, fused_point_cloud):
        raise NotImplementedError

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        raise NotImplementedError

    def get_optimizer_param_groups(self, training_args, spatial_lr_scale):
        del training_args, spatial_lr_scale
        return []

    def regularization_terms(self, x0, deform_spatial_scale=1.0, smoothness_step=0.05):
        del deform_spatial_scale, smoothness_step
        zero = x0.sum() * 0.0
        return {"anchor": zero, "smoothness": zero}

    def save_state(self, path):
        torch.save({"motion_model_type": self.motion_model_type, "state_dict": self.state_dict()}, path)

    def load_state(self, path, map_location="cuda"):
        if not os.path.exists(path):
            return False

        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            checkpoint_motion_type = payload.get("motion_model_type")
            if checkpoint_motion_type is not None and checkpoint_motion_type != self.motion_model_type:
                raise ValueError(
                    f"Motion checkpoint expects '{checkpoint_motion_type}', but current model is '{self.motion_model_type}'."
                )
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        self.load_state_dict(state_dict, strict=False)
        return True

    def flatten_control_point(self):
        if self.control_xyz.numel() == 0 or self.current_control_num.numel() == 0:
            self.flat_control_xyz = torch.empty(0, device=self.control_xyz.device)
            self.index_offset = torch.empty(0, 1, dtype=torch.long, device=self.control_xyz.device)
            return

        current_control_num = self.current_control_num.reshape(-1).long()
        flat_control_point = []
        for i in range(self.control_xyz.shape[0]):
            flat_control_point.append(self.control_xyz[i][: current_control_num[i]])

        self.flat_control_xyz = torch.cat(flat_control_point, dim=0).contiguous()
        if current_control_num.numel() == 1:
            self.index_offset = torch.zeros((1, 1), dtype=torch.long, device=self.control_xyz.device)
        else:
            prefix = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=self.control_xyz.device),
                    torch.cumsum(current_control_num[:-1], dim=0),
                ],
                dim=0,
            )
            self.index_offset = prefix[:, None]


def expand_tau(tau, count, device, dtype):
    if not torch.is_tensor(tau):
        tau = torch.tensor(tau, device=device, dtype=dtype)
    else:
        tau = tau.to(device=device, dtype=dtype)

    if tau.ndim == 0:
        return tau.expand(count, 1)
    if tau.ndim == 1:
        if tau.shape[0] == 1:
            return tau[:, None].expand(count, 1)
        return tau[:, None]
    if tau.ndim == 2:
        return tau
    raise ValueError(f"Unsupported tau shape: {tuple(tau.shape)}")
