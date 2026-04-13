import math

import torch
from torch import nn

from scene.motion_model_base import MotionModelBaseCZ, expand_tau


def positional_encode_tau(tau, num_freqs):
    encodings = [tau]
    if num_freqs <= 0:
        return tau

    freq_bands = tau.new_tensor([2**idx * math.pi for idx in range(num_freqs)]).view(1, -1)
    angles = tau * freq_bands
    encodings.extend([torch.sin(angles), torch.cos(angles)])
    return torch.cat(encodings, dim=-1)


class MLPMotionModelCZ(MotionModelBaseCZ):
    motion_model_type = "mlp"

    def __init__(self, control_num, hidden_dim=128, num_layers=3, pe_freqs=4):
        super().__init__(control_num)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pe_freqs = pe_freqs

        input_dim = 3 + 3 + 1 + 2 * pe_freqs
        hidden_layers = max(num_layers - 1, 1)
        layers = []
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 3))
        self.mlp = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_layers = [module for module in self.mlp if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        final_linear = linear_layers[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _motion_code(self, device, dtype):
        if self.control_xyz.numel() != 0:
            return self.control_xyz[:, 1, :].to(device=device, dtype=dtype)
        if self.flat_control_xyz.numel() != 0 and self.index_offset.numel() != 0:
            return torch.gather(self.flat_control_xyz, 0, self.index_offset.expand(-1, 3) + 1).to(device=device, dtype=dtype)
        raise RuntimeError("MLPMotionModelCZ is not initialized.")

    def _build_input(self, x0, tau):
        motion_code = self._motion_code(x0.device, x0.dtype)
        tau_features = positional_encode_tau(tau, self.pe_freqs)
        return torch.cat([x0, motion_code, tau_features], dim=-1), motion_code

    def initialize_random(self, fused_point_cloud):
        control_xyz = torch.zeros((fused_point_cloud.shape[0], 2, 3), dtype=fused_point_cloud.dtype, device=fused_point_cloud.device)
        current_control_num = torch.full((fused_point_cloud.shape[0], 1), 2, dtype=torch.int64, device=fused_point_cloud.device)
        self.control_xyz = nn.Parameter(control_xyz.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def initialize_from_trajectory(self, canonical_xyz, dyn_trajectory, deform_spatial_scale):
        del deform_spatial_scale
        final_delta = dyn_trajectory[:, -1, :] - canonical_xyz
        control_xyz = torch.zeros((canonical_xyz.shape[0], 2, 3), dtype=canonical_xyz.dtype, device=canonical_xyz.device)
        control_xyz[:, 1, :] = final_delta
        current_control_num = torch.full((canonical_xyz.shape[0], 1), 2, dtype=torch.int64, device=canonical_xyz.device)
        self.control_xyz = nn.Parameter(control_xyz.requires_grad_(True))
        self.current_control_num = nn.Parameter(current_control_num, requires_grad=False)

    def forward(self, x0, tau, deform_spatial_scale=1.0):
        del deform_spatial_scale
        tau = expand_tau(tau, x0.shape[0], x0.device, x0.dtype)
        mlp_input, motion_code = self._build_input(x0, tau)
        base_delta = tau * motion_code
        residual_delta = self.mlp(mlp_input)
        return x0 + base_delta + residual_delta

    def get_optimizer_param_groups(self, training_args, spatial_lr_scale):
        del spatial_lr_scale
        return [{"params": list(self.mlp.parameters()), "lr": training_args.motion_mlp_lr_init, "name": "motion_model"}]

    def regularization_terms(self, x0, deform_spatial_scale=1.0, smoothness_step=0.05):
        del deform_spatial_scale
        anchor = torch.mean(torch.abs(self.forward(x0, 0.0) - x0))

        step = max(min(float(smoothness_step), 0.49), 1e-3)
        tau = torch.rand((1, 1), device=x0.device, dtype=x0.dtype) * (1.0 - 2.0 * step) + step
        prev_tau = tau - step
        next_tau = tau + step

        x_prev = self.forward(x0, prev_tau)
        x_mid = self.forward(x0, tau)
        x_next = self.forward(x0, next_tau)
        smoothness = torch.mean(torch.abs(x_next - 2.0 * x_mid + x_prev))
        return {"anchor": anchor, "smoothness": smoothness}
